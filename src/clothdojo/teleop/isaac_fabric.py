from __future__ import annotations

import numpy as np


class NewtonFabricBridge:
    """Project Newton state into Isaac's reliable cloth and rigid paths.

    Cloth points are published through Fabric's CPU attribute API on every GPU,
    without retaining pointers into renderer-owned GPU memory across frames.
    Isaac 5's RTX delegate does not reliably publish runtime
    local-matrix edits for the instanceable ALOHA visual meshes, so the small
    rigid topology is authored through standard USD xforms.
    """

    def __init__(self, model, usd_context, *, device: str, root_path: str = "/World") -> None:
        from usdrt import Sdf, Usd, Vt

        if not str(device).startswith("cuda:"):
            raise ValueError("teleop Fabric requires a CUDA device")
        usd_stage = usd_context.get_stage()
        if usd_stage is None:
            raise RuntimeError("Isaac USD context has no stage")
        self._stage = Usd.Stage.Attach(usd_context.get_stage_id())
        self._rigid_writer = _UsdRigidBodyWriter(
            usd_stage, model.body_label, root_path=root_path
        )
        # SelectPrims searches Fabric, not the source USD stage. Traversing the
        # attached USDRT stage is the documented population step.
        for _prim in self._stage.Traverse():
            pass

        self._cloth_attr = None
        self._cloth_vt_type = None
        self._cloth_point_count = 0
        cloth = self._stage.GetPrimAtPath(Sdf.Path(f"{root_path.rstrip('/')}/Cloth"))
        if cloth and cloth.IsValid():
            # OmniHydra only consumes runtime Fabric point edits from meshes
            # carrying this tag.  Without it RTX keeps rendering the USD
            # points authored when the stage was opened.
            if not cloth.HasAttribute("Deformable"):
                cloth.CreateAttribute(
                    "Deformable", Sdf.ValueTypeNames.PrimTypeTag, True
                )
            attr = cloth.GetAttribute("points")
            if attr and attr.IsValid():
                vt_points = attr.Get()
                self._cloth_point_count = len(vt_points)
                self._cloth_attr = attr
                if self._cloth_point_count != int(model.particle_count):
                    raise RuntimeError(
                        "Isaac Fabric cloth vertex count does not match Newton particles: "
                        f"{self._cloth_point_count} != {int(model.particle_count)}"
                    )
                self._cloth_vt_type = Vt.Vec3fArray
        if int(model.particle_count) > 0 and self._cloth_attr is None:
            raise RuntimeError("Isaac Fabric did not map the combined cloth mesh")
        print(
            f"Isaac replay bridge ready: rigid={int(model.body_count)}, "
            f"cloth={self._cloth_attr is not None}, sync=rigid-usd+cloth-fabric-cpu",
            flush=True,
        )

    def update(self, state) -> None:
        self.update_arrays(state.particle_q.numpy(), state.body_q.numpy())

    def update_arrays(self, particle_q: np.ndarray, body_q: np.ndarray) -> None:
        if self._cloth_attr is not None:
            # Set publishes through Fabric's ownership API rather than
            # asynchronously writing a borrowed GPU pointer.
            points = np.asarray(particle_q, dtype=np.float32)
            if points.shape != (self._cloth_point_count, 3):
                raise ValueError("particle_q must have shape [particle, 3]")
            if not self._cloth_attr.Set(self._cloth_vt_type(points)):
                raise RuntimeError("Isaac Fabric rejected the cloth point update")

        self._rigid_writer.update(body_q)


class _UsdRigidBodyWriter:
    """Write Newton environment poses through the actual USD transform hierarchy.

    Input poses are relative to root_path, which places the environment in USD.
    USD parents may be installation groups or siblings of the physical joint
    parent. Keep authored body stretch and descendant transforms, including
    visual meshes and cameras, while changing only the body's rigid world pose.
    """

    def __init__(self, stage, labels, *, root_path: str = "/World") -> None:
        from pxr import Gf, UsdGeom

        self._Gf = Gf
        self._cache = UsdGeom.XformCache()
        paths = _find_render_body_paths(stage, labels, root_path=root_path)
        self._root = stage.GetPrimAtPath(root_path)
        missing = [str(label) for index, label in enumerate(labels) if index not in paths]
        if missing:
            raise RuntimeError(f"Isaac did not map Newton rigid bodies: {missing}")
        if len(set(paths.values())) != len(paths):
            raise RuntimeError("Isaac mapped multiple Newton bodies to the same USD prim")
        self._bodies = []
        # Parents must be updated first even when the Newton body order differs.
        for index, path in sorted(paths.items(), key=lambda item: item[1].count("/")):
            prim = stage.GetPrimAtPath(path)
            xform = UsdGeom.Xformable(prim)
            world = self._cache.GetLocalToWorldTransform(prim)
            rigid = Gf.Matrix4d(1.0)
            rigid.SetRotateOnly(Gf.Transform(world).GetRotation())
            rigid.SetTranslateOnly(world.ExtractTranslation())
            stretch = world * rigid.GetInverse()
            local = xform.GetLocalTransformation()
            reset = xform.GetResetXformStack()
            op = xform.MakeMatrixXform()
            op.Set(local)
            xform.SetResetXformStack(reset)
            self._bodies.append((index, prim, op, stretch, reset))

    def update(self, body_q: np.ndarray) -> None:
        world = np.asarray(body_q, dtype=np.float64)
        if world.shape != (len(self._bodies), 7):
            raise ValueError("body_q must have shape [body, 7]")
        norms = np.linalg.norm(world[:, 3:7], axis=1)
        if not np.isfinite(world).all() or np.any(norms <= 1.0e-12):
            raise ValueError("body_q contains an invalid rigid transform")
        self._cache.Clear()
        root_world = self._cache.GetLocalToWorldTransform(self._root)
        for index, prim, op, stretch, reset in self._bodies:
            pose = world[index]
            quaternion = pose[3:7] / norms[index]
            rigid = self._Gf.Matrix4d(1.0)
            rigid.SetRotateOnly(
                self._Gf.Quatd(float(quaternion[3]), self._Gf.Vec3d(*quaternion[:3]))
            )
            rigid.SetTranslateOnly(self._Gf.Vec3d(*pose[:3]))
            # Newton poses are local to this environment, including when a
            # robot body resets its USD transform stack.
            desired = stretch * rigid * root_world
            parent = (
                self._Gf.Matrix4d(1.0)
                if reset else self._cache.GetParentToWorldTransform(prim)
            )
            op.Set(desired * parent.GetInverse())
            # XformCache does not observe authored changes. Descendants must see
            # their parent's just-written world transform, not the previous frame.
            self._cache.Clear()


def _find_render_body_paths(stage, labels, *, root_path: str = "/World") -> dict[int, str]:
    """Use workstation body labels, retaining name lookup for untagged Piper."""
    from pxr import Usd, UsdGeom

    root = stage.GetPrimAtPath(root_path)
    if not root:
        raise RuntimeError(f"Isaac render root does not exist: {root_path}")
    root_path = str(root.GetPath())
    rigid_prefix = f"{root_path}/RigidObjects/"
    robot_prefix = f"{root_path}/Robot/"
    candidates: dict[tuple[str | None, str], list[str]] = {}
    rigid_objects: dict[str, str] = {}
    authored_labels: dict[str, str] = {}
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Xformable):
            continue
        path = prim.GetPath().pathString
        if path.startswith(rigid_prefix) and "/" not in path[len(rigid_prefix):]:
            rigid_objects[prim.GetName()] = path
            continue
        if not path.startswith(robot_prefix):
            continue
        label = prim.GetAttribute("clothdojo:bodyLabel")
        if label and label.HasAuthoredValue():
            name = str(label.Get()).strip("/")
            if name in authored_labels:
                raise RuntimeError(f"duplicate Isaac rigid-body label {name!r}")
            authored_labels[name] = path
        parts = path[len(robot_prefix):].split("/")
        side = "left" if "left" in parts else "right" if "right" in parts else None
        candidates.setdefault((side, prim.GetName()), []).append(path)

    result = {}
    for index, label in enumerate(labels):
        authored_path = authored_labels.get(str(label).strip("/"))
        if authored_path is not None:
            result[index] = authored_path
            continue
        parts = str(label).strip("/").split("/")
        rigid_path = rigid_objects.get(parts[-1])
        if rigid_path is not None:
            result[index] = rigid_path
            continue
        side = "left" if "left" in parts else "right" if "right" in parts else None
        matches = candidates.get((side, parts[-1]), ())
        if matches:
            result[index] = min(matches, key=lambda value: value.count("/"))
    return result
