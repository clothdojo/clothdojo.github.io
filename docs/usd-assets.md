# Converted USD assets and metadata hygiene

The included Piper assets derive from AgileX agx_arm_urdf models; attribution and the upstream MIT notice are retained in the README and third_party/agilex/LICENSE. The geometry layer is referenced by the instance layer and remains necessary for the included composed robot. It has not been removed or replaced with an unverified upstream conversion.

The distributed geometry and robot layers have conversion-origin documentation removed. Geometry, topology, units, authored references, transforms and typed numerical values were compared before and after export with OpenUSD 26.3. The composed Piper scenes, including instance proxies, also retained their property and relationship values. This is a USD equivalence check, not a new GPU rendering or policy benchmark result.

After converting an upstream URDF with your chosen converter, export a fresh sanitized USD layer:

```bash
# usd-core is already pinned in the simulation dependencies.
python scripts/sanitize_usd_metadata.py converted.usd sanitized.usd
```

The tool removes local-path documentation/comments and personal or path-bearing entries in custom metadata. It preserves geometry, reference fields, units, numeric data and copyright/license notices. It refuses to overwrite the source or an existing output. Reopen the export and compare authored and composed content before replacing an asset; inspect source ownership and notices separately.

Re-running conversion requires the exact upstream revision, source meshes, converter version, units, axis convention, joint and mimic settings, collision options, dual-arm installation transforms and material setup. These inputs are not yet fully pinned here, so a fresh conversion is not claimed to reproduce the included asset. Metadata-only cleanup is the narrower operation for this existing verified conversion; the same cleaner can be applied after future conversions.
