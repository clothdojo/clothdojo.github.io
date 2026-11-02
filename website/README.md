# ClothDojo project website

Buildless research website. GitHub Pages publishes this directory only; code, tests and benchmark fixtures remain in the same repository.

## Structure

- `camera-study.html` / `camera-live.js` / `camera-study.css`: continuous Three.js camera prototype over 48 real recorded episodes. Camera distance and action playback have independent clocks; dragging a loaded scene makes no network requests. Two workcells load first, with the remainder loaded in the background. Cloth uses reduced topology and quantized positions with presentation interpolation; lighting differs from the offline RTX film. Sources, precision and geometry are recorded in `assets/live46/manifest.json`. The research homepage is unchanged. `camera-study-video.html` retains the earlier three-view video prototype using `camera-study.js` and `assets/camera45/`.
- `atlas-white.css` / `assets/atlas32/`: the v31 homepage asset collection shares the pure-white page background. Depth masks restrict clean-plate normalization to the ground; real garment RGB and rendered shadows are retained. The asset section is presentation-only, with no image links or hover zoom.
- `research-preview-v31.html` / `generation-loop.js` / `embedded-hero.css`: the approved research UI with a compact automatic studio film, followed immediately by RGB, assets, results and method. Showcase videos expose only play/pause. Current first-film provenance is `assets/studio44/manifest.json`: a 60-second two-case loop. All peripheral cells share one temporal fade; complete workcells fade at spatial boundaries. Every 2/8/48/384 camera is solved from complete real trajectory geometry across both cases, with centered perspective framing and no image crop or resize. Arrays are arranged per viewport to use the frame. Short camera holds retain active motion; background clocks are scheduled per aspect ratio for motion at first readability. Complete tasks return home before direct case transitions. Older films remain archived.
- `assets/rgb34/manifest.json`: the enlarged 96-episode cam_high collection, with 18 full desktop panels and six mobile panels, sampled from original 50fps recordings for genuine 2× replay at 25fps. The 64-second scrolling loop resets sources only offscreen.
- `rgb-collection.css`: aligns the v31 RGB preview with the shared content width used by the garment collection. Showcase videos use a compact play/pause control without metadata captions; replay speed, recording counts and sources live in expandable explanations. Ambient playback follows the most visible automatic video.
- `research-preview-v30.html` / `research30.css`: white centred-title research UI study, referencing SuperMimic; includes a separate full-width RGB dataset ribbon after model results.
- `assets/rgb30/manifest.json`: original `cam_high` source episodes, hashes, 2× replay timing and periodic 72-second ribbon composition. Desktop shows 24 trajectories; mobile shows 12 with larger panels. Resets happen outside the viewport.
- `index.html`: an automatic task demonstration, model evaluations, method, generation scale, combined domain/robot comparison and practical entry points.
- `site.css` / `studio.css` / `site.js`: the shared visual system and homepage media interactions. The homepage does not depend on the archived homepage CSS or player scripts.
- `results.html` / `results.css`: five complete experiment groups. Homepage research charts and all research tables use these same components.
- `explore.html` / `explore.css` / `explorer.js`: the 24-trajectory catalog and interactive 500-garment atlas.
- `assets/experiment-results.json`: machine-readable source metrics.
- `assets/studio26/manifest.json`: provenance for the current demonstration set, including trajectories, frame ranges, cameras, renderer settings and encodings.
- `scene-preview-v27.html` / `scene-study.css` / `scene-study.js`: the complete homepage layout with a separate scene study. It compares pure-white ground and a dark studio lit only by 384 workcell overhead lights, using matched stills and a 2.4-second, 25 fps moving excerpt. `?scene=dark&theme=dark` opens the dark comparison; environment and page theme can be changed independently. The study does not replace the homepage's full hero film.
- `assets/studio27/manifest.json`: source identities, cotton appearance, exact lighting scope and motion-excerpt audit for that study; 384 independent trajectories on 202 assets, arranged as 192 flattening workspaces on the left and 192 folding workspaces on the right.
- `scene-preview-v28.html` / `fold-fit.css` / `fold-fit.js`: the selected white direction, with a closed hero fitted to one viewport. Desktop uses a 3.33:1 banner and mobile uses a separate 4:3 composition; media is contained without CSS cropping. White ground is composited from RTX matte/shadow-catcher renders using a depth-derived ground mask and uniform clean reference. Only ground pixels change; subject pixels are retained. `assets/studio28/manifest.json` records the audits and 25 fps motion excerpts. The original homepage film remains unchanged.
- `scene-preview-v29.html` / `island-study.js`: a centred opening with two complete, inward-facing workcells, left flattening and right folding. Finite groups expand through 2/8/48/384 cases; new groups enter by a documented whole-group opacity edit after the camera contains the group. Still views and a 2.8-second, 25 fps reveal sample keep a 16-pixel subject safety border in the source renders. The partial sample plays once; it is not the finished continuous 32-second loop. Exact layer/source correspondence and display transforms are recorded in `assets/studio29/manifest.json`.
- `assets/hero.sources.json`: source frames and timing for the linked expert demonstration montage in the explorer.
- `design-directions.html`: three labeled concept compositions. Its AI-generated environment board is a design study, never manipulation evidence. Earlier `design-preview.html` and `scene-preview.html` remain archived.

Serve the website locally with Node.js and a server that supports byte-range video seeking:

```bash
npx --yes http-server@14.1.1 website -a 127.0.0.1 -p 8765 -c-1
```

HTTP byte-range support is required to verify native video seeking and expanded replay in the local preview.

## Presentation and evidence

The homepage retains the complete formal title and opens with a continuous demonstration. The reading order after this visual introduction is model benchmarking, method, generation scale and the asset pool, then a joint visual-randomization/robot-transfer scene and practical code entry points. Other generation, data-value and outcome-quality experiments remain on the full results page.

The opening film automatically shows flattening, folding and a scale reveal. Optional chapter buttons seek within the same video; folding does not require a source switch. The scene contains 96 independent recorded task trajectories across 63 garment assets. Asset identity determines a stable muted color. All cloth meshes use visual Catmull-Clark subdivision; physical task samples are unchanged. The desktop film is 2560×1440, 25 fps and 32 seconds at 2× task replay. Mobile uses a center crop from these independently rendered frames, encoded at 1080×1350. Both use CRF18 H.264 and faststart.

Task motion preserves saved initial-to-verified-successful-released states. After release, the arms smoothly return to configured home in a separately identified scripted kinematic presentation tail while the final cloth state is held. This tail is neither recorded autonomous policy behavior nor a physics evaluation. The camera moves to folding immediately after the flattening return, then reveals the larger scene and returns to its opening pose. Background replay cycles are staggered. The replay arrangement is not measured live generation throughput.

The generated training set contains 7,848 trajectories: 3,992 flattening and 3,856 folding, covering 400 and 399 garments respectively. These counts differ from the 500-asset pool and 96 presentation workspaces. The atlas is freshly rendered on a neutral studio floor from the existing catalog geometry and material assignments.

The combined comparison uses the paper’s actual `sample_visual_layout` sampler with `visual_profile="train"`, base seed 20260911 and appearance indices 1 and 5. No sampled cloth/table/floor/background/light parameters are overridden. Each 12-second replay holds its sampled appearance fixed; the two complete appearances are shown consecutively in one 24-second film. Both PIPER and retargeted UR5e + 2F85 share each scene and camera. The showcase camera and expanded two-workspace arrangement are presentation choices. Mobile preserves the complete two-robot frame in a smaller 1440×810 encoding; the hero has its separate portrait crop. Kinematic reference replay and scripted home return are distinct from closed-loop policy results. Exact layouts, resource IDs and protocol hashes are retained in the source manifest.

The model comparison reports the same four garment–appearance conditions for all four models. π₀.₅, GR00T N1.7 and the internal StarVLA-based policy are fine-tuned; Diffusion Policy is trained from scratch. Pretraining differences are included, so these results are not an architecture-only ablation.

## Shared research presentation

All homepage and experiment-page charts are HTML components from `results.css`, with a common 0–100% success scale. Flattening is blue-green; folding is brown. The data source or model is identified by its row label, not a different chart style.

All research tables use `.table-block` with a caption outside the horizontally scrolling `.table-wrap > .data-table`. They share typography, numeric alignment, headings and borders; narrow screens show a scrolling hint for wide tables. Captions remain fully readable while numeric columns scroll. Full numerical results and experimental qualifications are retained. Combined data does not exceed Generated alone; three epochs do not mean equal optimizer steps; pretrained models are distinguished from Diffusion Policy trained from scratch.

The first usage entry is an executable CPU saved-state scoring example. Generation, training and full evaluation workflows disclose their dependence on forthcoming model/data/visual-resource bundles.

## Delivery and playback

Web videos use H.264 with `yuv420p` and fast-start metadata. They preserve the rendered aspect ratio. Only the selected variation is attached to its player. The shared player policy permits one playing video at a time, pauses offscreen/hidden media, and suppresses autoplay for reduced-motion or data-saver preferences. A manually paused visible video remains paused until it leaves view or the visitor chooses another item.

Videos referenced by the current homepage, results and explorer (including catalog clips and hover previews) run at 25 fps. Frames sample saved physical states or the original RGB recordings; low-rate video is not repeated or interpolated to manufacture a higher motion frame rate. Completed generation trajectories retain their verified terminal state, and short expert demonstrations retain their final frame until the shared phase boundary at 29.64 seconds. Archived design-study media are outside the current page set.

Native controls remain available. Expanded replay loads on explicit request and clears its source when closed. Source switches ignore late autoplay callbacks when their video is offscreen or a dialog is open.

## Browser verification

`lighting-study.html` compares the restored-cloth scene under the previous lighting and a PMREM room environment, fitted VSM shadows and near-field SSAO. Official Three.js r180 `RoomEnvironment`, `SSAOPass` and their direct imports are vendored under the existing MIT license. The environment illuminates surfaces without displaying a background room. SSAO uses 12 samples at half CSS resolution, fading out with camera distance; the far view skips it. Both modes share geometry, materials and action time. This is a real-time approximation, not RTX indirect lighting. `.github/scripts/lighting-study-review.cjs` captures matched near frames and far views, and verifies camera/lighting interaction without additional network requests.

`cloth-study.html` compares the two opening garments using their complete source triangle meshes and the original cotton base-color, normal and roughness maps against the previous simplified cloth, sharing the same action time, camera, robot and lights. All source triangles and face-varying UVs are retained; UV seams duplicate some display vertices. Textures are resized to at most 1024 pixels with source tint/roughness transforms retained. This restores source simulation geometry, not offline subdivision or RTX lighting. Assets and precision are recorded in `assets/cloth48/manifest.json`. Both variants load before comparison is enabled; switching and dragging make no requests. Paused camera movement does not recompute cloth positions/normals. `.github/scripts/cloth-study-review.cjs` captures matched frames and exercises continuous dragging on desktop and mobile. Software WebGL timings do not measure target-device GPU performance.

`.github/workflows/website-visual-review.yml` uses Playwright 1.56.1 and the runner's installed Google Chrome for H.264 playback. It accepts a Git ref and page path, and reviews the website work branches before publication on `main`.

The verification server is pinned to `http-server` 14.1.1 and supports HTTP byte ranges for native video seeking. Before browser checks, a 1,024-byte media request must return HTTP 206 and a matching `Content-Range`.

The review captures 1440 px desktop and 390/320 px mobile viewports, with decoded images and real video frames. It checks the full title's visibility, task selection, synchronized re-rendering presentation, native dialog cleanup, clipboard copying, exclusive playback, mobile defaults, page overflow, disclosure interactions and shared computed table styles. The automatic demonstration is captured at 0, 8.1, 13, 19.2, 26 and 31.8 seconds on desktop and mobile, covering both tasks, their home returns, scale and the loop boundary. Browser checks also verify automatic progression into folding without clicking a task selector. Actual timestamps and decoded dimensions are recorded. Results-page screenshots cover all five experiment groups. JPEG screenshots and a report containing the tested commit and browser version are kept on `codex/site-visual-review`.

Inspect screenshots as well as automated results: passing interaction checks is not a visual-quality assessment.

## Sources and hosting

The manuscript source for the reported experiments is `paper/sections/experiments.tex` in the development repository. Original trajectories and raw render outputs are provenance references and are not bundled with the code preview.

The 24-case explorer links originals through `assets/catalog/catalog.json`. Its recordings are saved RGB trajectories, and the atlas is a geometry/material preview. The detailed atlas is loaded only on request.

The public URL is https://clothdojo.github.io/. `.github/workflows/pages.yml` deploys only `website/` from `main`. GitHub Pages must use GitHub Actions as its source.

Source assets retain their licenses; see `THIRD_PARTY_NOTICES.md` in the repository root.
