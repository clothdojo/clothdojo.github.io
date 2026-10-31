# ClothDojo project website

Buildless research website. GitHub Pages publishes this directory only; code, tests and benchmark fixtures remain in the same repository.

## Structure

- `index.html`: complete-task generation across distinct garments, observation re-rendering, robot transfer, generation/learning/transfer evaluations, method and practical entry points.
- `site.css` / `studio.css` / `site.js`: the shared visual system and homepage media interactions. The homepage does not depend on the archived homepage CSS or player scripts.
- `results.html` / `results.css`: five complete experiment groups. Homepage research charts and all research tables use these same components.
- `explore.html` / `explore.css` / `explorer.js`: the 24-trajectory catalog and interactive 500-garment atlas.
- `assets/experiment-results.json`: machine-readable source metrics.
- `assets/studio/manifest.json`: provenance for the current demonstration set, including trajectories, frame ranges, cameras, renderer settings and encodings.
- `assets/hero.sources.json`: source frames and timing for the linked expert demonstration montage in the explorer.
- `design-directions.html`: three labeled concept compositions. Its AI-generated environment board is a design study, never manipulation evidence. Earlier `design-preview.html` and `scene-preview.html` remain archived.

Serve the website locally with Node.js and a server that supports byte-range video seeking:

```bash
npx --yes http-server@14.1.1 website -a 127.0.0.1 -p 8765 -c-1
```

HTTP byte-range support is required to verify native video seeking and expanded replay in the local preview.

## Demonstration narrative

The homepage places a large project wordmark beside the complete paper title. It follows a continuous studio scene with garment diversity, controlled observation re-rendering, robot transfer, experiments, method and practical entry points. Explanations and authors remain in default-closed native disclosures.

The primary demonstration separates flattening and folding. Flattening shows 54 independent trajectories on 54 garments; folding shows 54 trajectories on 52 garments, including three saved initial configurations of one garment. Different configurations do not establish multiple solutions from an identical state. Uniform cloth material makes geometry and motion legible; the separate asset-pool atlas illustrates the broader garment collection.

Both task films run for 34 seconds at 25 actual rendered fps and explicitly use 2× playback. Output frame `i` in workspace `j` reads state `min(4*((i+phase_j)%850), T_j)`, where `T_j` is the selected successful, released endpoint. The foreground case begins at its actual initial state and completes while the camera holds for 13 seconds; the camera then moves laterally through the studio and returns to its opening pose. Shorter actions retain their actual terminal state. Staggered cycles are saved replays, not live generation or a generation-throughput measurement.

Observation re-rendering shows two full workspaces in one continuous 3D scene and one camera. They replay matching physical states; the second garment uses a different material. Robot transfer similarly shows PIPER and UR5e + 2F85 together in one 3D scene at matching timestamps. These are real state replays, not composited video panels. Both clips run for 12 seconds at 25 fps, with 2× replay and a terminal hold. Desktop uses 1920×1080; a separately rendered 1080×1350 camera and spatial arrangement serves narrow screens. Changing viewport preserves replay time and paused state. Kinematic retargeted reference replay is distinct from closed-loop policy evaluation in the results section.

The generated training set has 7,848 trajectories: 3,992 flattening and 3,856 folding, covering 400 and 399 garments respectively. These counts differ from the 500-asset pool and selected demonstration subsets.

## Shared research presentation

All homepage and experiment-page charts are HTML components from `results.css`, with a common 0–100% success scale. Flattening is blue-green; folding is brown. The data source or model is identified by its row label, not a different chart style.

All research tables use `.table-block` with a caption outside the horizontally scrolling `.table-wrap > .data-table`. They share typography, numeric alignment, headings and borders; narrow screens show a scrolling hint for wide tables. Captions remain fully readable while numeric columns scroll. Full numerical results and experimental qualifications are retained. Combined data does not exceed Generated alone; three epochs do not mean equal optimizer steps; pretrained models are distinguished from Diffusion Policy trained from scratch.

The first usage entry is an executable CPU saved-state scoring example. Generation, training and full evaluation workflows disclose their dependence on forthcoming model/data/visual-resource bundles.

## Delivery and playback

Web videos use H.264 with `yuv420p` and fast-start metadata. They preserve the rendered aspect ratio. Only the selected variation is attached to its player. The shared player policy permits one playing video at a time, pauses offscreen/hidden media, and suppresses autoplay for reduced-motion or data-saver preferences. A manually paused visible video remains paused until it leaves view or the visitor chooses another item.

Videos referenced by the current homepage, results and explorer (including catalog clips and hover previews) run at 25 fps. Frames sample saved physical states or the original RGB recordings; low-rate video is not repeated or interpolated to manufacture a higher motion frame rate. Completed generation trajectories retain their verified terminal state, and short expert demonstrations retain their final frame until the shared phase boundary at 29.64 seconds. Archived design-study media are outside the current page set.

Native controls remain available. Expanded replay loads on explicit request and clears its source when closed. Source switches ignore late autoplay callbacks when their video is offscreen or a dialog is open.

## Browser verification

`.github/workflows/website-visual-review.yml` uses Playwright 1.56.1 and the runner's installed Google Chrome for H.264 playback. It accepts a Git ref and page path, and reviews the website work branches before publication on `main`.

The verification server is pinned to `http-server` 14.1.1 and supports HTTP byte ranges for native video seeking. Before browser checks, a 1,024-byte media request must return HTTP 206 and a matching `Content-Range`.

The review captures 1440 px desktop and 390/320 px mobile viewports, with decoded images and real video frames. It checks the full title's visibility, task selection, synchronized re-rendering presentation, native dialog cleanup, clipboard copying, exclusive playback, mobile defaults, page overflow, disclosure interactions and shared computed table styles. Each generation film is also captured at 0, 12.8, 22 and 33.8 seconds on desktop and mobile, covering initial state, the near workcell's completed task, camera movement and the loop boundary. Actual timestamps and decoded dimensions are recorded. Results-page screenshots cover all five experiment groups. JPEG screenshots and a report containing the tested commit and browser version are kept on `codex/site-visual-review`.

Inspect screenshots as well as automated results: passing interaction checks is not a visual-quality assessment.

## Sources and hosting

The manuscript source for the reported experiments is `paper/sections/experiments.tex` in the development repository. Original trajectories and raw render outputs are provenance references and are not bundled with the code preview.

The 24-case explorer links originals through `assets/catalog/catalog.json`. Its recordings are saved RGB trajectories, and the atlas is a geometry/material preview. The detailed atlas is loaded only on request.

The public URL is https://clothdojo.github.io/. `.github/workflows/pages.yml` deploys only `website/` from `main`. GitHub Pages must use GitHub Actions as its source.

Source assets retain their licenses; see `THIRD_PARTY_NOTICES.md` in the repository root.
