# ClothDojo project website

Buildless research website. GitHub Pages publishes this directory only; code, tests and benchmark fixtures remain in the same repository.

## Structure

- `index.html`: complete-workcell demonstrations, repeatable generation, observation re-rendering, robot transfer, three experimental comparisons, method and practical entry points.
- `site.css` / `site.js`: the shared visual system and homepage media interactions. The homepage does not depend on the archived homepage CSS or player scripts.
- `results.html` / `results.css`: five complete experiment groups. Homepage research charts and all research tables use these same components.
- `explore.html` / `explore.css` / `explorer.js`: the 24-trajectory catalog and interactive 500-garment atlas.
- `assets/experiment-results.json`: machine-readable source metrics.
- `assets/launch/manifest.json`: provenance for the current demonstration set, including trajectories, frame ranges, cameras, renderer settings and encodings.
- `design-preview.html` and `scene-preview.html`: retained earlier design studies, outside the current project narrative.

Serve the website locally:

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory website
```

## Demonstration narrative

The primary demonstration shows complete robot workcells, with batch replay on desktop and a readable single workcell on mobile. The source recordings are distinct saved autonomous trajectories. Batch playback illustrates the collection; it is not a live parallel-generation session or proof of complete-task success. Close views are reserved for the observation/material comparison and optional detail inspection.

The generated collection uses different garments and recorded trajectories rendered with the same new camera. Desktop visitors can inspect the overview or one trajectory. Mobile starts with a readable single trajectory.

Observation re-rendering uses matching physical states, robot motion, camera and frame times with different cloth materials. Selecting a material preserves the replay time. It changes observations, not physical execution.

Robot transfer shows a source replay and a retargeted kinematic reference. Desktop supports side-by-side comparison; mobile starts with the source and can switch to the target at the same timestamp. The separate closed-loop policy table remains clearly distinguished from replay evidence.

The generated training set has 7,848 trajectories: 3,992 flattening and 3,856 folding, covering 400 and 399 garments respectively. These counts differ from the 500-asset pool and selected demonstration subsets.

## Shared research presentation

All homepage and experiment-page charts are HTML components from `results.css`, with a common 0–100% success scale. Flattening is blue-green; folding is brown. The data source or model is identified by its row label, not a different chart style.

All research tables use `.table-block` with a caption outside the horizontally scrolling `.table-wrap > .data-table`. They share typography, numeric alignment, headings and borders; narrow screens show a scrolling hint for wide tables. Captions remain fully readable while numeric columns scroll. Full numerical results and experimental qualifications are retained. Combined data does not exceed Generated alone; three epochs do not mean equal optimizer steps; pretrained models are distinguished from Diffusion Policy trained from scratch.

The first usage entry is an executable CPU saved-state scoring example. Generation, training and full evaluation workflows disclose their dependence on forthcoming model/data/visual-resource bundles.

## Delivery and playback

Web videos use H.264 with `yuv420p` and fast-start metadata. They preserve the rendered aspect ratio. Only the selected variation is attached to its player. The shared player policy permits one playing video at a time, pauses offscreen/hidden media, and suppresses autoplay for reduced-motion or data-saver preferences. A manually paused visible video remains paused until it leaves view or the visitor chooses another item.

Native controls remain available. Expanded replay loads on explicit request and clears its source when closed. Source switches ignore late autoplay callbacks when their video is offscreen or a dialog is open.

## Browser verification

`.github/workflows/website-visual-review.yml` uses Playwright 1.56.1 and the runner's installed Google Chrome for H.264 playback. It accepts a Git ref and page path, and can review `codex/launch-rebuild` before publication on `main`.

The review captures 1440 px desktop and 390/320 px mobile viewports, with decoded images and real video frames. It checks source selection, material timestamp preservation, native dialog cleanup, clipboard copying, exclusive playback, mobile defaults, page overflow and shared computed table styles. Results-page screenshots cover all five experiment groups. JPEG screenshots and a report containing the tested commit and browser version are kept on `codex/site-visual-review`.

Inspect screenshots as well as automated results: passing interaction checks is not a visual-quality assessment.

## Sources and hosting

The manuscript source for the reported experiments is `paper/sections/experiments.tex` in the development repository. Original trajectories and raw render outputs are provenance references and are not bundled with the code preview.

The 24-case explorer links originals through `assets/catalog/catalog.json`. Its recordings are saved RGB trajectories, and the atlas is a geometry/material preview. The detailed atlas is loaded only on request.

The public URL is https://clothdojo.github.io/. `.github/workflows/pages.yml` deploys only `website/` from `main`. GitHub Pages must use GitHub Actions as its source.

Source assets retain their licenses; see `THIRD_PARTY_NOTICES.md` in the repository root.
