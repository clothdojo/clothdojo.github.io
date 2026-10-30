# ClothDojo project website

Buildless research website. GitHub Pages publishes this directory only; code, tests and benchmark fixtures remain in the same repository.

## Structure

- `index.html`: complete-task generation across distinct garments, observation re-rendering, robot transfer, generation/learning/transfer evaluations, method and practical entry points.
- `site.css` / `site.js`: the shared visual system and homepage media interactions. The homepage does not depend on the archived homepage CSS or player scripts.
- `results.html` / `results.css`: five complete experiment groups. Homepage research charts and all research tables use these same components.
- `explore.html` / `explore.css` / `explorer.js`: the 24-trajectory catalog and interactive 500-garment atlas.
- `assets/experiment-results.json`: machine-readable source metrics.
- `assets/launch/manifest.json`: provenance for the current demonstration set, including trajectories, frame ranges, cameras, renderer settings and encodings.
- `assets/hero.sources.json`: source frames and timing for the linked expert demonstration montage in the explorer.
- `design-preview.html` and `scene-preview.html`: retained earlier design studies, outside the current project narrative.

Serve the website locally with Node.js and a server that supports byte-range video seeking:

```bash
npx --yes http-server@14.1.1 website -a 127.0.0.1 -p 8765 -c-1
```

HTTP byte-range support is required to verify native video seeking and expanded replay in the local preview.

## Demonstration narrative

The homepage opens with the full paper title, aligned with the demonstration below it. Desktop uses three semantic lines; narrow screens wrap the complete title without removing words. Collection counts and the authors' disclosure follow the video. Project details and authors, replay explanations, experiment protocols, the detailed method and setup requirements use native, default-closed `details` sections. Videos, plots, key tables and entry points remain visible; expanding a section keeps its context on the page.

The primary demonstration has two task choices: 400 distinct garments for flattening and 399 for folding. Each starts at an oblique third-person view of one complete robot workcell, then reveals the task's larger collection in one shared 3D scene. Cloth material and environment are uniform so shape and motion carry the comparison. Every workcell replays a different saved autonomous trajectory from its initial state through its selected successful, released terminal state. Shorter executions retain that actual terminal state while others finish. These presentation replays illustrate the saved collection; they do not measure simultaneous generation throughput.

Both task films run for 17 seconds at 25 fps and explicitly use 4× playback. With source states recorded at 50 Hz, output frame `i` reads state `min(8*i, T)`, where `T` is the exported episode's successful terminal index. The maximum selected execution lasts less than 59 seconds, leaving more than two seconds with every garment at its completed state. Desktop and mobile use the same task choices. The earlier four-sample excerpt collage is no longer part of the homepage.

Observation re-rendering directly pairs two appearances of the same saved motion. Both views use matching physical states, robot motion, camera, environment, lighting and frame times; only cloth material changes. A single 25 fps video keeps the views synchronized and needs only one decoder. Desktop places them side by side; mobile stacks them vertically with dedicated label space so native player controls do not cover most of the content. This five-second, 1× excerpt demonstrates observation reuse rather than complete-task success. Both views are rendered RGB observations; “Reference appearance” does not imply an original camera recording.

Robot transfer pairs a source replay with a retargeted kinematic reference at matching timestamps. Both robot identities are labeled, with horizontal desktop and vertical mobile presentations. Changing viewport preserves the comparison time and paused state. The closed-loop policy table is in the evaluation section, separate from replay evidence. Full downstream policy comparisons remain on the results page.

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

The review captures 1440 px desktop and 390/320 px mobile viewports, with decoded images and real video frames. It checks the full title's visibility, task selection, synchronized re-rendering presentation, native dialog cleanup, clipboard copying, exclusive playback, mobile defaults, page overflow, disclosure interactions and shared computed table styles. Each generation film is also captured at 0, 4.75, 8 and 16 seconds on desktop and mobile, covering initial state, the near workcell's completed task, camera movement and the completed collection. Actual timestamps and decoded dimensions are recorded. Results-page screenshots cover all five experiment groups. JPEG screenshots and a report containing the tested commit and browser version are kept on `codex/site-visual-review`.

Inspect screenshots as well as automated results: passing interaction checks is not a visual-quality assessment.

## Sources and hosting

The manuscript source for the reported experiments is `paper/sections/experiments.tex` in the development repository. Original trajectories and raw render outputs are provenance references and are not bundled with the code preview.

The 24-case explorer links originals through `assets/catalog/catalog.json`. Its recordings are saved RGB trajectories, and the atlas is a geometry/material preview. The detailed atlas is loaded only on request.

The public URL is https://clothdojo.github.io/. `.github/workflows/pages.yml` deploys only `website/` from `main`. GitHub Pages must use GitHub Actions as its source.

Source assets retain their licenses; see `THIRD_PARTY_NOTICES.md` in the repository root.
