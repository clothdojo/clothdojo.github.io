# ClothDojo project website

Buildless project website, published from this directory only. For a local preview, serve this directory, not the repository root:

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory website
```

## Page structure

- `index.html`: task-selectable detailed demonstration with three additional garments and dataset counts, generation workflow and collection overview, matched three-material re-rendering, source/target robot comparisons and UR5e policy evaluation, three main experimental comparisons, method, and practical usage guides.
- `results.html`: all five experiment groups, full split results, protocols, quality images, and limitations. Homepage links target each group directly.
- `explore.html`: 24 selected trajectory replays linked to a zoomable 500-garment atlas. Filters, hover previews, modal navigation, keyboard selection, and garment/replay links live here.
- `styles.css`: shared styles and full experiment presentation. `home.css`: homepage composition. `app.js`: shared media dialogs, visibility loading and exclusive playback. `homepage.js`: task and target-robot selection using that shared playback policy. `explore.css` / `explorer.js`: the data explorer.

The pages use Arial/Helvetica and restrained research typography. The homepage has a 1200 px content boundary and light neutral surfaces; full results and the explorer retain a 1160 px boundary. Media retains its encoded aspect ratio; the modal fits its natural dimensions. Videos have native controls, load near visibility, and pause offscreen. Reduced-motion and data-saver preferences disable ambient playback and autoplay after task/robot selection; native playback controls remain available.

The homepage includes three practical entry points: policy evaluation, trajectory/RGB generation, and RGB training. It distinguishes included code and one-garment fixtures from pending model/data/visual-resource downloads. A CPU-only saved-state scoring example is runnable from the released code without a simulator or checkpoint.

### Playback and delivery

`prepare_previews.py` creates 1280- or 1600-pixel-wide, 24 fps H.264/yuv420p previews with fast-start metadata. `assets/preview/manifest.json` records source files and actual byte sizes. Full-resolution originals remain separate and load only when an HD replay is requested. The opening hero compositions are 1440×810, 24 fps and 10 seconds at 1×: 1.71 MB for flattening and 1.46 MB for folding. Only the selected task is loaded. Each composition combines one detailed external-camera excerpt with three independent generated trajectories in one player. `assets/curated/hero-sources.json` records each source. The collection overview below is one precomposed 2,410,651-byte video; the browser does not create 200 players. The paired task-detail preview is 816,446 bytes, compared with its 3,788,709-byte original.

`app.js` permits one playing video at a time, preserves a manually paused preview until it leaves view, and pauses playback when hidden or offscreen. Reduced-motion and browser data-saver preferences disable ambient autoplay. Explicit playback remains available. This limits downloads and concurrent decoding; it is not a substitute for browser frame-rate or device profiling.

## Media curation

`assets/curated/media.json` records primary sources, composition settings, scope, and exclusions. Original supporting assets are retained.

The collection `assets/curated/trajectory-overview.mp4` contains 200 distinct saved RGB trajectories across 169 garments, in a fixed 20×10 grid. The top five rows are flattening and the bottom five are folding. Each tile shows eight seconds at 1×, sampled at 25 fps. Source offsets vary between examples to show different progress points; `trajectory-overview.json` records every source, offset, and frame rate. Composition uses 96×72 source tiles in a 1920×720 master. This is a montage of saved rollouts, not new or concurrent simulation.

The task-detail film pairs two distinct autonomous rollout excerpts using matching external camera, appearance, 1600×1200 source resolution, 25 fps, and 1× playback. Flattening uses source seconds 10–20; folding uses 12–22. They are not one continuous episode and do not establish whole-task success. Source manifests are `assets/flatten-hq.sources.json` and `assets/fold-hq.sources.json`.

The earlier `generation-200.mp4` pullback is retained with its source metadata. It uses 12-second excerpts sampled at 10 Hz. `make_scale_showcase.py` reproduces that archived composition; the homepage uses the newer 25 fps, full-grid overview described above.

Robot films use the matched scene in `outputs/presentation/retarget_appearance_v03/matched/`. Five columns correspond to PIPER, ALOHA2, X5, FR3 Duo, and UR5e + 2F-85; flattening is above folding. The visual layout is shared, and each robot uses the same camera for both tasks. PIPER is recorded motion; the other four are kinematic retargeting references, not autonomous policy evaluations.

The source/target comparison films `assets/curated/compare-*.mp4` rearrange these matched robot clips into four panels: PIPER left, target right, flattening above and folding below. Each 1280×960 film is 10 seconds at 24 fps and 1×; `compare-sources.json` records the crop and source mapping.

`assets/preview/appearance-triptych.mp4` shows three fresh Isaac RTX renders of the same recorded folding trajectory at source seconds 14–19, with identical camera, frame times and robot motion. Only cloth appearance changes. The three materials are patterned fabric, velour velvet and checkered teddy fabric. `assets/curated/appearance-triptych.json` records render and encoding provenance; the full-resolution film is stored beside it.

The cropped HIL example marked ineligible for training and the old appearance-changing clips no longer appear on the homepage. Their files and provenance in `assets/evidence/sources.json` remain available.

The full generated training set has 7,848 trajectories (3,992 flattening / 3,856 folding), covering 400 / 399 garments. These counts differ from the displayed 200-trajectory subset, 24 detailed examples, and 500-asset pool.

## Explorer sources

`assets/catalog/catalog.json` links all 24 selected clips to the atlas. Full videos are 640×480 / 25 fps / 1×; six-second hover previews are 320×240. Posters sample source time 0.5 s.

The atlas copies `outputs/presentation/asset_stage_500_v6_project_floor/asset_master_500_project_floor.png` and `preview_374.jpg`. The source order is `outputs/presentation/asset_stage_500_v1/manifest.json`: 25 columns × 20 rows, canvas 8000×4500, margins (200,112), cells 304×213. The 23 MB detailed image loads only when the atlas opens. It is an existing geometry/material preview, not a new simulation.

## Results and validation

Manuscript metrics and protocols are retained. Combined data does not exceed Generated alone; the page does not assert a verified causal explanation.

The homepage exposes generation ablations, downstream training-data value, and four policy-model comparisons, plus the PIPER/UR5e joint-policy transfer results. The full experiment page retains all five experiment groups, split results, quality examples, and protocols. `assets/experiment-results.json` records the numbers and manuscript table labels; `paper/sections/experiments.tex` is the source. Bars share a 0–100% scale. Policy pretraining differences and the distinction between closed-loop transfer experiments and kinematic showcase replays are disclosed.

Static references, JavaScript syntax, media decoding, source uniqueness, and DOM-model explorer behavior can be checked locally. Actual browser review runs through `.github/workflows/website-visual-review.yml` with Playwright 1.56.1 and the runner's installed Google Chrome (for H.264 support). It captures a 1440 px desktop viewport and 390/320 px mobile viewports, exercises task/robot selection and HD dialogs, checks horizontal page overflow, and verifies exclusive video playback. PNG screenshots and the source-commit/browser-version report are retained on `codex/site-visual-review`. The workflow is manually dispatchable and runs when its tooling changes. Inspect the rendered screenshots separately from the automated checks; passing interactions does not establish visual quality.

The public target is https://clothdojo.github.io/. `.github/workflows/pages.yml` publishes only `website/`; repository code, tests, and benchmark fixtures are not included in the Pages artifact. In repository Settings → Pages, select GitHub Actions as the source; leave Custom domain empty.

Source assets retain their licenses; see `THIRD_PARTY_NOTICES.md` at the repository root. The original manuscript and source render outputs are provenance references and are not bundled with the code preview.
