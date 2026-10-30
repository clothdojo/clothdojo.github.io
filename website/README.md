# ClothDojo project-page preview

Buildless project website, published from this directory only. For a local preview, serve this directory, not the repository root:

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory website
```

## Page structure

- `index.html`: paired external-camera task film, 500-asset preview and training-set counts, method, a large selectable robot replay, and the main downstream-data result. The 200-trajectory pullback is available in an expandable screening; publication details follow the main presentation.
- `results.html`: all five experiment groups, full split results, protocols, quality images, and limitations. Homepage links target each group directly.
- `explore.html`: 24 selected trajectory replays linked to a zoomable 500-garment atlas. Filters, hover previews, modal navigation, keyboard selection, and garment/replay links live here.
- `styles.css`: shared styles and full experiment presentation. `home.css`: homepage composition. `app.js`: media dialogs, visibility loading, and matched robot selection. `explore.css` / `explorer.js`: the data explorer.

Arial/Helvetica, white and light-neutral backgrounds, and a shared 1160 px content boundary establish the page frame. Media retains its encoded aspect ratio; the modal fits its natural dimensions. Videos have native controls, load near visibility, and pause offscreen. Reduced-motion preferences disable automatic ambient playback and hover previews; explicit task/robot playback remains available.

## Media curation

`assets/curated/media.json` records primary sources, composition settings, scope, and exclusions. Original supporting assets are retained.

The first film pairs two distinct autonomous rollout excerpts using matching external camera, appearance, 1600×1200 source resolution, 25 fps, and 1× playback. Flattening uses source seconds 10–20; folding uses 12–22. They are not one continuous episode and do not establish whole-task success. Source manifests are `assets/flatten-hq.sources.json` and `assets/fold-hq.sources.json`.

The scale film uses 200 distinct source trajectories covering 169 garments: 100 flattening and 100 folding. It shows 12 seconds at 1×, sampled at 10 Hz and encoded at 25 fps. The camera pulls back over a 20×10 grid; these are saved recordings, not concurrent simulation. `make_scale_showcase.py` reproduces composition from the 320×240/10 Hz CPU excerpts under `outputs/website_media_20261030/reorganized/thumb-clips/`. Its source manifest is `outputs/presentation/website_rgb_pool_20260928/manifest.json`.

Robot films use the matched scene in `outputs/presentation/retarget_appearance_v03/matched/`. Five columns correspond to PIPER, ALOHA2, X5, FR3 Duo, and UR5e + 2F-85; flattening is above folding. The visual layout is shared, and each robot uses the same camera for both tasks. PIPER is recorded motion; the other four are kinematic retargeting references, not autonomous policy evaluations.

The cropped HIL example marked ineligible for training and the old appearance-changing clips no longer appear on the homepage. Their files and provenance in `assets/evidence/sources.json` remain available.

The full generated training set has 7,848 trajectories (3,992 flattening / 3,856 folding), covering 400 / 399 garments. These counts differ from the displayed 200-trajectory subset, 24 detailed examples, and 500-asset pool.

## Explorer sources

`assets/catalog/catalog.json` links all 24 selected clips to the atlas. Full videos are 640×480 / 25 fps / 1×; six-second hover previews are 320×240. Posters sample source time 0.5 s.

The atlas copies `outputs/presentation/asset_stage_500_v6_project_floor/asset_master_500_project_floor.png` and `preview_374.jpg`. The source order is `outputs/presentation/asset_stage_500_v1/manifest.json`: 25 columns × 20 rows, canvas 8000×4500, margins (200,112), cells 304×213. The 23 MB detailed image loads only when the atlas opens. It is an existing geometry/material preview, not a new simulation.

## Results and validation

Manuscript metrics and protocols are retained. Combined data does not exceed Generated alone; the page does not assert a verified causal explanation.

The full experiment page exposes cumulative generation ablations, seen/unseen garment splits, Human-to-Generated gains in all four evaluation conditions, four policy-model comparisons, source-linked terminal-state quality images, and the supplementary PIPER/UR5e task-composition results. Full model results and training-set sizes remain expandable. `assets/experiment-results.json` records the displayed numbers and manuscript table labels; `paper/sections/experiments.tex` is the source. Bars share a 0–100% scale. Policy pretraining differences and the distinction between closed-loop transfer experiments and kinematic showcase replays are disclosed. The homepage summarizes the same results and retains links to every experiment group.

Static references, JavaScript syntax, media decoding, source uniqueness, and DOM-model explorer behavior can be checked locally. A DOM model is not browser layout verification. Browser screenshot QA is unavailable in this session.

The public target is https://clothdojo.github.io/. `.github/workflows/pages.yml` publishes only `website/`; repository code, tests, and benchmark fixtures are not included in the Pages artifact. In repository Settings → Pages, select GitHub Actions as the source; leave Custom domain empty.

Source assets retain their licenses; see `THIRD_PARTY_NOTICES.md` at the repository root. The original manuscript and source render outputs are provenance references and are not bundled with the code preview.
