# CPU tests

Run from the release root after installing the base dependencies and `requirements/testing.txt`. No simulator, CUDA, model weight or internal service is required. Tests use synthetic states and the included frozen folding examples.

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src
export ARTIFACT_DIR="$(mktemp -d)"
export COVERAGE_FILE="$ARTIFACT_DIR/coverage.db"
python -B -m coverage run --source=clothdojo -m pytest tests -p no:cacheprovider --basetemp="$ARTIFACT_DIR/tmp" --junitxml="$ARTIFACT_DIR/junit.xml"
python -B -m coverage report
python -B -m coverage json -o "$ARTIFACT_DIR/coverage.json"
python -B -m coverage html -d "$ARTIFACT_DIR/htmlcov"
```

Report full-package coverage and focused module coverage separately. CPU tests and the single offline scoring entry do not establish GPU training, rendering, headset operation or paper success rates. The exact dependency versions and measured results accompany the release validation report outside the source distribution.

Four metadata-hygiene cases additionally run when `usd-core==26.3` is installed; without this optional simulation dependency, that module is explicitly skipped.
