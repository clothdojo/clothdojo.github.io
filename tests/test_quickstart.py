import os
import json
from pathlib import Path
import subprocess
import sys

import pytest

from clothdojo import quickstart


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    for task in ("flatten", "fold"):
        (root / f"models/nocs-teachers/{task}/weights").mkdir(parents=True)
        norm = root / f"models/nocs-teachers/{task}/norm_stats"
        norm.mkdir()
        (norm / "norm_stats.json").write_text("{}")
        states = root / f"benchmark/conditions/nocs-{task}/states"
        states.mkdir(parents=True)
        (states / "meta.json").write_text("{}")
    (root / "benchmark/assets").mkdir()
    stub = tmp_path / "python-stub"
    stub.write_text(f"#!{sys.executable}\n" + '''
import os, signal, socket, sys, time, json
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['STUB_CALL_LOG'], 'a') as log:
    log.write(json.dumps(args) + '\\n')
if args[0] == '-c':
    sys.exit(0)
def value(flag):
    return args[args.index(flag) + 1]
if 'policy' in args:
    marker = Path(os.environ['STUB_MARKER'])
    marker.write_text(str(os.getpid()))
    def finish(*_):
        sys.exit(0)
    signal.signal(signal.SIGINT, finish)
    server = socket.socket()
    server.bind(('127.0.0.1', int(value('--port'))))
    server.listen()
    while True:
        connection, _ = server.accept()
        connection.close()
elif 'rollout' in args:
    if os.environ.get('STUB_FAIL'):
        sys.exit(7)
    count = (int(value('--seen-assets')) + int(value('--unseen-assets'))) * int(value('--limit-per-asset'))
    for i in range(count):
        case = Path(value('--output')) / 'trajectories' / ('example' if count == 1 else f'example-{i}')
        case.mkdir(parents=True)
        (case / 'manifest.json').write_text('{}')
elif 'data' in args:
    directory = Path(value('--output')) / 'videos' / value('--camera')
    directory.mkdir(parents=True)
    for case in (Path(value('--source')) / 'trajectories').iterdir():
        (directory / (case.name + '.mp4')).write_bytes(b'fake video for lifecycle testing')
''')
    stub.chmod(0o755)
    monkeypatch.setenv("STUB_MARKER", str(tmp_path / "teacher.pid"))
    monkeypatch.setenv("STUB_REAL_PYTHON", sys.executable)
    monkeypatch.setenv("STUB_CALL_LOG", str(tmp_path / "calls.jsonl"))
    monkeypatch.setattr(quickstart.sys, "executable", str(stub))
    monkeypatch.setattr(quickstart, "SERVER_START_TIMEOUT", 5)
    return root


def assert_teacher_exited(marker):
    pid = int(marker.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.parametrize("task", ["flatten", "fold"])
def test_run_exports_two_views_and_reaps_its_teacher(project, tmp_path, task):
    result = quickstart.run(task, project)
    assert (result / "review/videos/cam_high/example.mp4").is_file()
    assert (result / "review-low/videos/cam_low/example.mp4").is_file()
    assert_teacher_exited(tmp_path / "teacher.pid")


def test_generation_failure_keeps_logs_and_does_not_stop_unrelated_process(
    project, tmp_path, monkeypatch
):
    monkeypatch.setenv("STUB_FAIL", "1")
    # The stub is only used by the quickstart; launch a real unrelated process.
    real_python = Path(os.environ["STUB_REAL_PYTHON"])
    unrelated = subprocess.Popen([str(real_python), "-c", "import time; time.sleep(30)"])
    try:
        with pytest.raises(RuntimeError, match="status 7"):
            quickstart.run("flatten", project, tmp_path / "failed-run")
        assert unrelated.poll() is None
        assert (tmp_path / "failed-run/logs/generation.log").is_file()
        assert_teacher_exited(tmp_path / "teacher.pid")
    finally:
        unrelated.terminate()
        unrelated.wait()


def test_existing_output_is_preserved_before_starting_a_teacher(project, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing output")
    with pytest.raises(FileExistsError):
        quickstart.run("flatten", project, output)
    assert marker.read_text() == "existing output"
    assert not (tmp_path / "teacher.pid").exists()


def test_config_generates_with_custom_paths_then_replays_without_a_teacher(project, tmp_path):
    (project / 'benchmark').rename(tmp_path / 'external-benchmark')
    (project / 'models/nocs-teachers').rename(tmp_path / 'external-teachers')
    config = project / 'run.yaml'
    config.write_text('''task: fold
benchmark: ../external-benchmark
teachers: ../external-teachers
output: runs/custom
gpu: 3
seen_assets: 1
unseen_assets: 1
states_per_asset: 2
''')
    collection = quickstart.generate(config, project)
    assert collection == project / 'runs/custom/rollout'
    assert len(list((collection / 'trajectories').iterdir())) == 4
    assert not (collection / 'review').exists()
    assert_teacher_exited(tmp_path / 'teacher.pid')
    calls = [json.loads(line) for line in (tmp_path / 'calls.jsonl').read_text().splitlines()]
    server = next(call for call in calls if 'policy' in call)
    rollout = next(call for call in calls if 'rollout' in call)
    assert server[server.index('--device') + 1] == 'cuda:3'
    assert server[server.index('--checkpoint') + 1] == str(tmp_path / 'external-teachers/fold/weights')
    assert server[server.index('--norm-stats') + 1] == str(tmp_path / 'external-teachers/fold/norm_stats')
    assert rollout[rollout.index('--gpus') + 1] == '3'
    before = len(calls)
    quickstart.replay(collection, project, benchmark='../external-benchmark', gpu=2)
    later = [json.loads(line) for line in (tmp_path / 'calls.jsonl').read_text().splitlines()][before:]
    assert len(later) == 2
    assert all('policy' not in call and call[call.index('--device') + 1] == 'cuda:2' for call in later)
    for view, folder in [('cam_high', 'review'), ('cam_low', 'review-low')]:
        assert len(list((collection / folder / 'videos' / view).glob('*.mp4'))) == 4


@pytest.mark.parametrize('settings', ['gpu: true', 'gpu: -1', 'task: typo', 'unknown: 1',
                                     'benchmark: null', 'seen_assets: 0\nunseen_assets: 0',
                                     'states_per_asset: 0', '[]', 'task: ['])
def test_invalid_config_does_not_start_processes(project, tmp_path, settings):
    config = project / 'invalid.yaml'
    config.write_text(settings)
    with pytest.raises(ValueError):
        quickstart.generate(config, project)
    assert not (tmp_path / 'teacher.pid').exists()
