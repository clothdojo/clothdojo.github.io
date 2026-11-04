import os
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
import os, signal, socket, sys, time
from pathlib import Path
args = sys.argv[1:]
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
    Path(value('--output')).mkdir()
elif 'data' in args:
    directory = Path(value('--output')) / 'videos' / value('--camera')
    directory.mkdir(parents=True)
    (directory / 'example.mp4').write_bytes(b'fake video for lifecycle testing')
''')
    stub.chmod(0o755)
    monkeypatch.setenv("STUB_MARKER", str(tmp_path / "teacher.pid"))
    monkeypatch.setenv("STUB_REAL_PYTHON", sys.executable)
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
