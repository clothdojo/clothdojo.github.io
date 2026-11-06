import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def installer(tmp_path):
    root = tmp_path / "project with spaces"
    (root / "scripts").mkdir(parents=True)
    shutil.copyfile(Path(__file__).resolve().parents[1] / "scripts/setup.sh",
                    root / "scripts/setup.sh")
    (root / "patches").mkdir()
    (root / "patches/rlinf-recap-value.patch").write_text("test patch\n")
    tools = tmp_path / "tools"
    tools.mkdir()
    patch = tools / "patch"
    patch.write_text("#!/bin/sh\nexit 0\n")
    patch.chmod(0o755)
    git = tools / "git"
    git.write_text(f"#!{sys.executable}\n" +
                   "import json, os, sys\n" +
                   "with open(os.environ['SETUP_CALL_LOG'], 'a') as f: " +
                   "f.write(json.dumps({'executable': 'git', 'args': sys.argv[1:]}) + '\\n')\n")
    git.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    bootstrap = tools / "bootstrap"
    bootstrap.write_text(f"#!{sys.executable}\n" + '''
import json, os, shutil, sys
from pathlib import Path
root = Path.cwd()
args = sys.argv[1:]
with open(os.environ['SETUP_CALL_LOG'], 'a') as stream:
    stream.write(json.dumps({'executable': str(Path(sys.argv[0]).resolve()), 'args': args}) + '\\n')
if args[:2] == ['-m', 'venv']:
    dest = root / args[2] / 'bin/python'
    dest.parent.mkdir(parents=True)
    shutil.copyfile(__file__, dest)
    dest.chmod(0o755)
elif args[:3] == ['-m', 'pip', 'install']:
    if '-r' in args and os.environ.get('FAIL_BASE'):
        sys.exit(7)
    if '-e' in args:
        cli = root / '.venv/bin/clothdojo'
        cli.write_text('#!/bin/sh\\ntest -f .fake-base-ready\\n')
        cli.chmod(0o755)
        (root / '.fake-base-ready').touch()
    if 'isaacsim==5.0.0' in args:
        if os.environ.get('FAIL_ISAAC'):
            sys.exit(8)
        (root / '.fake-isaac-installed').touch()
elif args[0] == '-c':
    if 'getsitepackages' in args[1]:
        print(root / '.venv/site-packages')
    if '_require_isaac5_runtime' in args[1]:
        assert (root / '.fake-isaac-installed').exists()
''')
    bootstrap.chmod(0o755)
    env = dict(os.environ, PYTHON=str(bootstrap), PATH=f"{tools}:{os.environ['PATH']}",
               SETUP_CALL_LOG=str(log))

    def invoke(*arguments, existing=False, compatible=True, **extra_env):
        if existing:
            binary = root / ".venv/bin/python"
            binary.parent.mkdir(parents=True)
            shutil.copyfile(bootstrap, binary)
            binary.chmod(0o755)
            cli = binary.parent / "clothdojo"
            cli.write_text("#!/bin/sh\ntest -f .fake-base-ready\n")
            cli.chmod(0o755)
            (root / ".venv/keep-me").write_text("existing environment")
            if compatible:
                (root / ".fake-base-ready").touch()
        result = subprocess.run(["bash", "scripts/setup.sh", *arguments], cwd=root,
                                env=dict(env, **extra_env), capture_output=True, text=True)
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return root, result, calls

    return invoke


def pip_calls(calls):
    return [call for call in calls if call['args'][:3] == ['-m', 'pip', 'install']]


def test_xpolicy_addon_preserves_environment_and_pins_upstream_submodule(installer):
    root, result, calls = installer('--xpolicy', existing=True)
    assert result.returncode == 0, result.stderr
    assert (root / '.venv/keep-me').read_text() == 'existing environment'
    assert any(call['args'] == ['submodule', 'update', '--init', 'XPolicyLab'] for call in calls)
    installs = pip_calls(calls)
    assert len(installs) == 2
    assert 'msgpack-numpy==0.4.8' in installs[0]['args']
    assert installs[1]['args'][-3:] == ['--no-deps', '-e', 'XPolicyLab']
    assert not any('-r' in call['args'] or 'isaacsim==5.0.0' in call['args'] for call in installs)


@pytest.mark.parametrize("rgb", [False, True])
def test_fresh_install_builds_base_and_optionally_adds_isaac(installer, rgb):
    root, result, calls = installer(*(['--rgb'] if rgb else []))
    assert result.returncode == 0, result.stderr
    installs = pip_calls(calls)
    assert '-r' in installs[0]['args']
    assert '-e' in installs[1]['args']
    assert len(installs) == (3 if rgb else 2)
    assert all(Path(call['executable']) == root / '.venv/bin/python' for call in installs)
    if rgb:
        assert '--no-deps' in installs[2]['args']
        assert 'isaacsim==5.0.0' in installs[2]['args']
        assert (root / '.fake-isaac-installed').exists()
    else:
        assert not (root / '.fake-isaac-installed').exists()


def test_rgb_addon_preserves_ready_environment_and_skips_base_install(installer):
    root, result, calls = installer('--rgb', existing=True)
    assert result.returncode == 0, result.stderr
    assert (root / '.venv/keep-me').read_text() == 'existing environment'
    assert not any(call['args'][:2] == ['-m', 'venv'] for call in calls)
    installs = pip_calls(calls)
    assert len(installs) == 1
    assert '--no-deps' in installs[0]['args']
    assert 'isaacsim==5.0.0' in installs[0]['args']
    assert 'adding Isaac only' in result.stdout
    _, repeated, calls = installer('--rgb')
    assert repeated.returncode == 0, repeated.stderr
    assert len(pip_calls(calls)) == 2
    assert all('isaacsim==5.0.0' in call['args'] for call in pip_calls(calls))
    assert (root / '.venv/keep-me').read_text() == 'existing environment'


def test_incomplete_environment_gets_base_before_isaac(installer):
    root, result, calls = installer('--rgb', existing=True, compatible=False)
    assert result.returncode == 0, result.stderr
    installs = pip_calls(calls)
    assert '-r' in installs[0]['args']
    assert 'isaacsim==5.0.0' in installs[-1]['args']
    assert (root / '.venv/keep-me').exists()


@pytest.mark.parametrize('failure', ['FAIL_BASE', 'FAIL_ISAAC'])
def test_install_failure_does_not_report_ready(installer, failure):
    _, result, calls = installer('--rgb', **{failure: '1'})
    assert result.returncode != 0
    assert 'Ready.' not in result.stdout
    if failure == 'FAIL_BASE':
        assert not any('isaacsim==5.0.0' in call['args'] for call in calls)


@pytest.mark.parametrize('argument, code', [('--help', 0), ('--unknown', 2)])
def test_argument_handling_does_not_create_an_environment(installer, argument, code):
    root, result, calls = installer(argument)
    assert result.returncode == code
    assert not (root / '.venv').exists()
    assert not calls
