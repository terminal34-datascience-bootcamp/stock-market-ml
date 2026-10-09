import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import colab_setup


@pytest.fixture
def local_remote(tmp_path):
    remote = tmp_path / 'remote'
    remote.mkdir()
    subprocess.run(['git', 'init', '-q', '-b', 'main', str(remote)], check=True)
    (remote / 'requirements-colab.txt').write_text('# Fixture: package installation is mocked.\n')
    (remote / 'version.txt').write_text('first\n')
    subprocess.run(['git', '-C', str(remote), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(remote), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'First revision'], check=True)
    return remote


@pytest.fixture
def mock_pip(monkeypatch):
    original = subprocess.check_call
    installs = []

    def run(command, **kwargs):
        if command[1:4] == ['-m', 'pip', 'install']:
            installs.append(command)
            return 0
        return original(command, **kwargs)

    monkeypatch.setattr(colab_setup.subprocess, 'check_call', run)
    return installs


def test_colab_sessions_share_revision_and_exact_versions(tmp_path, local_remote, mock_pip):
    artifacts = tmp_path / 'drive' / 'experiment'
    first = colab_setup.prepare_colab_checkout(artifacts, tmp_path / 'session1', str(local_remote))
    manifest = json.loads((artifacts / 'colab_environment.json').read_text())
    assert manifest['python'] == f'{sys.version_info.major}.{sys.version_info.minor}'
    assert set(manifest['packages']) == set(colab_setup.RUNTIME_PACKAGES)
    (local_remote / 'version.txt').write_text('new main revision\n')
    subprocess.run(['git', '-C', str(local_remote), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(local_remote), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'Newer revision'], check=True)
    second = colab_setup.prepare_colab_checkout(artifacts, tmp_path / 'session2', str(local_remote))
    assert (first / 'version.txt').read_text() == (second / 'version.txt').read_text() == 'first\n'
    assert mock_pip[0][-1].endswith('requirements-colab.txt')
    assert mock_pip[1][-1].endswith('colab-runtime-requirements.txt')
    pins = (artifacts / 'colab-runtime-requirements.txt').read_text()
    assert all(f'{name}=={version}\n' in pins for name, version in manifest['packages'].items())


def test_colab_setup_preserves_local_edits(tmp_path, local_remote, mock_pip):
    artifacts = tmp_path / 'drive'
    project = colab_setup.prepare_colab_checkout(artifacts, tmp_path / 'session', str(local_remote))
    (project / 'version.txt').write_text('my edits\n')
    with pytest.raises(RuntimeError, match='local edits'):
        colab_setup.prepare_colab_checkout(artifacts, project, str(local_remote))
    assert (project / 'version.txt').read_text() == 'my edits\n'


def test_private_token_stays_in_git_child_environment(tmp_path, local_remote, mock_pip, monkeypatch):
    token = 'dummy-unit-test-secret'
    monkeypatch.setenv('GIT_TRACE_CURL', '1')
    env = colab_setup.github_git_environment(token)
    assert 'GIT_TRACE_CURL' not in env
    assert env['GIT_CONFIG_KEY_0'] == 'http.https://github.com/.extraheader'
    assert base64.b64decode(env['GIT_CONFIG_VALUE_0'].split()[-1]).decode() == f'x-access-token:{token}'
    artifacts = tmp_path / 'drive'
    project = colab_setup.prepare_colab_checkout(artifacts, tmp_path / 'session', str(local_remote), token=token)
    for path in [project / '.git' / 'config', artifacts / 'colab_environment.json', artifacts / 'colab-runtime-requirements.txt']:
        assert token not in path.read_text()
        assert env['GIT_CONFIG_VALUE_0'] not in path.read_text()
    assert all(token not in ' '.join(command) for command in mock_pip)


def test_colab_handoff_fails_with_actionable_message(tmp_path):
    colab_setup.require_handoff(tmp_path, 1)
    with pytest.raises(RuntimeError, match='Run Notebook 1 first'):
        colab_setup.require_handoff(tmp_path, 2)
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data' / 'metadata.json').write_text('{}')
    colab_setup.require_handoff(tmp_path, 2)
    with pytest.raises(RuntimeError, match='Run Notebook 2 first'):
        colab_setup.require_handoff(tmp_path, 3)


def test_colab_rejects_python_version_change(tmp_path, local_remote, mock_pip):
    artifacts = tmp_path / 'drive'
    colab_setup.prepare_colab_checkout(artifacts, tmp_path / 'session', str(local_remote))
    path = artifacts / 'colab_environment.json'
    manifest = json.loads(path.read_text())
    manifest['python'] = '0.0'
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match='Python version'):
        colab_setup.prepare_colab_checkout(artifacts, tmp_path / 'new-session', str(local_remote))


def test_setup_requests_restart_if_loaded_package_was_replaced(tmp_path, local_remote, mock_pip, monkeypatch):
    import numpy  # Ensure the test simulates an already-imported Colab package.
    original = colab_setup.importlib.metadata.version
    monkeypatch.setattr(colab_setup.importlib.metadata, 'version',
                        lambda name: '999.0' if name == 'numpy' else original(name))
    with pytest.raises(RuntimeError, match='Restart session'):
        colab_setup.prepare_colab_checkout(tmp_path / 'drive', tmp_path / 'session', str(local_remote))
    assert (tmp_path / 'drive' / 'colab_environment.json').exists()


def test_bootstrap_does_not_import_numerical_libraries_before_install():
    import ast
    tree = ast.parse(Path(colab_setup.__file__).read_text())
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            assert all(alias.name.split('.')[0] not in colab_setup.RUNTIME_PACKAGES.values() for alias in statement.names)
        if isinstance(statement, ast.ImportFrom):
            assert statement.module.split('.')[0] not in colab_setup.RUNTIME_PACKAGES.values()
