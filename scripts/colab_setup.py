"""Standard-library-only bootstrap, embedded into each notebook before third-party imports."""
from pathlib import Path
import base64
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys

REPOSITORY_URL = 'https://github.com/terminal34-datascience-bootcamp/stock-market-ml.git'
RUNTIME_PACKAGES = {
    'numpy': 'numpy', 'pandas': 'pandas', 'scipy': 'scipy',
    'scikit-learn': 'sklearn', 'xgboost': 'xgboost', 'yfinance': 'yfinance',
    'pyarrow': 'pyarrow', 'matplotlib': 'matplotlib', 'seaborn': 'seaborn', 'joblib': 'joblib',
}


def in_colab():
    try:
        return importlib.util.find_spec('google.colab') is not None
    except (ImportError, ValueError):
        return False


def require_handoff(artifacts, notebook_number):
    required = {
        1: [],
        2: ['data/metadata.json'],
        3: ['frozen_experiment.json', 'competition_evaluation.json',
            'predictions/assignment.parquet', 'predictions/competition.parquet'],
    }
    if notebook_number not in required:
        raise ValueError('Notebook number must be 1, 2, or 3')
    missing = [name for name in required[notebook_number] if not (artifacts / name).is_file()]
    if missing:
        raise RuntimeError(
            f'Missing handoff files in {artifacts}: {missing}. '
            f'Run Notebook {notebook_number - 1} first, with the same EXPERIMENT_NAME and smoke setting.'
        )


def github_git_environment(token=None):
    # Headers live only in child-process environments, never URLs, arguments, or .git/config.
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith('GIT_TRACE') and key != 'GIT_CURL_VERBOSE'}
    environment['GIT_TERMINAL_PROMPT'] = '0'
    if token:
        encoded = base64.b64encode(f'x-access-token:{token}'.encode()).decode()
        environment.update({'GIT_CONFIG_COUNT': '1',
                            'GIT_CONFIG_KEY_0': 'http.https://github.com/.extraheader',
                            'GIT_CONFIG_VALUE_0': f'Authorization: Basic {encoded}'})
    return environment


def prepare_colab_checkout(artifacts, project, repository_url=REPOSITORY_URL, token=None):
    """Use one source revision and resolved package set for all three sessions."""
    artifacts, project = Path(artifacts), Path(project)
    artifacts.mkdir(parents=True, exist_ok=True)
    manifest_path = artifacts / 'colab_environment.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    python_version = f'{sys.version_info.major}.{sys.version_info.minor}'
    if manifest and (manifest['repository_url'] != repository_url or manifest['python'] != python_version):
        raise RuntimeError('This experiment belongs to a different repository or Python version. '
                           'Use the recorded runtime version to reuse its models.')
    git_environment = github_git_environment(token)
    if not project.exists():
        try:
            subprocess.check_call(['git', 'clone', '--quiet', repository_url, str(project)], env=git_environment)
        except subprocess.CalledProcessError as error:
            raise RuntimeError('Cannot clone the repository. For this private repo, add a GITHUB_TOKEN '
                               'Colab secret with Contents: Read access, enable notebook access, and rerun setup.') from error
    if not (project / '.git').is_dir():
        raise RuntimeError(f'{project} exists but is not a Git checkout; choose a fresh Colab session.')
    git = ['git', '-C', str(project)]
    origin = subprocess.check_output(git + ['remote', 'get-url', 'origin'], text=True).strip()
    if origin != repository_url:
        raise RuntimeError(f'Refusing to use an unrelated checkout at {project}')
    if subprocess.check_output(git + ['status', '--porcelain'], text=True).strip():
        raise RuntimeError('The Colab checkout has local edits. Save them before using a fresh runtime; '
                           'setup will not overwrite your work.')
    ref = manifest['commit'] if manifest else 'main'
    subprocess.check_call(git + ['fetch', '--quiet', 'origin', ref], env=git_environment)
    commit = subprocess.check_output(git + ['rev-parse', 'FETCH_HEAD'], text=True).strip()
    subprocess.check_call(git + ['checkout', '--quiet', '--detach', commit])
    requirements = project / 'requirements-colab.txt'
    if not requirements.is_file():
        raise RuntimeError('This GitHub revision does not include Colab support. Publish the updated project first.')
    if manifest:
        requirements = artifacts / 'colab-runtime-requirements.txt'
        requirements.write_text(''.join(f'{name}=={version}\n' for name, version in manifest['packages'].items()))
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', '-r', str(requirements)])
    resolved = {name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES}
    if manifest and resolved != manifest['packages']:
        raise RuntimeError('Installed packages do not match the saved Colab environment')
    if manifest is None:
        manifest = {'repository_url': repository_url, 'commit': commit,
                    'python': python_version, 'packages': resolved}
        temporary = manifest_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        temporary.replace(manifest_path)
        (artifacts / 'colab-runtime-requirements.txt').write_text(
            ''.join(f'{name}=={version}\n' for name, version in resolved.items()))
    changed = [name for name, module in RUNTIME_PACKAGES.items()
               if module in sys.modules and getattr(sys.modules[module], '__version__', resolved[name]) != resolved[name]]
    if changed:
        raise RuntimeError(f'Installed versions changed for already-imported packages: {changed}. '
                           'Choose Runtime > Restart session, then run this setup cell again.')
    print('Pinned GitHub revision:', commit)
    return project


def setup_notebook(notebook_number, experiment_name='competition-v1', colab_smoke=False):
    if in_colab():
        # Authorization is performed by Colab's own Drive prompt, in the user's runtime.
        from google.colab import drive, userdata
        if not experiment_name or experiment_name in ['.', '..'] or any(c in experiment_name for c in '/\\'):
            raise ValueError('EXPERIMENT_NAME must be one folder name')
        drive.mount('/content/drive')
        artifacts = Path('/content/drive/MyDrive/stock-market-ml') / experiment_name / ('smoke' if colab_smoke else 'real')
        require_handoff(artifacts, notebook_number)
        try:
            token = userdata.get('GITHUB_TOKEN')
        except userdata.SecretNotFoundError:
            token = None  # Public forks need no token; private-repo errors explain the setup.
        except userdata.NotebookAccessError as error:
            raise RuntimeError('Enable notebook access for GITHUB_TOKEN in the Colab Secrets panel.') from error
        project = prepare_colab_checkout(artifacts, Path('/content/stock-market-ml'), token=token)
        os.environ['STOCK_ML_ARTIFACTS'] = str(artifacts)
        os.environ['STOCK_ML_SMOKE'] = '1' if colab_smoke else '0'
        os.chdir(project)
    else:
        project = Path.cwd()
        if not (project / 'stockml').is_dir():
            raise RuntimeError('For local execution, open the notebook from the repository root.')
        artifacts = Path(os.environ.get('STOCK_ML_ARTIFACTS', 'artifacts')).resolve()
        require_handoff(artifacts, notebook_number)
    sys.path.insert(0, str(project))
    os.environ.setdefault('MPLCONFIGDIR', str(project / '.cache' / 'matplotlib'))
    Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
    print('Artifact handoff folder:', artifacts)
    return project
