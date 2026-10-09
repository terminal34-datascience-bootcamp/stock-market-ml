"""Execute all three notebooks in separate kernels; save executed copies under artifacts."""
import argparse
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

PROJECT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--smoke', action='store_true', help='Offline synthetic data and eight-tree models; not financial results')
parser.add_argument('--artifacts', type=Path, help='Snapshot/output directory')
args = parser.parse_args()
os.chdir(PROJECT)
folder = (args.artifacts or PROJECT / ('artifacts-smoke' if args.smoke else 'artifacts')).resolve()
folder.mkdir(parents=True, exist_ok=True)
os.environ['STOCK_ML_ARTIFACTS'] = str(folder)
os.environ['STOCK_ML_SMOKE'] = '1' if args.smoke else '0'
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT / '.cache' / 'matplotlib'))
os.environ.setdefault('IPYTHONDIR', str(PROJECT / '.cache' / 'ipython'))
os.environ.setdefault('JUPYTER_RUNTIME_DIR', str(PROJECT / '.cache' / 'jupyter'))
for variable in ['MPLCONFIGDIR', 'IPYTHONDIR', 'JUPYTER_RUNTIME_DIR']:
    Path(os.environ[variable]).mkdir(parents=True, exist_ok=True)
kernel_dir = folder / 'kernels' / 'stockml'
kernel_dir.mkdir(parents=True, exist_ok=True)
(kernel_dir / 'kernel.json').write_text(json.dumps({
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'Stock ML isolated kernel', 'language': 'python',
}))
output = folder / 'executed'
output.mkdir(exist_ok=True)
for path in sorted(PROJECT.glob('0[123]_*.ipynb')):
    print(f'Executing {path.name}', flush=True)
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name='stockml', kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)]))
    client = NotebookClient(notebook, km=manager, timeout=7200, resources={'metadata': {'path': str(PROJECT)}})
    try:
        client.execute(cleanup_kc=True)
    finally:
        nbformat.write(notebook, output / path.name)
    print(f'Completed {path.name}', flush=True)
print(f'Executed notebooks: {output}', flush=True)
