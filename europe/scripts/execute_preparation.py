"""Выполнение одного самостоятельного шага подготовки."""
from pathlib import Path
import argparse
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client.manager import KernelManager
from split_preparation_notebooks import TOPICS

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--step', choices=TOPICS)
legacy = dict(zip([
    '--aircraft-diagnostics-only', '--reference-review-only', '--apply-reference-only',
    '--unresolved-diagnostics-only', '--c40-diagnostics-only', '--apply-c40-flags-only',
    '--flight-id-diagnostics-only', '--attempts-diagnostics-only'], list(TOPICS)[2:]))
for flag in legacy:
    parser.add_argument(flag, action='store_true')
args = parser.parse_args()
selected = [step for flag, step in legacy.items() if getattr(args, flag[2:].replace('-', '_'))]
if args.step:
    selected.append(args.step)
if len(selected) != 1:
    parser.error('Укажите ровно один шаг: --step 03_01 ... 03_10')
region = Path(__file__).resolve().parents[1]
path = region / 'notebooks/03_data_preparation' / TOPICS[selected[0]][0]
notebook = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name='python3')
manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
client = NotebookClient(notebook, km=manager, timeout=600,
                        resources={'metadata': {'path': str(path.parent)}}, allow_errors=False)
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
nbformat.validate(notebook)
print('Executed step:', selected[0])
print('Executed code cells:', sum(c.cell_type == 'code' for c in notebook.cells))
