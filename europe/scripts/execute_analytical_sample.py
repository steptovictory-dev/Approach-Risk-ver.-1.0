"""Выполнение выбранного шага 04 в отдельном ядре окружения проекта."""
from pathlib import Path
import argparse
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client.manager import KernelManager

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--step', choices=['04_01', '04_02', '04_03', '04_04', '04_05', '04_06', '04_07', '04_08', '04_09', '04_10'], default='04_01')
args = parser.parse_args()
filename = {'04_01': '04_01_task_definition.ipynb', '04_02': '04_02_feature_availability.ipynb',
            '04_03': '04_03_c40_coverage.ipynb', '04_04': '04_04_c40_matching.ipynb',
            '04_05': '04_05_c40_pair_review.ipynb', '04_06': '04_06_independent_check.ipynb',
            '04_07': '04_07_public_pair_check.ipynb', '04_08': '04_08_exclude_ambiguous_pairs.ipynb',
            '04_09': '04_09_c40_selection_audit.ipynb',
            '04_10': '04_10_apply_c40_selection.ipynb'}[args.step]
path = Path(__file__).resolve().parents[1] / 'notebooks/c40' / filename
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
print('Executed:', args.step)
