"""Выполнение выбранного шага 06 в отдельном ядре окружения проекта."""
from pathlib import Path
import argparse
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client.manager import KernelManager

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--step',choices=['06_01','06_02','06_03','06_04','06_05'],default='06_01')
args = parser.parse_args()
filename = {'06_01':'06_01_attempt_intervals.ipynb','06_02':'06_02/LSGG/06_02_lsgg_runway_case.ipynb','06_03':'06_02/LSGG/06_03_indirect_evidence.ipynb','06_04':'multiple_attempts/06_04_high_attempt_cases.ipynb','06_05':'multiple_attempts/06_05_three_attempt_cases.ipynb'}[args.step]
path = Path(__file__).resolve().parents[1] / 'notebooks/06_attempts' / filename
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
