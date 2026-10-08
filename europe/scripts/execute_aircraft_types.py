"""Выполнение выбранного шага 05 в отдельном ядре окружения проекта."""
from pathlib import Path
import argparse
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client.manager import KernelManager

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--step', choices=['05_01','05_02','05_03','05_04','05_05'], default='05_01')
args = parser.parse_args()
filename = {'05_01': '05_01_aircraft_types_audit.ipynb', '05_02':'05_02_missing_types_review.ipynb', '05_03':'05_03_restore_classification.ipynb', '05_04':'05_04_model_names_restoration.ipynb', '05_05':'05_05_finish_type_review.ipynb'}[args.step]
path = Path(__file__).resolve().parents[1] / 'notebooks/05_aircraft_types' / filename
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
