"""Выполнение обзора 07_01 в Python-окружении проекта."""
from pathlib import Path
import argparse
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client.manager import KernelManager


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step',choices=['07_01','07_02'],default='07_01')
    args=parser.parse_args()
    filename={'07_01':'07_01_airport_overview.ipynb',
              '07_02':'EGCC/07_02_egcc_coverage.ipynb'}[args.step]
    path = Path(__file__).resolve().parents[1] / 'notebooks/07_airports' / filename
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    client = NotebookClient(notebook, km=manager, timeout=600,
                            resources={'metadata': {'path': str(path.parent)}}, allow_errors=False)
    client.execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print(f'Executed: {args.step}')


if __name__ == '__main__':
    main()
