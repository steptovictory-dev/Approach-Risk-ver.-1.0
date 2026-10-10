"""Выполнение ноутбука текущим интерпретатором с сохранением диагностики."""
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client.manager import KernelManager

root = Path(__file__).resolve().parents[1]
name = sys.argv[1] if len(sys.argv) > 1 else '01_selection_effects.ipynb'
assert Path(name).name == name and name.endswith('.ipynb')
path = root / 'notebooks' / name
notebook = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name='python3')
manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
client = NotebookClient(notebook, km=manager, timeout=1800,
    resources={'metadata': {'path': str(root / 'notebooks')}}, allow_errors=False)
client.on_cell_start = lambda cell, cell_index, **kwargs: print(
    f'Cell {cell_index + 1}/{len(notebook.cells)}: {cell.cell_type}', flush=True)
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
nbformat.validate(notebook)
assert all(c.execution_count is not None for c in notebook.cells if c.cell_type == 'code')
print('Notebook executed successfully', flush=True)
