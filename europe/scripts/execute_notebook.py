from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client.manager import KernelManager

REGION=Path(__file__).resolve().parents[1]
SOURCE=REGION/'notebooks/01_database_audit.ipynb'

def main():
    notebook=nbformat.read(SOURCE,as_version=4)
    nbformat.validate(notebook)
    manager=KernelManager(kernel_name='python3')
    # Ядро использует тот же интерпретатор, что и команда выполнения.
    manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    client=NotebookClient(notebook,km=manager,timeout=900,resources={'metadata':{'path':str(REGION/'notebooks')}},allow_errors=False)
    try:
        client.execute()
    except Exception:
        nbformat.write(notebook,SOURCE)
        raise
    nbformat.validate(notebook)
    nbformat.write(notebook,SOURCE)
    code_cells=[cell for cell in notebook.cells if cell.cell_type=='code']
    if any(cell.execution_count is None for cell in code_cells): raise RuntimeError('Не все ячейки выполнены')
    if any(output.output_type=='error' for cell in code_cells for output in cell.get('outputs',[])):
        raise RuntimeError('В ноутбуке остались ошибки')
    print('Выполнено кодовых ячеек:',len(code_cells),flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf8')
    main()
