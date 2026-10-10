"""Повторить ячейки подключения данных 04–08 с внешним корнем, без расчетов."""
from pathlib import Path
import os,sys,json
import nbformat
from nbclient import NotebookClient
from jupyter_client.manager import KernelManager

root=Path(__file__).resolve().parents[1]
external=Path(os.environ.get('SAMPLE_SELECTION_DATA_ROOT',root.parents[1])).resolve()
os.environ['SAMPLE_SELECTION_DATA_ROOT']=str(external)
results=[]
for number in range(4,9):
    path=next((root/'notebooks').glob(f'{number:02d}_*.ipynb'))
    notebook=nbformat.read(path,as_version=4)
    index=next(i for i,c in enumerate(notebook.cells) if c.cell_type=='code')
    check="\nassert Path(os.environ['SAMPLE_SELECTION_DATA_ROOT']).resolve()==Path("+repr(str(external))+ ").resolve()\n"
    if number==5:check+='assert PROJECT.resolve()==Path(os.environ["SAMPLE_SELECTION_DATA_ROOT"]).resolve()\n'
    else:check+='assert DATA.resolve().is_relative_to(Path(os.environ["SAMPLE_SELECTION_DATA_ROOT"]).resolve())\n'
    temp=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(notebook.cells[index].source),
                                      nbformat.v4.new_code_cell(check)])
    manager=KernelManager(kernel_name='python3')
    manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    NotebookClient(temp,km=manager,timeout=300,resources={'metadata':{'path':str(root/'notebooks')}}).execute()
    notebook.cells[index].outputs=temp.cells[0].outputs
    nbformat.write(notebook,path)
    results.append({'notebook':path.name,'setup_reexecuted':True,'external_root_honored':True,
                    'downstream_calculations_changed':False})
    print(path.name,'external paths OK',flush=True)
(root/'reports/external_paths_verification.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
