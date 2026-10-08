from pathlib import Path
import sys,nbformat
from nbclient import NotebookClient
from jupyter_client.manager import KernelManager
R=Path(__file__).resolve().parents[1]
p=R/'notebooks/02_data_visualization.ipynb'
nb=nbformat.read(p,as_version=4);km=KernelManager(kernel_name='python3')
km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
client=NotebookClient(nb,km=km,timeout=600,resources={'metadata':{'path':str(R/'notebooks')}},allow_errors=False)
try:client.execute()
finally:nbformat.write(nb,p)
nbformat.validate(nb)
print('Executed cells:',len([c for c in nb.cells if c.cell_type=='code']))
