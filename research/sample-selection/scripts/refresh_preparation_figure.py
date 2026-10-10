"""Перерисовать категории 07 из агрегатов, сохранив остальные результаты."""
from pathlib import Path
import contextlib
import io
import runpy
import sys
from unittest.mock import patch
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client.manager import KernelManager

root=Path(__file__).resolve().parents[1]
path=root/'notebooks/07_feature_preparation.ipynb'
old=nbf.read(path,as_version=4)
captured=[]
with patch.object(nbf,'write',side_effect=lambda nb,*args,**kwargs:captured.append(nb)),contextlib.redirect_stdout(io.StringIO()):
    runpy.run_path(str(root/'scripts/build_preparation_notebook.py'))
assert len(captured)==1
new=captured[0]
assert len(old.cells)==len(new.cells)
plot_index=next(i for i,c in enumerate(new.cells) if c.cell_type=='code' and "save('22_category_missing_and_unseen')" in c.source)
for i,(before,after) in enumerate(zip(old.cells,new.cells)):
    assert before.cell_type==after.cell_type
    if i!=plot_index:assert before.source==after.source
    if before.cell_type=='code':
        after.outputs=before.outputs
        after.execution_count=before.execution_count

setup="""from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path.cwd().resolve().parent
R=ROOT/'reports'; F=ROOT/'figures'
states=pd.read_csv(R/'preparation_category_states.csv')
plt.rcParams.update({'figure.dpi':120,'font.size':11,'axes.spines.top':False,
 'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
def save(name):
    plt.tight_layout()
    for ext in ['png','svg']:plt.savefig(F/f'{name}.{ext}',bbox_inches='tight')
    plt.show()
"""
plot=new.cells[plot_index].source.split('from matplotlib.ticker import FuncFormatter',1)[1]
plot='from matplotlib.ticker import FuncFormatter'+plot
temp=nbf.v4.new_notebook(cells=[nbf.v4.new_code_cell(setup),nbf.v4.new_code_cell(plot)])
manager=KernelManager(kernel_name='python3')
manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
NotebookClient(temp,km=manager,timeout=300,resources={'metadata':{'path':str(root/'notebooks')}}).execute()
images=[o for o in temp.cells[1].outputs if o.output_type=='display_data' and 'image/png' in o.get('data',{})]
old_images=[o for o in old.cells[plot_index].outputs if o.output_type=='display_data' and 'image/png' in o.get('data',{})]
assert len(images)==len(old_images)==1
new.cells[plot_index].outputs=[images[0] if o in old_images else o for o in old.cells[plot_index].outputs]
nbf.write(new,path)
print('Category figure and notebook image refreshed from unchanged aggregates')
