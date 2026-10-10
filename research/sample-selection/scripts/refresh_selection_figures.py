"""Перерисовать графики 01 из проверенных агрегатов, сохранив результаты аудита."""
from pathlib import Path
import sys
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client.manager import KernelManager
root=Path(__file__).resolve().parents[1]
path=root/'notebooks/01_selection_effects.ipynb'
nb=nbf.read(path,as_version=4)
setup=next(c.source for c in nb.cells if c.cell_type=='code')
plot_cells=[c for c in nb.cells if c.cell_type=='code' and 'savefig(' in c.source and 'def savefig' not in c.source]
assert len(plot_cells)==2
first=plot_cells[0].source.split('fig, axes = plt.subplots(1, 2',1)[1]
first='fig, axes = plt.subplots(1, 2'+first
second='airport = comparisons[\'airport\']'+plot_cells[1].source.split("airport = comparisons['airport']",1)[1]
cache="""
summary=pd.read_csv(REPORTS/'selection_summary.csv').set_index('stage').reindex(STAGES)
comparisons={g:pd.read_csv(REPORTS/f'comparison_{g}.csv',index_col=0)
             for g in ['airport','year','month','typecode','market_segment']}
"""
temp=nbf.v4.new_notebook(cells=[nbf.v4.new_code_cell(setup+'\n'+cache),
    nbf.v4.new_code_cell(first),nbf.v4.new_code_cell(second)])
manager=KernelManager(kernel_name='python3')
manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
NotebookClient(temp,km=manager,timeout=300,resources={'metadata':{'path':str(root/'notebooks')}}).execute()
for target,new in zip(plot_cells,temp.cells[1:]):
    images=[o for o in new.outputs if o.output_type=='display_data' and 'image/png' in o.get('data',{})]
    old=[o for o in target.outputs if o.output_type=='display_data' and 'image/png' in o.get('data',{})]
    assert len(images)==len(old)
    iterator=iter(images)
    target.outputs=[next(iterator) if o in old else o for o in target.outputs]
nbf.write(nb,path)
print('Six figures and inline outputs refreshed from existing aggregates')
