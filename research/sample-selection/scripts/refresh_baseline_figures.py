"""Перерисовать baseline из сохраненных прогнозов и отчетов, без переобучения."""
from pathlib import Path
import runpy,sys
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client.manager import KernelManager
root=Path(__file__).resolve().parents[1];path=root/'notebooks/08_baseline.ipynb'
old=nbf.read(path,as_version=4)
runpy.run_path(str(root/'scripts/build_baseline_notebook.py'))
new=nbf.read(path,as_version=4)
assert len(old.cells)==len(new.cells)
plot_index=next(i for i,c in enumerate(new.cells) if c.cell_type=='code' and "save('24_baseline_pr_roc')" in c.source)
for i,(before,after) in enumerate(zip(old.cells,new.cells)):
    if before.cell_type=='code':
        if i!=plot_index:assert before.source==after.source
        after.outputs=before.outputs;after.execution_count=before.execution_count
setup="""from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,roc_curve
ROOT=Path.cwd().resolve().parent;R=ROOT/'reports';F=ROOT/'figures'
plt.rcParams.update({'figure.dpi':120,'font.size':11,'axes.spines.top':False,
 'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
COLORS={'constant':'#546E63','geometry':'#00897B','geometry_aircraft':'#D65A4A'}
pred=pq.read_table(ROOT/'data/08_test_predictions.parquet').to_pandas()
ytest=pred.GoA.to_numpy(dtype=np.int8)
test_predictions={name:pred[name].to_numpy() for name in COLORS}
metrics=pd.read_csv(R/'baseline_metrics.csv');calibration=pd.read_csv(R/'baseline_calibration.csv')
intervals=pd.read_csv(R/'baseline_intervals.csv');airports=pd.read_csv(R/'baseline_airport_metrics.csv')
def save(name):
    plt.tight_layout()
    for ext in ['png','svg']:plt.savefig(F/f'{name}.{ext}',bbox_inches='tight')
    plt.show()
"""
temp=nbf.v4.new_notebook(cells=[nbf.v4.new_code_cell(setup),nbf.v4.new_code_cell(new.cells[plot_index].source)])
manager=KernelManager(kernel_name='python3')
manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
try:
    NotebookClient(temp,km=manager,timeout=300,resources={'metadata':{'path':str(root/'notebooks')}}).execute()
    new.cells[plot_index].outputs=temp.cells[1].outputs
    nbf.write(new,path)
except Exception:
    nbf.write(old,path)
    raise
print('Four baseline figures refreshed from fixed predictions')
