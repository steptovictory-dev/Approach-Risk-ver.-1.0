"""Resume post-fit reporting cells after repairing a date-input compatibility bug."""
from pathlib import Path
import nbformat as nb
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'research/boosting-weather-comparison'
path=BASE/'notebooks/01_catboost_training.ipynb';original=nb.read(path,as_version=4)
indices=[i for i,c in enumerate(original.cells) if c.cell_type=='code' and ('interval,bootstrap=paired_intervals' in c.source or 'saved=[]' in c.source)]
assert len(indices)==2
setup=nb.v4.new_code_cell('''from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from IPython.display import display
ROOT=Path.cwd();BASE=ROOT/'research/boosting-weather-comparison'
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(BASE/'scripts'))
from boosting_experiment import Experiment,DEST,write_json
from approach_risk.evaluation.bootstrap import paired_intervals
experiment=Experiment()
selection=json.loads((BASE/'reports/02_selection.json').read_text(encoding='utf8'))
assert selection['fingerprint']==experiment.fingerprint and not selection['test_used_for_selection']
experiment.selected=selection['selected']
experiment.rows=pd.read_csv(BASE/'reports/01_validation_candidates.csv').to_dict('records')
assert len(experiment.rows)==10
''')
mini=nb.v4.new_notebook(cells=[setup]+[nb.v4.new_code_cell(original.cells[i].source) for i in indices],metadata=original.metadata)
NotebookClient(mini,timeout=1800,resources={'metadata':{'path':str(ROOT)}}).execute()
for i,new in zip(indices,mini.cells[1:]):
    count=sum(c.cell_type=='code' for c in original.cells[:i+1])
    original.cells[i].outputs=new.outputs;original.cells[i].execution_count=count
    for output in original.cells[i].outputs:
        if output.output_type=='execute_result':output.execution_count=count
original.metadata['reporting_recovery']={'reason':'Series date input normalized to DatetimeIndex for UTC-day blocks','fit_rerun':False,'selection_changed':False,'recovered_cells':indices}
nb.write(original,path)
print('Reporting cells completed; fit and selected models unchanged.')
