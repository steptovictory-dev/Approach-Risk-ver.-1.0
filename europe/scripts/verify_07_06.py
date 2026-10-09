"""Проверка агрегатов, ссылок и ячеек 07_06."""
import json
import re
import pandas as pd
import nbformat
from audit_airports import ROOT


def main():
    folder=ROOT/'europe/reports/airports/distributions'
    audit=json.loads((folder/'07_06_audit.json').read_text(encoding='utf8'))
    histogram=pd.read_csv(folder/'07_06_bearing_histogram.csv')
    assert histogram.rows.sum()==audit['rows']-audit['diagnostic_out_of_range']
    assert (histogram.groupby(['airport','GoA']).within_label_percent.sum()-100).abs().lt(1e-8).all()
    monthly=pd.read_csv(folder/'07_06_monthly.csv')
    assert monthly.rows.sum()==histogram.rows.sum()
    assert monthly.loc[monthly.GoA,'rows'].sum()<=audit['go_arounds']
    assert monthly.resultant_length.between(0,1+1e-12).all()
    stats=pd.read_csv(folder/'07_06_summary.csv')
    for row in stats.itertuples():
        values=[row.min,row.q01,row.q05,row.q25,row.q50,row.q75,row.q95,row.q99,row.max]
        assert all(a<=b for a,b in zip(values,values[1:]))
    profiles=pd.read_csv(folder/'07_06_profile_records.csv')
    selected=pd.read_csv(folder/'07_06_frequent_profiles.csv')
    assert len(profiles)==selected.rows.sum()
    keys=['airport','bearing_bin','lat_bin','lon_bin']
    actual=profiles.groupby(keys).GoA.agg(rows='size',go_arounds='sum').reset_index()
    check=selected.merge(actual,on=keys,suffixes=('_expected','_actual'),validate='one_to_one')
    assert check.rows_expected.eq(check.rows_actual).all() and check.go_arounds_expected.eq(check.go_arounds_actual).all()
    path=ROOT/'europe/notebooks/07_airports/07_06_c40_distributions.ipynb'
    nb=nbformat.read(path,as_version=4);nbformat.validate(nb);namespace={}
    for cell in nb.cells:
        if cell.cell_type=='code':exec(compile(cell.source,str(path),'exec'),namespace)
        else:
            for link in re.findall(r'\]\(([^)]+)\)',cell.source):
                if not link.startswith(('http','#')):assert (path.parent/link).resolve().exists(),link
    print('Гистограммы и месяцы сходятся; квантили и счетчики профилей проверены. Ячейки выполнены, ссылки действуют.')


if __name__=='__main__':main()
