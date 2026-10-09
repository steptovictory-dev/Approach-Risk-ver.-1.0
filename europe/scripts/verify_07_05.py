"""Проверяет счетчики, навигацию и исполнимость ячеек 07_05."""
import re
import pandas as pd
import nbformat
from audit_feature_availability import ROOT


def main():
    report=ROOT/'europe/reports/airports/composition'
    missing=pd.read_csv(report/'07_05_missing.csv')
    totals=pd.read_csv(report/'07_05_label_totals.csv')
    cats=pd.read_csv(report/'07_05_categories.csv',keep_default_na=False)
    assert missing['missing'].le(missing.rows).all()
    assert missing['missing'].eq(missing['null']+missing['empty']+missing['whitespace']).all()
    for field in ['typecode','market_segment']:
        counts=cats[cats.field.eq(field)].groupby(['airport','GoA']).rows.sum().sort_index()
        expected=totals.set_index(['airport','GoA']).rows.sort_index()
        assert counts.equals(expected)
    old=pd.read_csv(ROOT/'europe/reports/airports/missing_by_label.csv')
    for field in ['typecode','registration','icaoaircrafttype','market_segment','AC_CLASS','AP_C_RWY']:
        new=missing[missing.field.eq(field)].set_index(['airport','GoA'])
        prior=old.set_index(['airport','GoA'])[field]
        assert (new['null']+new['empty']).sort_index().equals(prior.sort_index())
    path=ROOT/'europe/notebooks/07_airports/07_05_composition_missing.ipynb'
    nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
    namespace={}
    for cell in nb.cells:
        if cell.cell_type=='code':exec(compile(cell.source,str(path),'exec'),namespace)
        else:
            for link in re.findall(r'\]\(([^)]+)\)',cell.source):
                if not link.startswith(('http','#')):assert (path.parent/link).resolve().exists(),link
    print('Счетчики сверены с 07_01; категории сходятся с обеими метками. Ячейки выполнены, ссылки проверены.')


if __name__=='__main__':main()
