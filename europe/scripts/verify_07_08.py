"""Проверяет агрегаты 07_08 и описывает состав диагностического хвоста."""
import json
import re
import numpy as np
import pandas as pd
import nbformat
from audit_c40_radial_cases import ROOT, ellipsoid_distance


def main():
    report=ROOT/'europe/reports/airports/c40_geometry'
    audit=json.loads((report/'07_08_audit.json').read_text(encoding='utf8'))
    counts=pd.read_csv(report/'07_08_radius_by_label.csv')
    months=pd.read_csv(report/'07_08_radius_by_month.csv')
    records=pd.read_csv(report/'07_08_radial_records.csv',dtype={'icao24':str})
    assert counts.rows.sum()==audit['rows'] and months.rows.sum()==audit['rows']
    assert counts.tail_1km.sum()==months.tail_1km.sum()==len(records)==audit['tail_1km']
    assert counts.tail_3km.sum()==months.tail_3km.sum()==audit['tail_3km']
    assert records.abs_deviation_km.gt(1).all() and records.source_c40_exact_match.all()
    assert records.minutes_c40_to_first_attempt.gt(0).all()
    assert not records.duplicated(['icao24','airport','landing_time']).any()
    forward=ellipsoid_distance(51.,0.,np.array([51.1]),np.array([.2]))
    reverse=ellipsoid_distance(51.1,.2,np.array([51.]),np.array([0.]))
    assert np.allclose(forward,reverse,rtol=0,atol=1e-9)
    records['date_c40']=pd.to_datetime(records.C40_CROSS_TIME,utc=True).dt.strftime('%Y-%m-%d')
    composition=records.groupby(['airport','GoA']).agg(rows=('GoA','size'),aircraft=('icao24','nunique'),dates=('date_c40','nunique'),months=('month_c40','nunique'),first_c40=('C40_CROSS_TIME','min'),last_c40=('C40_CROSS_TIME','max'),before_first_min=('minutes_c40_to_first_attempt','min')).reset_index()
    composition.to_csv(report/'07_08_tail_composition.csv',index=False)
    airports=composition.groupby('airport').rows.sum().sort_values(ascending=False)
    peak=months[months.rows.gt(0)].sort_values('tail_1km_percent',ascending=False).head(10)
    peak.to_csv(report/'07_08_tail_months_review.csv',index=False)
    dominant=pd.read_csv(report/'07_08_dominant_points.csv')
    raw_profiles=pd.read_csv(report/'07_08_dominant_point_records.csv',dtype={'icao24':str})
    assert len(dominant)==19 and dominant.rows.sum()==len(raw_profiles)==audit['dominant_point_rows']
    assert not raw_profiles.duplicated(['icao24','airport','landing_time']).any()
    first=dominant.sort_values('rows',ascending=False).iloc[0]
    text='\n## Проверка состава диагностических групп\n\n'
    text+='Наибольшее число записей с радиальным отклонением более 1 км: '+', '.join(f'{airport} – {int(rows)}' for airport,rows in airports.head(5).items())+'. Это ранжирование абсолютных чисел, а не вероятностей.\n\n'
    text+=f'Всего в хвосте {records.icao24.nunique()} разных ICAO24; временное покрытие {records.C40_CROSS_TIME.min()} – {records.C40_CROSS_TIME.max()}. Состав по бортам, дням и месяцам сохранен в `07_08_tail_composition.csv`. Месяцы с наибольшей долей хвоста – в `07_08_tail_months_review.csv`, вместе с n каждой метки: малый знаменатель нельзя интерпретировать как массовую проблему.\n\n'
    text+=f'Наиболее частая точная точка: {first.airport}, {int(first.rows)} записей ({first.share_airport_percent:.3f}% аэропорта), {int(first.aircraft)} бортов, {int(first.dates)} дней и {int(first.months)} месяцев. Здесь {int(first.go_arounds)} GoA=True; доля {first.ga_percent:.3f}% против {first.airport_ga_percent:.3f}% во всем аэропорте. Это описательное сравнение без календарной и погодной корректировки, не оценка влияния направления.\n\n'
    text+='Цифровая точность координат и точный расчет bearing объясняют совпадение числовой геометрии. Источник многокилометровых радиальных отклонений остается открытым; в проверенных записях подтверждены исходные значения и допустимый временной порядок C40, но не соответствие CPR физическому полету.\n'
    path=report/'07_08_review.md';base=path.read_text(encoding='utf8').split('\n## Проверка состава диагностических групп')[0];path.write_text(base+text,encoding='utf8')
    path=ROOT/'europe/notebooks/07_airports/07_08_c40_radial_cases.ipynb';nb=nbformat.read(path,as_version=4)
    nb.cells[0].source=(base+text).replace('../../../figures/','../../figures/')
    nbformat.validate(nb);nbformat.write(nb,path);namespace={}
    for cell in nb.cells:
        if cell.cell_type=='code':exec(compile(cell.source,str(path),'exec'),namespace)
        else:
            for link in re.findall(r'\]\(([^)]+)\)',cell.source):
                if not link.startswith(('http','#')):assert (path.parent/link).resolve().exists(),link
    print(text)
    print('Счетчики сверены, симметрия расстояния проверена, ячейки выполнены и ссылки проверены.')


if __name__=='__main__':main()
