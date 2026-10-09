"""Проверяет идентификаторы и календарь точных повторов числового профиля."""
import pandas as pd
import nbformat
from audit_airports import ROOT


def main():
    report=ROOT/'europe/reports/airports/distributions'
    profiles=pd.read_csv(report/'07_06_profile_records.csv',dtype={'icao24':str})
    profiles['date']=profiles.C40_CROSS_TIME.str[:10]
    key=['airport','C40_BEARING','C40_CROSS_LAT','C40_CROSS_LON']
    groups=profiles.groupby(key).agg(rows=('GoA','size'),go_arounds=('GoA','sum'),aircraft=('icao24','nunique'),flight_ids=('flight_id','nunique'),dates=('date','nunique'),first_c40=('C40_CROSS_TIME','min'),last_c40=('C40_CROSS_TIME','max')).reset_index().sort_values('rows',ascending=False)
    top=groups.head(20);top.to_csv(report/'07_06_exact_profile_identity_review.csv',index=False)
    exact_keys=profiles.duplicated(['icao24','airport','landing_time'],keep=False).sum()
    assert exact_keys==0
    first=top.iloc[0]
    text=f'''\n## Дополнительная проверка точных повторов

В 1 319 933 записях выбранных частых округленных групп повторов составного ключа `icao24 + airport + landing_time` не найдено. Числовая геометрия повторяется у разных записей; совпадение трех числовых полей не является содержательным дублем.

Самая многочисленная точная комбинация в выбранных группах: **{first.airport}**, направление {first.C40_BEARING:.8f}°, широта {first.C40_CROSS_LAT:.8f}°, долгота {first.C40_CROSS_LON:.8f}°. Она встречается в **{int(first.rows):,} записях**, у **{int(first.aircraft):,} адресов бортов**, на **{int(first.dates):,} UTC-днях**; GoA=True: {int(first.go_arounds)}. Диапазон времени C40: {first.first_c40} – {first.last_c40}.

Сводка двадцати наиболее частых точных комбинаций с числом бортов и дат: `07_06_exact_profile_identity_review.csv`. Механизм повторения числовой геометрии пока не установлен; необходимо проверить происхождение и вычисление C40 по источнику. Нельзя считать эти повторы ни ошибкой, ни независимым подтверждением одинаковой траектории. Исходные значения и записи сохраняются.
'''
    path=report/'07_06_review.md';base=path.read_text(encoding='utf8').split('\n## Дополнительная проверка точных повторов')[0];path.write_text(base+text,encoding='utf8')
    path=ROOT/'europe/notebooks/07_airports/07_06_c40_distributions.ipynb';nb=nbformat.read(path,as_version=4)
    nb.cells[0].source=nb.cells[0].source.split('\n## Дополнительная проверка точных повторов')[0]+text
    nbformat.validate(nb);nbformat.write(nb,path)
    print(top.head(5).to_string(index=False))
    print('Повторов составного ключа:',exact_keys)


if __name__=='__main__':main()
