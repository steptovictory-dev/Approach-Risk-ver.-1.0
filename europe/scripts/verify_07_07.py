"""Проверяет восстановленную формулу на всех строках и дополняет выводы."""
import json
import re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import nbformat
from investigate_c40_geometry import ROOT, INPUT, bearing, difference, distance


def main():
    report=ROOT/'europe/reports/airports/c40_geometry'
    best=pd.read_csv(report/'07_07_best_formula.csv').set_index('airport')
    counts={airport:dict(rows=0,max_error_deg=0.,min_radius_km=float('inf'),max_radius_km=0.) for airport in best.index}
    for batch in pq.ParquetFile(INPUT).iter_batches(columns=['airport','C40_BEARING','C40_CROSS_LAT','C40_CROSS_LON'],batch_size=100000):
        frame=batch.to_pandas(ignore_metadata=True)
        for airport,group in frame.groupby('airport'):
            center=best.loc[airport]
            predicted=bearing(center.fit_lat,center.fit_lon,group.C40_CROSS_LAT.to_numpy(),group.C40_CROSS_LON.to_numpy())
            errors=np.abs(difference(predicted,group.C40_BEARING.to_numpy()))
            radii=distance(center.fit_lat,center.fit_lon,group.C40_CROSS_LAT.to_numpy(),group.C40_CROSS_LON.to_numpy())
            result=counts[airport];result['rows']+=len(group);result['max_error_deg']=max(result['max_error_deg'],float(errors.max()))
            result['min_radius_km']=min(result['min_radius_km'],float(radii.min()));result['max_radius_km']=max(result['max_radius_km'],float(radii.max()))
    check=pd.DataFrame.from_dict(counts,orient='index').reset_index(names='airport')
    check.to_csv(report/'07_07_full_formula_check.csv',index=False)
    audit=json.loads((report/'07_07_audit.json').read_text(encoding='utf8'))
    assert check.rows.sum()==audit['rows'] and check.max_error_deg.max()<1e-8
    assert audit['arcsecond_grid_values']==audit['coordinate_values'] and audit['pairs_multiple_bearings']==0
    maxerror=float(check.max_error_deg.max())
    audit['all_rows_bearing_formula_max_error_deg']=maxerror
    (report/'07_07_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    text=f'''\n## Установленный результат – проверка на всех строках

После подгонки на уникальных точках формула дополнительно проверена **на всех {audit['rows']:,} строках**. Максимальное абсолютное расхождение направления: **{maxerror:.3g}°**. Во всех 19 аэропортах успешно восстановлена ориентация **аэропорт → точка C40**. Для каждой точной пары координат внутри аэропорта bearing одинаков. Численно это вычислительная зависимость с точностью floating point; bearing не несет отдельного независимого измерения сверх аэропорта и координат.

Повторы объясняются конечной координатной сеткой: одинаковые сохраненные координаты дают одинаковый вычисленный азимут. Шаг сетки установлен во всех строках, но конкретный этап округления и причины частого попадания в отдельные ячейки остаются неизвестными. Возможная концентрация движения по определенным направлениям требует отдельной траекторной проверки.

Радиальные медианы по аэропортам составляют {best.radius_median_km.min():.3f}–{best.radius_median_km.max():.3f} км и согласуются с геометрией границы 40 NM (74,08 км) в выбранном сферическом приближении. Это не независимая валидация официальной границы: центр восстановлен из самих полей. **Расстояния 140–147 км в 07_06 были расстояниями до медианы точек входа, а не до аэропорта.** Медиана точек, сосредоточенных на одной стороне окружности, может лежать далеко от центра; большие расстояния до нее не означают ошибочный C40.

Практический вывод: все записи и исходные поля сохраняем. При подготовке признаков можно сравнить азимут (например, sin/cos), координаты и их совместное использование. Поле bearing не является курсом самолета или направлением ветра. Окончательное решение о признаках еще не принято.
'''
    path=report/'07_07_review.md';original=path.read_text(encoding='utf8').split('\n## Установленный результат')[0]
    original=original.replace('для подгонки берется систематическая подвыборка примерно до 5000 точек','для подгонки берется систематическая подвыборка; фактические размеры сохранены в таблице результатов')
    original=original.replace('Если связь подтверждается, три поля не являются тремя независимыми измерениями.', 'Связь подтверждена численно: три поля не являются тремя независимыми измерениями.')
    path.write_text(original+text,encoding='utf8')
    path=ROOT/'europe/notebooks/07_airports/07_07_c40_geometry.ipynb';nb=nbformat.read(path,as_version=4)
    nb.cells[0].source=(original+text).replace('../../../figures/','../../figures/')
    nbformat.validate(nb);nbformat.write(nb,path)
    namespace={}
    for cell in nb.cells:
        if cell.cell_type=='code':exec(compile(cell.source,str(path),'exec'),namespace)
        else:
            for link in re.findall(r'\]\(([^)]+)\)',cell.source):
                if not link.startswith(('http','#')):assert (path.parent/link).resolve().exists(),link
    notebook=json.loads((ROOT/'docs/sources/files/c40_review/author_OSN_paper_code.ipynb').read_text(encoding='utf8'))
    hits=[]
    for i,cell in enumerate(notebook['cells']):
        source=''.join(cell.get('source',[]))
        if cell['cell_type']=='code' and any(token in source for token in ['C40_BEARING','C40_CROSS_LAT','C40_CROSS_LON']):hits.append(dict(cell_index=i,source=source))
    (report/'07_07_author_geometry_cells.json').write_text(json.dumps(hits,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(check.to_string(index=False))
    print('Ячейки автора с координатами и bearing:',len(hits))
    for hit in hits:print(hit['cell_index'],hit['source'][:1400])
    print('Полная проверка выполнена, ноутбук и ссылки проверены.')


if __name__=='__main__':main()
