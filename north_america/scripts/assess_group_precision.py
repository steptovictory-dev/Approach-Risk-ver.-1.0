import math
from pathlib import Path
from statistics import NormalDist

import pandas as pd

REGION = Path(__file__).resolve().parents[1]
ALPHA = 0.05
EPSILON = 0.002
Z = NormalDist().inv_cdf(1 - ALPHA / 2)
US_EXCLUDED = {'KDEN', 'KPWM', 'KPIT', 'KSYR', 'KJAX', 'KRIC'}
CAN_SELECTED = {'CYUL', 'CYOW'}  # CYUL – Монреаль, CYOW – Оттава
CITIES = {'KDEN': 'Денвер', 'KPWM': 'Портленд, Мэн', 'KPIT': 'Питтсбург',
          'KSYR': 'Сиракьюс', 'KJAX': 'Джэксонвилл', 'KRIC': 'Ричмонд',
          'CYUL': 'Монреаль', 'CYOW': 'Оттава', 'CYYJ': 'Виктория',
          'CYQB': 'Квебек', 'CYWG': 'Виннипег'}


def assess(frame):
    result = frame.copy()
    proportions, lower, upper, delta = [], [], [], []
    for row in frame.itertuples():
        n, k = int(row.rows), int(row.go_arounds)
        p = k / n
        denominator = 1 + Z**2 / n
        center = (p + Z**2 / (2 * n)) / denominator
        half = Z * math.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / denominator
        lo, hi = max(0, center - half), min(1, center + half)
        proportions.append(p)
        lower.append(lo)
        upper.append(hi)
        delta.append(max(p - lo, hi - p))
    result['p_hat'] = proportions
    result['wilson_lower'] = lower
    result['wilson_upper'] = upper
    result['delta'] = delta
    result['delta_pp'] = result['delta'] * 100
    result['precision_pass'] = result['delta'].le(EPSILON)
    result['epsilon'] = EPSILON
    return result


def main():
    airports = assess(pd.read_csv(REGION / 'reports/airport_summary.csv'))
    countries = assess(pd.read_csv(REGION / 'reports/country_summary.csv'))
    airports['included_current'] = (
        airports['airport_country'].eq('USA') & ~airports['airport'].isin(US_EXCLUDED)
    ) | (airports['airport_country'].eq('CAN') & airports['airport'].isin(CAN_SELECTED))
    airports['city'] = airports['airport'].map(CITIES).fillna('')
    airports.to_csv(REGION / 'reports/airport_precision.csv', index=False)
    countries.to_csv(REGION / 'reports/country_precision.csv', index=False)
    text = r'''# Обоснование состава аналитической выборки

## Критерий точности

Единица учета – запись о посадке, событие – наличие хотя бы одного отмеченного ухода

$$
\widehat p_i=\frac{k_i}{n_i},\qquad
\Delta_i=\max(\widehat p_i-L_i,\;U_i-\widehat p_i),\qquad
\Delta_i\leq\varepsilon
$$

$[L_i,U_i]$ – двусторонний интервал Уилсона с номинальным уровнем доверия 95%, $z=\Phi^{-1}(0{,}975)$. $\varepsilon$ – допустимое абсолютное отклонение в единицах вероятности. В таблицах $\Delta$ выражено в процентных пунктах

Согласован допуск $\varepsilon=0{,}002$ (0,2 п.п.), дата 07.10.2026. Интервалы рассчитаны для всех 74 аэропортов и шести стран и территорий, включая исключенные. Это выбранная абсолютная точность исследования. Параметр согласован после предварительного анализа, поэтому применение является ретроспективным

Расчет предполагает независимость наблюдений и общую вероятность события внутри группы. Полнота периода, зависимость полетов и корректность разметки оцениваются отдельно. Низкая точность отдельной оценки не требует удаления записей из исходной базы

## Исключенные страны и территории

| Группа | Записей | С уходом | Интервал Уилсона, % | $\Delta$, п.п. | Основание текущего решения |
| --- | ---: | ---: | --- | ---: | --- |
'''
    reasons = {'ABW': ('Аруба (ABW)', 'Критерий точности не выполнен; записи сохранены'),
               'BES': ('Карибы (BES)', 'Критерий точности не выполнен; записи сохранены'),
               'BHS': ('Багамы (BHS)', 'Критерий точности не выполнен; одна запись'),
               'MEX': ('Мексика (MEX)', 'Национальный нормативный контекст не проверен; отдельное исследование отложено')}
    for row in countries.itertuples():
        if row.airport_country in reasons:
            name, reason = reasons[row.airport_country]
            text += f'| {name} | {row.rows} | {row.go_arounds} | {row.wilson_lower*100:.4f}–{row.wilson_upper*100:.4f} | {row.delta_pp:.4f} | {reason} |\n'
    text += r'''
Малая доля страны в общей выборке не является самостоятельным доказательством нерепрезентативности. Исключение Мексики не объявляется следствием критерия точности. Национальные правила США и Канады на ее записи не переносятся

## Исключенные аэропорты США и Канады

| ICAO – город | Записей | С уходом | Интервал Уилсона, % | $\Delta$, п.п. |
| --- | ---: | ---: | --- | ---: |
'''
    excluded = airports.loc[airports['airport'].isin(US_EXCLUDED | {'CYYJ', 'CYQB', 'CYWG'})]
    for row in excluded.itertuples():
        text += f'| {row.airport} – {row.city} | {row.rows} | {row.go_arounds} | {row.wilson_lower*100:.4f}–{row.wilson_upper*100:.4f} | {row.delta_pp:.4f} |\n'
    text += '''
Шесть аэропортов США исключены по согласованному ограничению состава эксперимента группами с малым объемом наблюдений. Отсутствие уходов не использовано как самостоятельный критерий. Все шесть групп не проходят согласованный допуск. Формальный расчет дополняет первоначальное решение, принятое до согласования параметра

В Канаде текущий эксперимент ограничен CYUL – Монреаль и CYOW – Оттава, которые дают около 90,2% канадских наблюдений. CYYJ – Виктория и CYQB – Квебек имеют зарегистрированные события; меньший объем не доказывает их непригодность. CYWG – Виннипег представлен шестью записями. Все три группы не проходят согласованный допуск. Ограничение частотных оценок количественно подтверждено; исходные записи сохраняются

Критерий рассчитан для всех аэропортов, включая оставленные. Из 57 текущих аэропортов 11 не проходят допуск; дополнительные исключения не применены. Порог не подбирается под прежний список исключений. Ограничение точности аэропортной оценки и отбор записей для общей модели рассматриваются отдельно

## Сохранность и текущий состав

Исходный региональный CSV сохранен полностью. В аналитическом CSV 3 895 654 записи, 16 197 с отмеченным уходом, 55 аэропортов США и два аэропорта Канады. За пределами текущего анализа сохранены 13 662 записи

Значения температуры и других полей не удалены и не исправлены. Высокие доли уходов в KSFB – Санфорд и KCMA – Камарильо не стали основанием исключения; группы оставлены для проверки

Расчеты: `airport_precision.csv`, `country_precision.csv`. Повторение: `python north_america/scripts/assess_group_precision.py`
'''
    (REGION / 'reports/selection_rationale.md').write_text(text, encoding='utf-8')
    print(excluded[['airport', 'rows', 'go_arounds', 'delta_pp']].to_string(index=False))
    print(countries[['airport_country', 'delta_pp']].to_string(index=False))


if __name__ == '__main__':
    main()
