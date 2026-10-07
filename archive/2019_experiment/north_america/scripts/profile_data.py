import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data' / 'source' / 'north_america.csv'
NUMERIC_COLUMNS = [
    'n_approaches', 'n_rwy_approached', 'glide_slope_angle', 'rwy_length',
    'wind_speed_knts', 'wind_dir_deg', 'wind_gust_knts', 'visibility_m',
    'temperature_deg', 'press_sea_level_p', 'press_p',
]

# Это диагностические границы, не правила удаления и не нормы METAR
# Давление требует проверки вида, исходного сообщения и единиц
LIMITS = {
    'wind_speed_knts': (0, 150), 'wind_gust_knts': (0, 150),
    'wind_dir_deg': (0, 360), 'visibility_m': (0, 100_000),
    'temperature_deg': (-80, 60), 'press_p': (850, 1100),
    'press_sea_level_p': (850, 1100),
    'n_approaches': (1, np.inf), 'n_rwy_approached': (1, np.inf),
}
COLORS = ['#126782', '#E07A5F', '#7A9E7E', '#A78BBA', '#D5A33E', '#8296A8']


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    reports = ROOT / 'reports'
    review = ROOT / 'data' / 'review'
    for folder in (reports, review):
        folder.mkdir(parents=True, exist_ok=True)
    before_hash = file_hash(SOURCE)
    review_partial = review / 'flagged_rows.csv.partial'
    review_partial.write_text('', encoding='utf-8')
    row_count = 0
    flagged_count = 0
    flagged_ga = 0
    missing = None
    numeric_stats = {col: {'count': 0, 'missing': 0, 'parse_errors': 0,
                           'min': np.inf, 'max': -np.inf} for col in NUMERIC_COLUMNS}
    flag_counts = {}
    country_parts = []
    airport_parts = []
    month_parts = []
    pressure_parts = []
    examples = []
    bins = {
        'press_p': np.linspace(0, 3600, 181),
        'wind_speed_knts': np.linspace(0, 650, 131),
        'temperature_deg': np.linspace(-100, 2050, 216),
    }
    histograms = {}
    hist_outside = {}

    # Работаем с копиями числовых значений, исходные строки остаются неизменными
    for raw in pd.read_csv(SOURCE, dtype=str, keep_default_na=False, chunksize=100_000):
        start_row = row_count
        row_count += len(raw)
        counts = raw.eq('').sum()
        missing = counts if missing is None else missing.add(counts, fill_value=0)
        numeric = raw[NUMERIC_COLUMNS].apply(pd.to_numeric, errors='coerce')
        ga = raw['has_ga'].eq('True')
        times = pd.to_datetime(raw['time'], errors='coerce', utc=True)
        flags = {}
        for col in NUMERIC_COLUMNS:
            values = numeric[col]
            malformed = raw[col].ne('') & values.isna()
            stats = numeric_stats[col]
            stats['count'] += int(values.notna().sum())
            stats['missing'] += int(raw[col].eq('').sum())
            stats['parse_errors'] += int(malformed.sum())
            if values.notna().any():
                stats['min'] = min(stats['min'], float(values.min()))
                stats['max'] = max(stats['max'], float(values.max()))
            flags[f'{col}:parse_error'] = malformed
        for col, (lower, upper) in LIMITS.items():
            values = numeric[col]
            flags[f'{col}:outside_diagnostic_range'] = values.notna() & (
                values.lt(lower) | values.gt(upper)
            )
        flags['time:parse_error'] = times.isna()
        flags['has_ga:unexpected_value'] = ~raw['has_ga'].isin(['True', 'False'])
        flags['has_intersection:unexpected_value'] = ~raw['has_intersection'].isin(['True', 'False', ''])
        for col in ('n_approaches', 'n_rwy_approached'):
            flags[f'{col}:not_integer'] = numeric[col].notna() & numeric[col].mod(1).ne(0)
        for flag, mask in flags.items():
            count = flag_counts.setdefault(flag, {'rows': 0, 'go_arounds': 0})
            count['rows'] += int(mask.sum())
            count['go_arounds'] += int((mask & ga).sum())

        flag_frame = pd.DataFrame(flags)
        any_flag = flag_frame.any(axis=1)
        if any_flag.any():
            selected = raw.loc[any_flag].copy()
            selected.insert(0, 'source_row', np.flatnonzero(any_flag) + start_row)
            selected['review_flags'] = flag_frame.loc[any_flag].apply(
                lambda row: ';'.join(row.index[row]), axis=1,
            )
            selected.to_csv(review_partial, index=False, mode='a',
                            header=flagged_count == 0, encoding='utf-8')
            flagged_count += len(selected)
            flagged_ga += int(ga[any_flag].sum())
            # Примеры представлены по каждой причине, а не только из первого аэропорта
            for flag, mask in flags.items():
                previous = sum(example['review_reason'] == flag for example in examples)
                if previous < 3 and mask.any():
                    cols = ['time', 'airport', 'airport_country', 'has_ga',
                            'press_p', 'press_sea_level_p', 'temperature_deg', 'wind_speed_knts']
                    for idx in raw.index[mask][:3 - previous]:
                        item = raw.loc[idx, cols].to_dict()
                        item['source_row'] = start_row + int(raw.index.get_loc(idx))
                        item['review_reason'] = flag
                        examples.append(item)

        grouped = raw.assign(go_arounds=ga.astype(int))
        country_parts.append(grouped.groupby('airport_country').agg(
            rows=('has_ga', 'size'), go_arounds=('go_arounds', 'sum'),
        ))
        airport_parts.append(grouped.groupby(['airport_country', 'airport']).agg(
            rows=('has_ga', 'size'), go_arounds=('go_arounds', 'sum'),
        ))
        month_parts.append(grouped.assign(month=times.dt.strftime('%Y-%m')).groupby('month').agg(
            rows=('has_ga', 'size'), go_arounds=('go_arounds', 'sum'),
        ))
        pressure_mask = flags['press_p:outside_diagnostic_range']
        pressure_parts.append(grouped.assign(pressure_flags=pressure_mask.astype(int)).groupby(
            ['airport_country', 'airport'],
        ).agg(pressure_flags=('pressure_flags', 'sum')))

        for country in raw['airport_country'].unique():
            country_mask = raw['airport_country'].eq(country)
            for col, edges in bins.items():
                values = numeric.loc[country_mask, col].dropna().to_numpy()
                key = (country, col)
                histogram, _ = np.histogram(values, bins=edges)
                histograms[key] = histograms.get(key, np.zeros(len(edges) - 1, dtype=int)) + histogram
                hist_outside[key] = hist_outside.get(key, 0) + int(
                    ((values < edges[0]) | (values > edges[-1])).sum()
                )
        print(f'Processed {row_count:,} rows', flush=True)

    country = pd.concat(country_parts).groupby(level=0).sum()
    airport = pd.concat(airport_parts).groupby(level=[0, 1]).sum()
    monthly = pd.concat(month_parts).groupby(level=0).sum().sort_index()
    airport = airport.join(pd.concat(pressure_parts).groupby(level=[0, 1]).sum())
    for frame in (country, airport, monthly):
        frame['ga_percent'] = frame['go_arounds'] / frame['rows'] * 100
    airport['pressure_flag_percent'] = airport['pressure_flags'] / airport['rows'] * 100
    missing_table = pd.DataFrame({'missing': missing.astype(int)})
    missing_table['missing_percent'] = missing_table['missing'] / row_count * 100
    flags_table = pd.DataFrame.from_dict(flag_counts, orient='index')
    flags_table['percent'] = flags_table['rows'] / row_count * 100
    flags_table = flags_table.sort_values('rows', ascending=False)
    country.to_csv(reports / 'country_summary.csv')
    airport.to_csv(reports / 'airport_summary.csv')
    monthly.to_csv(reports / 'monthly_summary.csv')
    missing_table.to_csv(reports / 'missingness.csv')
    flags_table.to_csv(reports / 'review_flag_summary.csv')
    pd.DataFrame.from_dict(numeric_stats, orient='index').to_csv(reports / 'numeric_ranges.csv')
    pd.DataFrame(examples).to_csv(reports / 'review_examples.csv', index=False)
    if flagged_count:
        review_partial.replace(review / 'flagged_rows.csv')
    else:
        review_partial.unlink()
    after_hash = file_hash(SOURCE)
    if before_hash != after_hash:
        raise ValueError('Исходный CSV изменился во время анализа')
    if row_count != int(country['rows'].sum()) or row_count != int(airport['rows'].sum()):
        raise ValueError('Итоговые агрегаты не совпадают с числом строк')
    summary = {
        'rows': row_count, 'go_arounds': int(country['go_arounds'].sum()),
        'airports': len(airport), 'countries': len(country),
        'unique_flagged_rows': flagged_count, 'flagged_go_arounds': flagged_ga,
        'source_sha256': before_hash, 'source_unchanged': True,
        'rows_deleted': 0, 'values_corrected': 0,
        'review_rules_status': 'diagnostic_not_normative',
    }
    (reports / 'profile_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8',
    )

    report = f'''# Первичный региональный профиль Северной Америки

Дата: 07.10.2026. Все {row_count:,} строк проверены частями.
Аэропортов: {len(airport)}. Стран и территорий: {len(country)}.
Уходов: {summary['go_arounds']:,} ({summary['go_arounds']/row_count*100:.4f}%).

## Сохранность и границы проверки

SHA256 исходника: `{before_hash}`. Хеш до и после анализа совпадает.
Удалено строк: 0. Исправлено значений: 0.
Уникальных записей с диагностическими флагами: {flagged_count:,};
среди них has_ga=True: {flagged_ga:,}.
Они сохранены отдельно в `data/review/flagged_rows.csv` со всеми исходными
полями, номером строки `source_row` (с нуля, без заголовка) и причинами.
Флаги не означают доказанную ошибку, нестабильность или опасность захода.

## Ненулевые диагностические флаги

| Причина | Записей | Среди них has_ga=True |
| --- | --- | --- |
'''
    for reason, row in flags_table.loc[flags_table['rows'].gt(0)].iterrows():
        report += f"| `{reason}` | {int(row['rows']):,} | {int(row['go_arounds']):,} |\n"
    report += '''
## Визуализации

Визуализация запланирована следующим этапом в отдельном ноутбуке.
Папка figures не заполняется автоматически.

## Ограничения авиационной интерпретации

В CSV нет траекторной высоты, IAS, TAS, путевой скорости ВС, радиосвязи,
ATIS, назначения процедуры, clearance или кодов RNAV/RNP конкретного захода.
`wind_speed_knts` – ветер, не скорость самолета; `glide_slope_angle` –
характеристика ILS, не измеренная траектория. Проверить соблюдение переходной
высоты, ограничения скорости и навигационной процедуры по этим данным нельзя.
Эти вопросы ведем в реестре контекста и проверим при наличии траекторий,
исторических схем и разрешений. Современные документы не заменяют редакции 2019 года.

Следующий шаг – расследовать давление по аэропортам, времени и исходным METAR,
затем обсудить каждое правило обработки. Очищенный набор еще не создан.
'''
    (reports / 'regional_profile.md').write_text(report, encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(flags_table.loc[flags_table['rows'].gt(0)].to_string(), flush=True)


if __name__ == '__main__':
    main()
