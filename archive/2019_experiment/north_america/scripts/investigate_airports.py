import json
from pathlib import Path

import pandas as pd

REGION = Path(__file__).resolve().parents[1]


def main():
    parts = []
    for chunk in pd.read_csv(REGION / 'data/source/north_america.csv', dtype=str,
                             keep_default_na=False, chunksize=100_000):
        parts.append(chunk.loc[chunk['airport'].isin(['KCMA', 'KSFB'])])
    data = pd.concat(parts, ignore_index=True)
    assert len(data) == 2396
    data.to_csv(REGION / 'data/review/kcma_ksfb.csv', index=False)
    data['ga'] = data['has_ga'].eq('True')
    data['date'] = pd.to_datetime(data['time'], utc=True)
    data['month'] = data['date'].dt.strftime('%Y-%m')
    for col in ['n_approaches', 'n_rwy_approached', 'wind_speed_knts', 'wind_gust_knts',
                'visibility_m', 'temperature_deg', 'press_p']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    output = REGION / 'reports/airport_investigation'
    output.mkdir(exist_ok=True)
    groups = {'aircraft_types': ['airport', 'typecode'], 'runways': ['airport', 'runway'],
              'approach_counts': ['airport', 'n_approaches'],
              'months': ['airport', 'month'], 'aircraft': ['airport', 'icao24', 'typecode']}
    for filename, columns in groups.items():
        table = data.groupby(columns, dropna=False).agg(rows=('ga', 'size'),
                                                       go_arounds=('ga', 'sum')).reset_index()
        table['ga_percent'] = table['go_arounds'] / table['rows'] * 100
        # Идентификаторы бортов остаются только в локальной папке данных
        target = REGION / 'data/review' if filename == 'aircraft' else output
        table.to_csv(target / f'{filename}.csv', index=False)
    weather = data.groupby(['airport', 'ga'])[
        ['wind_speed_knts', 'wind_gust_knts', 'visibility_m', 'temperature_deg', 'press_p']
    ].agg(['count', 'median', 'min', 'max'])
    weather.to_csv(output / 'weather_summary.csv')
    summaries = []
    for airport, g in data.groupby('airport'):
        row = {'airport': airport, 'rows': len(g), 'go_arounds': int(g['ga'].sum()),
               'unique_aircraft': g['icao24'].nunique(), 'unique_dates': g['date'].dt.date.nunique(),
               'first_utc': str(g['date'].min()), 'last_utc': str(g['date'].max()),
               'observed_months': g['month'].nunique(),
               'ga_more_than_two_approaches': int((g['ga'] & g['n_approaches'].gt(2)).sum()),
               'max_approaches': int(g['n_approaches'].max()),
               'missing_typecode': int(g['typecode'].eq('').sum())}
        top = g.loc[g['ga']].groupby('icao24').size().sort_values(ascending=False)
        row['ga_top_five_aircraft'] = int(top.head(5).sum())
        summaries.append(row)
    (output / 'summary.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    types = pd.read_csv(output / 'aircraft_types.csv')
    for airport in ['KCMA', 'KSFB']:
        print(airport, 'types')
        print(types.loc[types.airport.eq(airport)].sort_values('rows', ascending=False).head(12).to_string(index=False))
    print('Weather medians')
    print(weather.xs('median', level=1, axis=1).to_string())
    print('Approach counts')
    print(pd.read_csv(output / 'approach_counts.csv').to_string(index=False))


if __name__ == '__main__':
    main()
