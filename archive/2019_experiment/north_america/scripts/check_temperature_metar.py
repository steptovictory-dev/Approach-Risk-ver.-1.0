import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

REGION = Path(__file__).resolve().parents[1]


def main():
    flagged = pd.read_csv(REGION / 'data/review/flagged_rows.csv')
    suspect = flagged.loc[pd.to_numeric(flagged['temperature_deg'], errors='coerce').eq(2018)]
    stations = sorted(suspect['airport'].unique())
    params = [('station', station if station == 'PANC' else station[1:]) for station in stations]
    params += [('data', 'all'), ('year1', 2019), ('month1', 7), ('day1', 20),
               ('year2', 2019), ('month2', 7), ('day2', 21), ('tz', 'Etc/UTC'),
               ('format', 'onlycomma'), ('latlon', 'no'), ('elev', 'no'),
               ('missing', 'M'), ('trace', 'T'), ('direct', 'no'),
               ('report_type', 3), ('report_type', 4)]
    url = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?' + urlencode(params)
    output = REGION / 'data/review/metar_20190720.csv'
    with urlopen(url, timeout=60) as response:
        content = response.read()
    output.write_bytes(content)
    weather = pd.read_csv(output, comment='#', na_values=['M'])
    assert 'metar' in weather.columns and 'tmpf' in weather.columns
    station_to_airport = {station if station == 'PANC' else station[1:]: station for station in stations}
    weather['airport'] = weather['station'].map(station_to_airport)
    weather['weather_time'] = pd.to_datetime(weather['valid'], utc=True)
    weather['temperature_c'] = (pd.to_numeric(weather['tmpf'], errors='coerce') - 32) * 5 / 9
    suspect = suspect.copy()
    suspect['event_time'] = pd.to_datetime(suspect['time'], utc=True)
    # Сопоставляем только предшествующее наблюдение в пределах 90 минут
    matched = pd.merge_asof(suspect.sort_values('event_time'),
                            weather.sort_values('weather_time'),
                            left_on='event_time', right_on='weather_time', by='airport',
                            direction='backward', tolerance=pd.Timedelta(minutes=90))
    matched['weather_age_minutes'] = (matched['event_time'] - matched['weather_time']).dt.total_seconds() / 60
    cols = ['source_row', 'airport', 'time', 'temperature_deg', 'weather_time',
            'temperature_c', 'weather_age_minutes', 'metar']
    matched[cols].to_csv(REGION / 'data/review/temperature_metar_matches.csv', index=False)
    result = {'url': url, 'sha256': hashlib.sha256(content).hexdigest(),
              'downloaded_observations': len(weather), 'suspect_rows': len(suspect),
              'matched_rows': int(matched['weather_time'].notna().sum()),
              'matched_temperature_min_c': float(matched['temperature_c'].min()),
              'matched_temperature_max_c': float(matched['temperature_c'].max()),
              'matching': 'preceding METAR within 90 minutes; original augmentation rule unknown',
              'values_corrected': 0}
    (REGION / 'reports/temperature_metar_check.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
