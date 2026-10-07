import hashlib
import json
from pathlib import Path

import pandas as pd

REGION = Path(__file__).resolve().parents[1]
SOURCE = REGION / 'data/source/north_america.csv'
TARGET = REGION / 'data/derived/us_canada.csv'
CANADIAN_AIRPORTS = ['CYUL', 'CYOW']  # CYUL – Монреаль, CYOW – Оттава
EXCLUDED_US_AIRPORTS = [
    'KDEN',  # Денвер
    'KPWM',  # Портленд, Мэн
    'KPIT',  # Питтсбург
    'KSYR',  # Сиракьюс
    'KJAX',  # Джэксонвилл
    'KRIC',  # Ричмонд
    'KCMA',  # Камарильо – временно, до проверки аномалии
    'KSFB',  # Санфорд – временно, до проверки аномалии
]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def main():
    before = digest(SOURCE)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    partial = TARGET.with_suffix('.csv.partial')
    selected_rows = excluded_rows = go_arounds = 0
    expected = hashlib.sha256()
    for chunk in pd.read_csv(SOURCE, dtype=str, keep_default_na=False, chunksize=100_000):
        # Применяем согласованный отбор аэропортов, значения полей не исправляем
        selected = chunk.loc[(chunk['airport_country'].eq('USA') &
                              ~chunk['airport'].isin(EXCLUDED_US_AIRPORTS)) | (
            chunk['airport_country'].eq('CAN') & chunk['airport'].isin(CANADIAN_AIRPORTS))]
        selected.to_csv(partial, mode='w' if selected_rows == 0 else 'a',
                        header=selected_rows == 0, index=False)
        expected.update(pd.util.hash_pandas_object(selected, index=False).values.tobytes())
        selected_rows += len(selected)
        excluded_rows += len(chunk) - len(selected)
        go_arounds += int(selected['has_ga'].eq('True').sum())
    actual = hashlib.sha256()
    verified_rows = 0
    for chunk in pd.read_csv(partial, dtype=str, keep_default_na=False, chunksize=100_000):
        assert set(chunk['airport_country']) <= {'USA', 'CAN'}
        assert not chunk['airport'].isin(EXCLUDED_US_AIRPORTS).any()
        assert set(chunk.loc[chunk['airport_country'].eq('CAN'), 'airport']) <= set(CANADIAN_AIRPORTS)
        actual.update(pd.util.hash_pandas_object(chunk, index=False).values.tobytes())
        verified_rows += len(chunk)
    assert actual.digest() == expected.digest()
    assert selected_rows == verified_rows == 3_893_258
    assert excluded_rows == 16_058 and go_arounds == 15_822
    assert digest(SOURCE) == before
    partial.replace(TARGET)
    result = {'source': str(SOURCE.relative_to(REGION)), 'target': str(TARGET.relative_to(REGION)),
              'rows': selected_rows, 'excluded_from_analysis': excluded_rows,
              'go_arounds': go_arounds, 'source_sha256': before,
              'target_sha256': digest(TARGET), 'source_unchanged': True,
              'selected_values_unchanged': True, 'values_corrected': 0,
              'canadian_airports': CANADIAN_AIRPORTS,
              'excluded_us_airports': EXCLUDED_US_AIRPORTS,
              'temporarily_excluded_airports': ['KCMA', 'KSFB'],
              'scope': '53 observed USA airports; CYUL and CYOW in Canada'}
    (REGION / 'reports/us_canada_selection.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
