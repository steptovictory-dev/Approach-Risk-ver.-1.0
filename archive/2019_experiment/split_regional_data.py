from pathlib import Path
import json
import hashlib

import pandas as pd


# Географический отбор по M49, не классификация единиц или правил УВД
# Перечислены страны и территории, присутствующие в исходном наборе
REGION_COUNTRIES = {
    'north_america': {'USA', 'CAN', 'MEX', 'BHS', 'ABW', 'BES'},
    'europe': {
        'AUT', 'BEL', 'BGR', 'BLR', 'CHE', 'DEU', 'DNK', 'ESP', 'FRA',
        'GBR', 'ISL', 'ITA', 'NLD', 'NOR', 'RUS', 'SWE', 'UKR',
    },
}


def main():
    root = Path(__file__).resolve().parent
    source = root / 'data' / 'raw' / 'go_arounds_augmented.csv.gz'
    output_dir = root / 'data' / 'regional'
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {region: output_dir / f'{region}.csv' for region in REGION_COUNTRIES}
    targets['north_america'] = root / 'north_america' / 'data' / 'source' / 'north_america.csv'
    targets['north_america'].parent.mkdir(parents=True, exist_ok=True)
    temporary = {region: path.with_suffix('.csv.partial') for region, path in targets.items()}
    if any(path.exists() for path in [*targets.values(), *temporary.values()]):
        raise FileExistsError('Региональные файлы уже существуют; перезапись не выполняется')

    counts = {region: 0 for region in targets}
    ga_counts = {region: 0 for region in targets}
    airports = {region: set() for region in targets}
    countries = {region: set() for region in targets}
    total = 0
    columns = None
    source_hashes = {region: hashlib.sha256() for region in targets}

    # Читаем строки без преобразования чисел и маркеров пропусков
    # Это разделение исходных записей, а не очистка или пересчет давления
    for chunk in pd.read_csv(
        source, compression='gzip', dtype=str, keep_default_na=False,
        chunksize=100_000,
    ):
        total += len(chunk)
        if columns is None:
            columns = chunk.columns.tolist()
        masks = {
            region: chunk['airport_country'].isin(country_codes)
            for region, country_codes in REGION_COUNTRIES.items()
        }
        if (masks['north_america'] & masks['europe']).any():
            raise ValueError('Региональные выборки пересекаются')
        for region, mask in masks.items():
            subset = chunk.loc[mask]
            if subset.empty:
                continue
            subset.to_csv(
                temporary[region], mode='a', header=counts[region] == 0,
                index=False, encoding='utf-8',
            )
            counts[region] += len(subset)
            ga_counts[region] += int(subset['has_ga'].eq('True').sum())
            airports[region].update(subset['airport'])
            countries[region].update(subset['airport_country'])
            source_hashes[region].update(
                pd.util.hash_pandas_object(subset, index=False).to_numpy().tobytes()
            )

    # Повторно читаем записанные файлы и проверяем сохраненные значения
    # Хеши строк сравниваем с отобранными исходными значениями в том же порядке
    readers = {
        region: pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=100_000)
        for region, path in temporary.items()
    }
    verified = {region: 0 for region in targets}
    for region, reader in readers.items():
        saved_hash = hashlib.sha256()
        for chunk in reader:
            if chunk.columns.tolist() != columns:
                raise ValueError('Изменилась схема регионального CSV')
            if not chunk['airport_country'].isin(REGION_COUNTRIES[region]).all():
                raise ValueError('В файл попала страна другого региона')
            verified[region] += len(chunk)
            saved_hash.update(
                pd.util.hash_pandas_object(chunk, index=False).to_numpy().tobytes()
            )
        if verified[region] != counts[region]:
            raise ValueError('Не совпало число сохраненных строк')
        if saved_hash.digest() != source_hashes[region].digest():
            raise ValueError('Изменились значения или порядок строк')

    for region, target in targets.items():
        temporary[region].replace(target)

    summary = {
        'source': source.name, 'source_rows': total,
        'selection_basis': 'UN M49, airport_country',
        'selection_reference': 'https://unstats.un.org/unsd/methodology/M49',
        'cleaning_applied': False, 'columns': columns,
        'excluded_other_regions_rows': total - sum(counts.values()),
        'regions': {
            region: {
                'file': str(target.relative_to(root)), 'rows': counts[region],
                'go_arounds': ga_counts[region], 'airports': len(airports[region]),
                'countries': sorted(countries[region]),
                'bytes': target.stat().st_size,
            }
            for region, target in targets.items()
        },
    }
    (output_dir / 'split_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8',
    )
    print(json.dumps(summary['regions'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
