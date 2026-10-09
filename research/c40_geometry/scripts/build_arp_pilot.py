"""Сверка исторических схем с восстановленными центрами, без изменения данных."""
from pathlib import Path
import hashlib
import json
import math
import pandas as pd
import nbformat

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'research/c40_geometry'


def dms(value):
    digits = value[:-1]
    result = int(digits[:-4]) + int(digits[-4:-2]) / 60 + int(digits[-2:]) / 3600
    return -result if value[-1] in 'SW' else result


def main():
    manifest = json.loads((STUDY / 'literature/source_manifest.json').read_text(encoding='utf8'))
    centers = pd.read_csv(ROOT / 'europe/reports/airports/c40_geometry/07_07_best_formula.csv').set_index('airport')
    rows = []
    for source in manifest:
        if source['status'] != 'saved':
            continue
        path = ROOT / source['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256']
        if not source.get('airport'):
            continue
        assert source['coordinate_tokens_verified']
        airport = source['airport']
        lat, lon = dms(source['arp_lat']), dms(source['arp_lon'])
        delta_lat = (lat - centers.loc[airport, 'fit_lat']) * 3600
        delta_lon = (lon - centers.loc[airport, 'fit_lon']) * 3600
        # Локальная линейная оценка для практически нулевых разностей, не геодезическая задача.
        distance = 6371000 * math.pi / (180 * 3600) * math.hypot(delta_lat, delta_lon * math.cos(math.radians(lat)))
        rows.append(dict(airport=airport, airac=source['airac'], source_id=source['id'],
                         published_lat=lat, published_lon=lon, delta_lat_arcsec=delta_lat,
                         delta_lon_arcsec=delta_lon, approximate_difference_m=distance,
                         designation='explicit_ARP' if airport == 'EGLL' else 'chart_heading_AD2_2_pending',
                         chart_information_date='not_yet_transcribed', url=source['url'], sha256=source['sha256']))
    reports = STUDY / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(reports / 'historical_arp_sources.csv', index=False)
    registry = centers[['fit_lat', 'fit_lon']].copy()
    registry['historical_status'] = 'not_checked'
    registry.loc['EGLL', 'historical_status'] = 'ARP_matches_selected_2019_2023_snapshots'
    registry.loc['EGCC', 'historical_status'] = 'heading_matches_2019_2022_AD2_2_pending'
    registry.to_csv(reports / 'arp_reference_registry.csv')
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell('# Исторические координаты: пилот EGLL и EGCC\n\nСравниваем опубликованные координаты с центрами из 07_07. Для EGLL назначение ARP явно указано; для EGCC пока проверено совпадение координат заголовка схемы. Раздел AD 2.2 еще требуется. Срезы AIRAC не подтверждают непрерывную историю всего периода. Дата информации схемы пока не перенесена в реестр.\n\nВсе записи сохраняются.'),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nroot = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'research/c40_geometry/reports/historical_arp_sources.csv').exists())\nstudy = root / 'research/c40_geometry'\npilot = pd.read_csv(study / 'reports/historical_arp_sources.csv')\npilot"),
        nbformat.v4.new_markdown_cell('Разности координат приведены в угловых секундах. Почти нулевая разность подтверждает совпадение выбранных координат, но не подтверждает точность физического положения самолета или механизм записи C40. Следующий шаг – исторические AD 2.2 и проверка остальных аэропортов.'),
        nbformat.v4.new_code_cell("pd.read_csv(study / 'reports/arp_reference_registry.csv')"),
    ])
    notebook.metadata['kernelspec'] = dict(display_name='Python 3', language='python', name='python3')
    folder = STUDY / 'notebooks'
    folder.mkdir(exist_ok=True)
    nbformat.write(notebook, folder / '01_historical_arp.ipynb')
    scope = {}
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            exec(compile(cell.source, '<arp-notebook>', 'exec'), scope)
    assert len(result) == 4 and len(registry) == 19
    print(result[['airport', 'airac', 'approximate_difference_m', 'designation']].to_string(index=False))
    print('Проверены SHA256 сохраненных PDF и код ноутбука; реестр: 19 аэропортов, 4 исторических среза.')


if __name__ == '__main__':
    main()
