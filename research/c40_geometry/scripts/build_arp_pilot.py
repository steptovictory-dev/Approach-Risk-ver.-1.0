"""Сверка исторических схем с восстановленными центрами, без изменения данных."""
from pathlib import Path
import hashlib
import json
import math
import re
import pandas as pd
import nbformat
from notebook_output import write_overview

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
                         designation='explicit_ARP' if airport == 'EGLL' else 'chart_heading_AD2_2_supported',
                         chart_information_date={'nats_egll_2019':'2019-01-08','nats_egll_2023':'2022-10-18','nats_egcc_2019':'2019-06-20','nats_egcc_2022':'2022-09-23'}[source['id']],
                         publication_date=None, source_type='chart', url=source['url'], sha256=source['sha256']))
    ad_manifest = json.loads((STUDY / 'literature/historical_ad_manifest.json').read_text(encoding='utf8'))
    for source in ad_manifest:
        if source['status'] != 'saved':
            continue
        assert source['effective_date_matches_requested']
        path = ROOT / source['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256']
        context = source['arp_context']
        lat_token = re.search(r'Lat:\s*(\d{6}[NS])', context).group(1)
        lon_token = re.search(r'Long:\s*(\d{7}[EW])', context).group(1)
        lat, lon = dms(lat_token), dms(lon_token)
        airport = source['airport']
        dl = (lat-centers.loc[airport,'fit_lat'])*3600
        dn = (lon-centers.loc[airport,'fit_lon'])*3600
        difference = 6371000*math.pi/(180*3600)*math.hypot(dl,dn*math.cos(math.radians(lat)))
        rows.append(dict(airport=airport,airac=source['airac'],source_id=f"ad2_{airport}_{source['airac']}",
                         published_lat=lat,published_lon=lon,delta_lat_arcsec=dl,delta_lon_arcsec=dn,
                         approximate_difference_m=difference,designation='explicit_ARP_AD2_2',
                         chart_information_date=None,publication_date=source['document_metadata'].get('DC.date'),
                         source_type='AD2_HTML',url=source['url'],sha256=source['sha256']))
    reports = STUDY / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(reports / 'historical_arp_sources.csv', index=False)
    registry = centers[['fit_lat', 'fit_lon']].copy()
    registry['historical_status'] = 'not_checked'
    registry.loc['EGLL', 'historical_status'] = 'ARP_matches_selected_2019_2023_snapshots'
    registry.loc['EGCC', 'historical_status'] = 'ARP_matches_selected_AD2_2_2019_2021_snapshots'
    registry.to_csv(reports / 'arp_reference_registry.csv')
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell('# Исторические координаты: пилот EGLL и EGCC\n\nНазначение ARP подтверждено по AD 2.2 обоих аэропортов. EGCC: середина ВПП 05L/23R; EGLL: середина ВПП 09L/27R. Шесть страниц HTML имеют внутреннюю дату вступления в силу, совпадающую с запрошенным AIRAC. Четыре PDF-схемы проверены отдельно; их даты информации перенесены в реестр. Печатные версии AD 2 по ссылкам из HTML вернули 404, HTML сохранены и проверены.\n\nСрезы AIRAC не подтверждают непрерывную историю всего периода. Все записи сохраняются.'),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nroot = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'research/c40_geometry/reports/historical_arp_sources.csv').exists())\nstudy = root / 'research/c40_geometry'\npilot = pd.read_csv(study / 'reports/historical_arp_sources.csv')\npilot"),
        nbformat.v4.new_markdown_cell('Разности координат приведены в угловых секундах. Почти нулевая разность подтверждает совпадение выбранных координат, но не подтверждает точность физического положения самолета или механизм записи C40. Первый проверенный документ EGCC имеет AIRAC 2019-11-07: ранняя часть 2019 года пока не покрыта этой независимой проверкой. Следующий шаг – остальные аэропорты и недостающие временные срезы.'),
        nbformat.v4.new_code_cell("pd.read_csv(study / 'reports/arp_reference_registry.csv')"),
    ])
    notebook.metadata['kernelspec'] = dict(display_name='Python 3', language='python', name='python3')
    folder = STUDY / 'notebooks'
    folder.mkdir(exist_ok=True)
    write_overview(notebook, folder / '01_historical_arp.ipynb')
    scope = {}
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            exec(compile(cell.source, '<arp-notebook>', 'exec'), scope)
    assert len(result) >= 10 and len(registry) == 19
    print(result[['airport', 'airac', 'approximate_difference_m', 'designation']].to_string(index=False))
    print(f'Проверены SHA256 источников и код ноутбука; реестр: 19 аэропортов, {len(result)} документов.')


if __name__ == '__main__':
    main()
