"""Сохраняет первичные источники C40 и фиксирует результат каждой загрузки."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import sys
import requests

ROOT=Path(__file__).resolve().parents[3]
STUDY=ROOT/'research/c40_geometry'
FILES=STUDY/'literature/files'
SOURCES=[
    dict(id='figuet_2023',title='Analysing the Impact of Go-Around Occurrences at Large European Airports',date='2023-12-14',url='https://journals.open.tudelft.nl/joas/article/download/7219/5898',filename='Figuet_et_al_2023_go_arounds.pdf',local='docs/sources/files/Figuet_et_al_2023_go_arounds.pdf',role='Происхождение C40 и объединение OpenSky/EU-NM; разделы 2.2–2.3',locator='PDF стр. 4, печатная стр. 4'),
    dict(id='eurocontrol_2011',title='Measuring Operational ANS Performance at Airports',date='2011-05-09',url='https://www.eurocontrol.int/sites/default/files/content/documents/single-sky/pru/publications/other/ans-airport-indicators-technical-note.pdf',filename='EUROCONTROL_2011_airport_indicators.pdf',role='Историческое описание ASMA 40 NM и индикаторов',locator='Раздел 3.4.3.2; PDF стр. 28, печатная стр. 24'),
    dict(id='eurocontrol_opadd_2021',title='EUROCONTROL Guidelines Operating Procedures AIS Dynamic Data, edition 4.1',date='2021',url='https://www.eurocontrol.int/sites/default/files/2021-07/eurocontrol-guidelines-opadd-ed4-1.pdf',filename='EUROCONTROL_OPADD_4_1_2021.pdf',role='Географическая ссылка NOTAM: координаты и область влияния',locator='Разделы 2.3.11–2.3.13; PDF стр. 33–35'),
    dict(id='eurocontrol_avenir_2025',title='AVENIR Working Group Pillar 1 Final Report',date='2025-07',url='https://www.eurocontrol.int/sites/default/files/2025-09/avenir-wg-pillar-1-final-report.pdf',filename='EUROCONTROL_AVENIR_2025.pdf',role='Изменение методики ASMA: последний/первый вход; поздний контекст, не доказательство реализации файла 2019–2023',locator='Раздел 5.4, PDF стр. 43'),
    dict(id='vincenty_1975',title='Direct and Inverse Solutions of Geodesics on the Ellipsoid with Application of Nested Equations',date='1975',url='https://www.ngs.noaa.gov/PUBS_LIB/inverse.pdf',filename='Vincenty_1975_inverse.pdf',role='Математика обратной геодезической задачи',locator='Формулы обратной задачи'),
    dict(id='karney_2013',title='Algorithms for geodesics',date='2013',url='https://arxiv.org/pdf/1109.4448',filename='Karney_algorithms_geodesics_arxiv_1109_4448.pdf',role='Современная геодезическая методика; arXiv-версия может отличаться от журнальной',locator='Версию проверять по первой странице'),
    dict(id='nats_egll_2019',title='UK AIP EGLL remote de-icing areas chart; ARP in heading',date='2019-12-05',airac='2019-12-05',airport='EGLL',arp_lat='512839N',arp_lon='0002741W',url='https://www.aurora.nats.co.uk/htmlAIP/Publications/2019-12-05-AIRAC/graphics/101582.pdf',filename='NATS_EGLL_AIRAC_2019_12_05_chart.pdf',role='Исторический официальный ARP в заголовке схемы; снимок AIRAC, не непрерывная история',locator='PDF стр. 1, заголовок ARP'),
    dict(id='nats_egll_2023',title='UK AIP EGLL aerodrome chart',date='2023-05-18',airac='2023-05-18',airport='EGLL',arp_lat='512839N',arp_lon='0002741W',url='https://www.aurora.nats.co.uk/htmlAIP/Publications/2023-05-18-AIRAC/graphics/321412.pdf',filename='NATS_EGLL_AIRAC_2023_05_18_chart.pdf',role='Исторический официальный ARP для позднего среза',locator='PDF стр. 1, заголовок ARP; дата информации схемы отличается от AIRAC'),
    dict(id='nats_egcc_2019',title='UK AIP EGCC aerodrome chart',date='2019-11-07',airac='2019-11-07',airport='EGCC',arp_lat='532114N',arp_lon='0021630W',url='https://www.aurora.nats.co.uk/htmlAIP/Publications/2019-11-07-AIRAC/graphics/123929.pdf',filename='NATS_EGCC_AIRAC_2019_11_07_chart.pdf',role='Исторические координаты в заголовке схемы; проверить назначение через AD 2.2',locator='PDF стр. 1, заголовок координат'),
    dict(id='nats_egcc_2022',title='UK AIP EGCC aerodrome chart',date='2022-12-01',airac='2022-12-01',airport='EGCC',arp_lat='532114N',arp_lon='0021630W',url='https://www.aurora.nats.co.uk/htmlAIP/Publications/2022-12-01-AIRAC/graphics/318470.pdf',filename='NATS_EGCC_AIRAC_2022_12_01_chart.pdf',role='Поздний официальный срез; позже последней сохраненной записи C40 EGCC, не подтверждает весь ранний период',locator='PDF стр. 1, заголовок координат'),
]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    FILES.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for source in SOURCES:
        entry=dict(source);path=FILES/source['filename']
        entry['checked_utc']=datetime.now(timezone.utc).isoformat()
        try:
            if path.exists():entry['acquisition']='existing_local_verified'
            elif source.get('local') and (ROOT/source['local']).exists():
                shutil.copyfile(ROOT/source['local'],path);entry['acquisition']='copy_of_existing_source'
                entry['original_sha256']=sha(ROOT/source['local'])
            else:
                response=requests.get(source['url'],timeout=35,headers={'User-Agent':'ApproachRiskResearch/1.0'})
                entry['http_status']=response.status_code;entry['final_url']=response.url
                if response.status_code == 403 and source['id'] == 'eurocontrol_opadd_2021':
                    entry['primary_http_status'] = response.status_code
                    mirror = 'https://www.icao.int/sites/default/files/APAC/Documents/EUROCONTROL-Operating-Procedures-for-AIS-Dynamic-Data-OPADD-Edition-4.1.pdf'
                    response = requests.get(mirror, timeout=35)
                    entry['official_mirror_url'] = mirror
                    entry['http_status'] = response.status_code
                    entry['final_url'] = response.url
                if response.status_code == 403 and source['id'] == 'eurocontrol_2011':
                    mirror = 'https://www.eurocontrol.int/archive_download/all/node/10686'
                    response = requests.get(mirror, timeout=35)
                    entry['official_mirror_url'] = mirror
                    entry['http_status'] = response.status_code
                    entry['final_url'] = response.url
                response.raise_for_status()
                if b'%PDF-' not in response.content[:1024]:raise ValueError('Ответ не является PDF')
                path.write_bytes(response.content);entry['acquisition']='downloaded'
            if b'%PDF-' not in path.read_bytes()[:1024]:raise ValueError('Локальный файл не является PDF')
            entry.update(status='saved',file=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path))
            try:
                from pypdf import PdfReader
                reader=PdfReader(path);entry['pages']=len(reader.pages)
                text='\n'.join(page.extract_text() or '' for page in reader.pages)
                textpath=path.with_suffix('.txt');textpath.write_text(text,encoding='utf8')
                entry['text_file']=str(textpath.relative_to(ROOT))
                if source.get('airport'):
                    flat=''.join(text.split())
                    entry['coordinate_tokens_verified']=source['arp_lat'] in flat and source['arp_lon'] in flat
            except ImportError:entry['pdf_text_status']='reader_not_available'
        except Exception as error:
            entry.update(status='failed',error=f'{type(error).__name__}: {error}')
        manifest.append(entry)
        (STUDY/'literature/source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(source['id'],entry['status'],entry.get('bytes'),entry.get('coordinate_tokens_verified'),flush=True)
    saved=sum(entry['status']=='saved' for entry in manifest)
    print(f'Сохранено {saved}/{len(manifest)} источников. Неудачи явно записаны в реестре.',flush=True)


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf8')
    main()
