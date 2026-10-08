"""Разбор обозначений ВПП LSGG и локальных материалов без исправлений."""
import json
import hashlib
import zipfile
from collections import Counter
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from audit_feature_availability import ROOT, digest
from audit_attempt_intervals import INPUT, REPORT


def main():
    before=digest(INPUT)
    previous=json.loads((REPORT/'06_01_audit.json').read_text(encoding='utf8'))
    assert before==previous['input_sha256']
    target_tables=[]
    patterns=Counter()
    events=Counter()
    attempts=Counter()
    rows=ga=0
    cols=['flight_id','icao24','airport','landing_time','attempt_times','ILS','n_attempts','GoA']
    for batch in pq.ParquetFile(INPUT).iter_batches(batch_size=100_000,columns=cols):
        table=pa.Table.from_batches([batch])
        frame=table.to_pandas(ignore_metadata=True)
        frame=frame[frame.airport.eq('LSGG')]
        rows+=len(frame)
        ga+=int(frame.GoA.sum())
        for row in frame.itertuples(index=False):
            sequence=tuple(row.ILS)
            patterns[(sequence,bool(row.GoA))]+=1
            for label in sequence: attempts[(label,bool(row.GoA))]+=1
            for label in set(sequence): events[(label,bool(row.GoA))]+=1
        selected=frame[frame.flight_id.eq('TAP948_37820') & frame.icao24.eq('495295') & frame.landing_time.eq(pd.Timestamp('2019-03-01 16:29:15',tz='UTC'))]
        if len(selected): target_tables.append(selected)
    target=pd.concat(target_tables,ignore_index=True)
    assert len(target)==1 and list(target.iloc[0].ILS)==['22R','22']
    target.to_csv(REPORT/'06_02_case_record.csv',index=False)
    target.to_parquet(ROOT/'europe/data/review/06_02_lsgg_case.parquet',index=False)
    label_frame=pd.DataFrame([dict(runway_label=label,GoA=positive,observations=v,attempts=attempts[(label,positive)]) for (label,positive),v in events.items()])
    label_frame.to_csv(REPORT/'06_02_lsgg_runway_labels.csv',index=False)
    pattern_frame=pd.DataFrame([dict(runway_sequence='|'.join(sequence),GoA=positive,observations=v) for (sequence,positive),v in patterns.items()]).sort_values('observations',ascending=False)
    pattern_frame.to_csv(REPORT/'06_02_lsgg_runway_patterns.csv',index=False)
    # Опубликованный архив не является независимым источником измерений
    archive_path=ROOT/'archive/2019_experiment/data/raw/validation_sample.zip'
    archive_status=[]
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.namelist():
            if '/LSGG_' in member:
                content=archive.read(member)
                archive_status.append(dict(member=member,sha256=hashlib.sha256(content).hexdigest(),target_identifier_found=b'TAP948_37820' in content))
    assert digest(INPUT)==before
    sources=[
        dict(url='https://www.gva.ch/en/Downloads/Aeroport/2017_Rapport_Annuel.pdf?ext=.',publication_year=2017,printed_page=38,evidence='Airport reports end of aeronautical operations on grass strip in 2017',access='official indexed passage reviewed; direct PDF download returned 404'),
        dict(url='https://www.gva.ch/getmedia/7e61960f-cf6a-41fd-b489-ad788fbc39c3/2018_Rapport-Annuel_EN.pdf?ext=.pdf',publication_year=2018,printed_page=34,evidence='Main runway renamed from 05-23 to 04-22 during night 12-13 September 2018; report states a NOTAM was published',access='official indexed passage reviewed; direct PDF download returned 404'),
        dict(url='https://www.eurocontrol.int/archive_download/all/node/12569',data_year=2019,evidence='Geneva single runway mixed operation with RWY04 and RWY22',access='official indexed/search and PDF text available through web tool; local download returned 403'),
        dict(url='https://skybriefing.com/documents/10156/94911/skybriefing_Folder/fbf81852-6f37-48f8-8fb2-eb2f6fe5846c',evidence='Official service advertises NOTAM briefing; 2019 target notices not obtained',access='web tool'),
        dict(url='https://traffic-viz.github.io/api_reference/traffic.algorithms.navigation.html',evidence='LandingAlignedOnILS detects aligned trajectory segments and assigns runway numbers',access='current documentation; historical algorithm version not established'),
    ]
    result=dict(input_sha256=before,input_unchanged=True,airport='LSGG',airport_rows=rows,airport_go_arounds=ga,
        case_flight_id='TAP948_37820',case_gap_seconds=12,case_runway_labels=['22R','22'],
        case_GoA=True,archived_materials=archive_status,
        historical_notam_obtained=False,notam_target_utc='2019-03-01 16:29:03 to 16:29:15',
        notam_search_result='No primary historical NOTAM for target interval retrieved; absence of notice not established',
        exact_historical_AIP_runway_designations_verified=False,
        hypothesis='Potential stale or inappropriate runway reference and overlapping event segmentation; not confirmed',
        rows_changed=0,rows_deleted=0,runway_corrected=False,label_corrected=False,
        checked_date='2026-10-08',sources=sources)
    (REPORT/'06_02_case_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return target,label_frame,pattern_frame,result
