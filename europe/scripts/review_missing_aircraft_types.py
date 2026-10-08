"""Разбор пропусков типов ВС и внутренних свидетельств без исправлений."""
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from audit_aircraft_types import INPUT, REPORT
from audit_feature_availability import ROOT, digest


def main():
    # Сохраняем все строки с пропуском и сравниваем их с остальной выборкой
    before = digest(INPUT)
    expected = json.loads((REPORT/'05_01_audit.json').read_text(encoding='utf8'))
    assert before == expected['input_sha256']
    cols = ['icao24','registration','typecode','icaoaircrafttype','AC_CLASS','market_segment',
            'airport','landing_time','GoA','n_attempts']
    missing_parts, group_parts = [], {f:[] for f in ['airport','year','market_segment','AC_CLASS','missing_pattern']}
    rows = positives = 0
    for batch in pq.ParquetFile(INPUT).iter_batches(batch_size=100_000,columns=cols):
        frame = pa.Table.from_batches([batch]).to_pandas(ignore_metadata=True)
        rows += len(frame)
        positives += int(frame.GoA.sum())
        a,b = frame.typecode.isna(),frame.icaoaircrafttype.isna()
        frame['missing_pattern'] = np.select([a & b,a & ~b,~a & b],['both_missing','typecode_only_missing','classification_only_missing'],default='complete')
        frame['year'] = frame.landing_time.dt.year
        missing_parts.append(frame[a|b].copy())
        for field in group_parts:
            group_parts[field].append(frame.groupby(list(dict.fromkeys([field,'missing_pattern'])),dropna=False).agg(rows=('GoA','size'),go_arounds=('GoA','sum')).reset_index())
    missing = pd.concat(missing_parts,ignore_index=True)
    assert rows == expected['rows_checked'] and positives == expected['go_arounds']
    assert len(missing)==285329 and int(missing.GoA.sum())==911
    # Исходная копия до C40 дает более широкий контекст для тех же адресов
    targets = set(missing.icao24.dropna())
    history_input = ROOT/'europe/data/working/03_c40_diagnostic_flags.parquet'
    history_hash = digest(history_input)
    prior = json.loads((ROOT/'europe/reports/preparation/07_c40_flags_applied.json').read_text(encoding='utf8'))
    assert history_hash == prior['output_sha256']
    history_parts=[]
    for batch in pq.ParquetFile(history_input).iter_batches(batch_size=100_000,columns=['icao24','registration','typecode','icaoaircrafttype','landing_time']):
        frame=pa.Table.from_batches([batch]).to_pandas(ignore_metadata=True)
        frame=frame[frame.icao24.isin(targets)]
        # Агрегация сохраняет варианты и временные границы без миллионов исходных строк
        history_parts.append(frame.groupby(['icao24','registration','typecode','icaoaircrafttype'],dropna=False).agg(rows=('landing_time','size'),first_seen=('landing_time','min'),last_seen=('landing_time','max')).reset_index())
    history=pd.concat(history_parts).groupby(['icao24','registration','typecode','icaoaircrafttype'],dropna=False).agg(rows=('rows','sum'),first_seen=('first_seen','min'),last_seen=('last_seen','max')).reset_index()
    histories={key:part for key,part in history.groupby('icao24',dropna=False)}
    # Однозначность в таблице является кандидатом, а не подтверждением идентичности борта
    assessments=[]
    for address in targets:
        part=histories[address]
        item={'icao24':address,'registrations_in_history':part.registration.nunique(dropna=True)}
        for field in ['typecode','icaoaircrafttype']:
            available=part[part[field].notna()]
            values=sorted(available[field].unique())
            item[field+'_variants']=len(values)
            item[field+'_candidate']=values[0] if len(values)==1 else None
            item[field+'_first_known']=available.first_seen.min() if len(values)==1 else pd.NaT
        assessments.append(item)
    assessments=pd.DataFrame(assessments)
    missing=missing.merge(assessments,on='icao24',how='left',validate='many_to_one')
    for field in ['typecode','icaoaircrafttype']:
        needed=missing[field].isna()
        variants=missing[field+'_variants'].fillna(0)
        missing[field+'_evidence']=np.where(~needed,'not_missing',np.where(variants==0,'no_internal_candidate',np.where(variants==1,'one_internal_candidate','conflicting_candidates')))
        missing[field+'_candidate_seen_before']=needed & (variants==1) & (missing[field+'_first_known'] < missing.landing_time)
    linked=missing[(missing.typecode.isna() & missing.typecode_variants.eq(1)) | (missing.icaoaircrafttype.isna() & missing.icaoaircrafttype_variants.eq(1))]
    assert digest(INPUT)==before and digest(history_input)==history_hash
    output={}
    for field,parts in group_parts.items():
        data=pd.concat(parts).groupby(list(dict.fromkeys([field,'missing_pattern'])),dropna=False)[['rows','go_arounds']].sum().reset_index()
        assert data.rows.sum()==rows and data.go_arounds.sum()==positives
        data.to_csv(REPORT/f'05_02_by_{field}.csv',index=False)
        output[field]=data
    evidence=missing.groupby(['missing_pattern','typecode_evidence','icaoaircrafttype_evidence'],dropna=False).agg(rows=('GoA','size'),go_arounds=('GoA','sum')).reset_index()
    evidence.to_csv(REPORT/'05_02_internal_evidence.csv',index=False)
    history.to_csv(REPORT/'05_02_history_variants.csv',index=False)
    positive_records=missing[missing.GoA].copy()
    positive_records.to_csv(REPORT/'05_02_positive_missing_records.csv',index=False)
    destination=ROOT/'europe/data/review/05_02_missing_aircraft_types.parquet'
    missing.to_parquet(destination,index=False,compression='gzip')
    audit=dict(input_sha256=before,history_input_sha256=history_hash,input_unchanged=True,history_input_unchanged=True,
        rows_checked=rows,positive_total=positives,missing_rows=len(missing),missing_positive=int(missing.GoA.sum()),
        missing_addresses=int(missing.icao24.nunique()),positive_missing_addresses=int(positive_records.icao24.nunique()),
        rows_with_any_unique_internal_candidate=len(linked),positive_with_any_unique_internal_candidate=int(linked.GoA.sum()),
        registration_missing_rows=int(missing.registration.isna().sum()),registration_missing_positive=int(missing.loc[missing.registration.isna(),'GoA'].sum()),
        rows_changed=0,rows_deleted=0,candidates_applied=False,
        internal_history_does_not_verify_aircraft_identity_or_historical_feature_availability=True)
    (REPORT/'05_02_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return output,evidence,audit
