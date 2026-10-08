"""Диагностика типов ВС в актуальной выборке C40 без изменения данных."""
import json
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from audit_feature_availability import ROOT, digest

INPUT = ROOT / 'europe/data/analytical/04_c40_available.parquet'
REPORT = ROOT / 'europe/reports/aircraft_types'


def main():
    # Сверяем вход с журналом последнего согласованного отбора
    previous = json.loads((ROOT / 'europe/reports/analytical_sample/04_10_c40_applied.json').read_text(encoding='utf8'))
    before = digest(INPUT)
    assert before == previous['included_sha256']
    fields = ['typecode', 'icaoaircrafttype', 'AC_CLASS', 'aircraft_reference_status']
    parts = {f: [] for f in fields + ['code_pairs', 'quality', 'unresolved']}
    unresolved_codes = ['82R', 'G-V', 'G550', 'GL6T', 'X']
    rows = positives = 0
    columns = fields + ['GoA', 'icao24', 'registration', 'airport', 'landing_time']
    for batch in pq.ParquetFile(INPUT).iter_batches(batch_size=100_000, columns=columns):
        frame = pa.Table.from_batches([batch]).to_pandas(ignore_metadata=True)
        rows += len(frame)
        positives += int(frame.GoA.sum())
        for f in fields:
            parts[f].append(frame.groupby(f, dropna=False).agg(rows=('GoA','size'), go_arounds=('GoA','sum')).reset_index())
        parts['code_pairs'].append(frame.groupby(['typecode','icaoaircrafttype'],dropna=False).agg(rows=('GoA','size'),go_arounds=('GoA','sum')).reset_index())
        # Маски качества пересекаются и не являются правилом удаления
        checks = {
            'all_rows': pd.Series(True,index=frame.index),
            'typecode_missing': frame.typecode.isna(),
            'icaoaircrafttype_missing': frame.icaoaircrafttype.isna(),
            'both_type_fields_missing': frame.typecode.isna() & frame.icaoaircrafttype.isna(),
            'either_type_field_missing': frame.typecode.isna() | frame.icaoaircrafttype.isna(),
            'five_unresolved_codes': frame.icaoaircrafttype.isin(unresolved_codes),
            'reference_status_unresolved': frame.aircraft_reference_status.eq('unresolved'),
            'icaoaircrafttype_nonstandard_shape': frame.icaoaircrafttype.notna() & ~frame.icaoaircrafttype.str.fullmatch(r'[ALHGT][1-8][JTPES]',na=False),
            'typecode_empty_or_whitespace': frame.typecode.notna() & frame.typecode.str.strip().eq(''),
            'icaoaircrafttype_empty_or_whitespace': frame.icaoaircrafttype.notna() & frame.icaoaircrafttype.str.strip().eq(''),
        }
        parts['quality'].append(pd.DataFrame([dict(check=k,rows=int(v.sum()),go_arounds=int(frame.loc[v,'GoA'].sum())) for k,v in checks.items()]))
        selected = frame[checks['five_unresolved_codes']].copy()
        selected['year'] = selected.landing_time.dt.year
        parts['unresolved'].append(selected)
    assert rows == previous['rows_included'] and positives == previous['positive_included']
    assert digest(INPUT) == before
    REPORT.mkdir(parents=True,exist_ok=True)
    output = {}
    for name in fields + ['code_pairs','quality']:
        keys = ['typecode','icaoaircrafttype'] if name=='code_pairs' else ['check' if name=='quality' else name]
        result = pd.concat(parts[name]).groupby(keys,dropna=False)[['rows','go_arounds']].sum().reset_index()
        result['ga_percent'] = 100*result.go_arounds/result.rows
        result = result.sort_values('rows',ascending=False)
        result.to_csv(REPORT / f'05_01_{name}.csv',index=False)
        output[name] = result
    unresolved = pd.concat(parts['unresolved'],ignore_index=True)
    unresolved.to_csv(REPORT / '05_01_unresolved_records.csv',index=False)
    detail = unresolved.groupby(['icaoaircrafttype','icao24','registration'],dropna=False).agg(rows=('GoA','size'),go_arounds=('GoA','sum'),first_landing=('landing_time','min'),last_landing=('landing_time','max')).reset_index()
    detail.to_csv(REPORT / '05_01_unresolved_aircraft.csv',index=False)
    output['unresolved_aircraft'] = detail
    affected = len(unresolved)
    affected_ga = int(unresolved.GoA.sum())
    journal = dict(input=str(INPUT.relative_to(ROOT)),input_sha256=before,input_unchanged=True,
        rows_checked=rows,go_arounds=positives,unresolved_rows=affected,unresolved_go_arounds=affected_ga,
        rows_if_five_codes_excluded=rows-affected,go_arounds_if_five_codes_excluded=positives-affected_ga,
        rows_changed=0,rows_deleted=0,filter_applied=False,
        classification_shape_is_diagnostic_only=True,reference_historical_applicability_verified=False)
    (REPORT / '05_01_audit.json').write_text(json.dumps(journal,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return output, journal
