"""Разбор всех 147 пар на полном составе исходных полей."""
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from audit_feature_availability import INPUT, ROOT, REPORT, digest


def main():
    before = digest(INPUT)
    audit = json.loads((REPORT / '04_04_matching_audit.json').read_text(encoding='utf8'))
    assert before == audit['input_sha256']
    projection = pq.read_table(ROOT / 'europe/data/review/04_04_repeated_c40_projection.parquet').to_pandas()
    positions = np.sort(projection.source_row_position.to_numpy())
    assert len(positions) == len(set(positions)) == 294
    pieces = []
    offset = 0
    # Читаем полные строки по исходным позициям, без восстановления из CSV
    for batch in pq.ParquetFile(INPUT).iter_batches(batch_size=100_000):
        indices = positions[(positions>=offset)&(positions<offset+len(batch))]-offset
        if len(indices):
            selected = pa.Table.from_batches([batch]).take(pa.array(indices))
            pieces.append(selected.append_column('source_row_position',pa.array(indices+offset)))
        offset += len(batch)
    assert offset == audit['rows_checked']
    raw = pa.concat_tables(pieces).replace_schema_metadata(None)
    frame = raw.to_pandas(ignore_metadata=True)
    assert set(frame.source_row_position) == set(positions)
    pairs, differences = [], []
    keys = ['icao24','airport','C40_CROSS_TIME']
    for number, (key, group) in enumerate(frame.groupby(keys,sort=True),1):
        assert len(group)==2
        ordered = group.sort_values('landing_time')
        a,b = list(ordered.itertuples(index=True))
        ta,tb = pd.to_datetime(a.attempt_times,utc=True),pd.to_datetime(b.attempt_times,utc=True)
        different = [field for field in raw.column_names if field != 'source_row_position'
                     and not raw[field].slice(a.Index,1).equals(raw[field].slice(b.Index,1))]
        for field in different:
            differences.append(dict(pair_id=number,field=field,
                                    early_value=str(raw[field][a.Index].as_py()),
                                    late_value=str(raw[field][b.Index].as_py())))
        # Группы отражают наблюдаемые метки, а не реконструированное число уходов
        label = 'both_false' if not a.GoA and not b.GoA else 'early_true_late_false' if a.GoA and not b.GoA else 'other'
        pairs.append(dict(pair_id=number,icao24=key[0],airport=key[1],C40_CROSS_TIME=key[2],
            early_position=a.source_row_position,late_position=b.source_row_position,
            early_flight_id=a.flight_id,late_flight_id=b.flight_id,
            early_callsign=a.callsign,late_callsign=b.callsign,
            early_registration=a.registration,late_registration=b.registration,
            early_typecode=a.typecode,late_typecode=b.typecode,
            early_landing=a.landing_time,late_landing=b.landing_time,
            early_first_attempt=ta[0],early_last_attempt=ta[-1],
            late_first_attempt=tb[0],late_last_attempt=tb[-1],
            early_n_attempts=a.n_attempts,late_n_attempts=b.n_attempts,
            early_GoA=a.GoA,late_GoA=b.GoA,label_pattern=label,
            landing_gap_minutes=(b.landing_time-a.landing_time).total_seconds()/60,
            early_landing_to_late_first_minutes=(tb[0]-a.landing_time).total_seconds()/60,
            early_last_attempt_to_landing_minutes=(a.landing_time-ta[-1]).total_seconds()/60,
            late_last_attempt_to_landing_minutes=(b.landing_time-tb[-1]).total_seconds()/60,
            early_c40_to_first_minutes=(ta[0]-a.C40_CROSS_TIME).total_seconds()/60,
            late_c40_to_first_minutes=(tb[0]-b.C40_CROSS_TIME).total_seconds()/60,
            sequential_id_suffix=int(b.flight_id.rsplit('_',1)[-1])==int(a.flight_id.rsplit('_',1)[-1])+1,
            same_callsign='callsign' not in different,same_registration='registration' not in different,
            same_typecode='typecode' not in different,same_ILS='ILS' not in different,
            same_c40_geometry=not any(f in different for f in ['C40_BEARING','C40_CROSS_LAT','C40_CROSS_LON']),
            different_fields=';'.join(different),decision='retain_both_pending_independent_validation'))
    pairs = pd.DataFrame(pairs)
    differences = pd.DataFrame(differences)
    assert len(pairs)==147 and (pairs.early_n_attempts + pairs.late_n_attempts).sum()==297
    assert pairs.same_callsign.all() and pairs.same_registration.all() and pairs.sequential_id_suffix.all()
    assert pairs.same_c40_geometry.all() and frame.landing_time.dt.year.eq(2019).all()
    assert digest(INPUT)==before
    review = ROOT / 'europe/data/review/04_05_c40_pair_full_records.parquet'
    pq.write_table(raw,review,compression='gzip')
    pairs.to_csv(REPORT / '04_05_pair_review.csv',index=False)
    differences.to_csv(REPORT / '04_05_pair_field_differences.csv',index=False)
    pairs.groupby(['airport','label_pattern']).agg(pairs=('pair_id','size'),
        landing_gap_median_minutes=('landing_gap_minutes','median')).reset_index().to_csv(REPORT/'04_05_pair_airports.csv',index=False)
    pairs.loc[pairs.label_pattern!='both_false'].to_csv(REPORT/'04_05_pairs_with_goa.csv',index=False)
    summary = dict(input_sha256=before,input_unchanged=True,pairs_reviewed=len(pairs),rows_reviewed=len(frame),
        all_pairs_year=2019,all_pairs_same_callsign=True,all_pairs_same_registration=True,
        all_pairs_sequential_identifier_suffix=True,all_pairs_same_c40_geometry=True,
        label_patterns=pairs.label_pattern.value_counts().to_dict(),
        pairs_with_different_typecodes=int((~pairs.same_typecode).sum()),
        pairs_with_different_ILS_lists=int((~pairs.same_ILS).sum()),
        all_late_records_one_attempt=bool(pairs.late_n_attempts.eq(1).all()),
        all_late_first_attempts_after_early_landing=bool((pairs.early_landing_to_late_first_minutes>0).all()),
        landing_gap_minutes={'min':float(pairs.landing_gap_minutes.min()),
                             'median':float(pairs.landing_gap_minutes.median()),'max':float(pairs.landing_gap_minutes.max())},
        pairs_with_c40_after_early_first=int((pairs.early_c40_to_first_minutes<0).sum()),
        rows_changed=0,rows_deleted=0,trajectories_available=False,
        physical_identity_confirmed=False,decision='retain both pending independent validation')
    (REPORT/'04_05_pair_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return pairs,differences
