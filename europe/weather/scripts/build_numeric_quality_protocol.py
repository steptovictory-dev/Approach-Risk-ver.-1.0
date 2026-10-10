"""Подготовить предложение; не изменять исходные или модельные признаки."""
import hashlib
import json
import pandas as pd
from audit_pilot_input import BASE

R = BASE / 'reports'
source = R / '15_numeric_candidates.csv'
extremes = pd.read_csv(R / '12_full_extreme_value_examples.csv').set_index('metar_id')
candidates = pd.read_csv(source)
preview = []
for metar_id, group in candidates.groupby('metar_id', sort=True):
    row = group.iloc[0]
    assert group[['linked_rows','linked_go_arounds','metar']].nunique().eq(1).all()
    reasons, fields = [], []
    if metar_id in extremes.index:
        reason = extremes.loc[metar_id, 'review_reason']
        mapping = {
            'qnh_hpa_minimum': ('qnh_audited_extreme', ['qnh_hpa']),
            'qnh_hpa_maximum': ('qnh_audited_extreme', ['qnh_hpa']),
            'dewpoint_c_minimum': ('dewpoint_audited_extreme', ['dewpoint_c']),
            'cloud_ceiling_ft_maximum': ('cloud_height_audited_extreme', ['cloud_ceiling_ft','lowest_known_cloud_base_ft']),
        }
        flag, mask = mapping[reason]
        reasons.append(flag)
        fields.extend(mask)
    if group.reason.eq('dewpoint_above_temperature').any():
        reasons.append('temperature_pair_inconsistent')
        fields.extend(['temperature_c','dewpoint_c'])
    assert reasons and fields
    preview.append(dict(metar_id=metar_id, station=row.station, metar_utc=row.metar_utc,
        metar=row.metar, numeric_quality_reasons=';'.join(reasons),
        proposed_mask_fields=';'.join(sorted(set(fields))), linked_rows=int(row.linked_rows),
        linked_go_arounds=int(row.linked_go_arounds), status='proposed_not_applied'))
table = pd.DataFrame(preview)
assert table.metar_id.is_unique and len(table)==18
assert table.linked_rows.sum()==44 and table.linked_go_arounds.sum()==0
table.to_csv(R / '17_numeric_quality_preview.csv', index=False)
protocol = dict(version='numeric-quality-v1-draft', status='proposed_pending_user_decision',
    input_sha256=json.loads((R/'11_full_protocol.json').read_text(encoding='utf8'))['input_sha256'],
    candidates_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    unique_candidate_messages=len(table), baseline_linked_rows=int(table.linked_rows.sum()),
    baseline_linked_go_arounds=int(table.linked_go_arounds.sum()),
    baseline=dict(max_age_minutes=60, assumed_latency_minutes=0, include_cor=True),
    rules=dict(audited_extremes='proposed_field_mask_by_metar_id',
        dewpoint_above_temperature='proposed_mask_both_temperature_and_dewpoint',
        qnh_diagnostic_interval_hpa=[850,1100], diagnostic_range_alone='review_only'),
    source_values_changed=False, feature_masks_applied=False, rows_deleted=0,
    quality_flags_as_predictors_approved=False, training_executed=False,
    retrospective_audit_registry=True, future_neighbors_as_features=False)
(R/'17_numeric_quality_protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
print(json.dumps(protocol, ensure_ascii=False, indent=2), flush=True)
