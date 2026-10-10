"""Декодирование архивных сообщений и присоединение признаков без потери строк."""
from datetime import datetime, timezone
from importlib.metadata import version
import json
import platform
import pandas as pd
from audit_pilot_input import BASE, INPUT, digest
from decode_metar import decode, PARSER_VERSION


def main():
    assert version('metar') == PARSER_VERSION
    manifest = json.loads((BASE / 'reports/03_pilot_manifest.json').read_text(encoding='utf8'))
    assert digest(INPUT) == manifest['input_sha256']
    input_matches = BASE / 'data/03_pilot_matches.parquet'
    matches_digest = digest(input_matches)
    reports = []
    for source in manifest['downloads']:
        path = BASE / 'data/raw' / f"{source['airport']}_{source['month']}.csv"
        assert digest(path) == source['sha256']
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        for row in frame.itertuples(index=False):
            stamp = pd.Timestamp(row.valid, tz='UTC')
            result = decode(row.metar, row.station, stamp.to_pydatetime())
            result['archive_month'] = source['month']
            reports.append(result)
    decoded = pd.DataFrame(reports)
    assert decoded.metar_id.is_unique
    decoded['metar_utc'] = pd.to_datetime(decoded.metar_utc, utc=True).astype('datetime64[ns, UTC]')
    decoded.to_parquet(BASE / 'data/04_decoded_reports.parquet', index=False)
    matches = pd.read_parquet(input_matches)
    assert matches.source_row.is_unique
    join_keys = ['airport', 'metar_utc', 'metar']
    right = decoded.rename(columns={'station': 'airport'}).drop(columns=['archive_month'])
    assert not right.duplicated(join_keys).any()
    features = matches.merge(right, on=join_keys, how='left', validate='many_to_one',
                             suffixes=('', '_decoded'))
    assert len(features) == len(matches) and features.source_row.is_unique
    assert features.GoA.sum() == matches.GoA.sum()
    features['weather_candidate_60min'] = (features.status.eq('matched')
                                          & features.decoded_candidate.eq(True))
    # COR не теряет строку, но помечается для более осторожного сценария.
    features['weather_candidate_60min_without_cor'] = (
        features.weather_candidate_60min & ~features.correction_reported.eq(True))
    features.to_parquet(BASE / 'data/04_pilot_features.parquet', index=False)
    counts = decoded.groupby(['station', 'archive_month']).agg(
        reports=('metar_id', 'size'), decoded_ok=('decoded_candidate', 'sum'),
        cavok=('cavok', 'sum'), variable_wind=('wind_variable', 'sum'),
        gust_reported=('gust_group_reported', 'sum'), trend=('has_trend', 'sum'),
        correction=('correction_reported', 'sum'))
    counts.to_csv(BASE / 'reports/04_report_counts.csv')
    core = ['wind_speed_kt', 'wind_gust_kt', 'visibility_m', 'cloud_ceiling_ft',
            'temperature_c', 'dewpoint_c', 'qnh_hpa']
    completeness = []
    for (airport, month, label), part in features.groupby(['airport', 'month', 'GoA']):
        for field in core:
            completeness.append(dict(airport=airport, month=month, GoA=bool(label),
                field=field, rows=len(part), value_present=int(part[field].notna().sum()),
                present_percent=100*part[field].notna().mean(),
                decoded_candidate=int(part.weather_candidate_60min.sum())))
    pd.DataFrame(completeness).to_csv(BASE / 'reports/04_field_completeness.csv', index=False)
    ranges = []
    for field in core:
        values = features[field].dropna()
        ranges.append(dict(field=field, count=len(values), missing=int(features[field].isna().sum()),
            minimum=float(values.min()) if len(values) else None,
            maximum=float(values.max()) if len(values) else None))
    pd.DataFrame(ranges).to_csv(BASE / 'reports/04_numeric_ranges.csv', index=False)
    residual = decoded.loc[~decoded.decoded_candidate].copy()
    affected = features.groupby('metar_id').agg(
        matched_rows=('source_row', 'size'), matched_go_arounds=('GoA', 'sum')).reset_index()
    residual = residual.merge(affected, on='metar_id', how='left', validate='one_to_one')
    residual[['matched_rows', 'matched_go_arounds']] = residual[
        ['matched_rows', 'matched_go_arounds']].fillna(0).astype(int)
    def classify(row):
        if 'unclassified_weather_after_qnh' in row.decode_issues:
            return 'unlabelled_weather_after_qnh'
        if 'qnh_group_missing' in row.decode_issues:
            return 'temperature_and_qnh_absent'
        if 'R26/5940' in row.metar:
            return 'unsupported_runway_state_group'
        if ' ///// ' in row.metar:
            return 'unclassified_five_slashes'
        return 'unresolved_other'
    residual['audit_category'] = residual.apply(classify, axis=1)
    residual[['metar_id', 'station', 'metar_utc', 'metar', 'observed_text',
              'parser_status', 'decode_issues', 'parser_error', 'audit_category',
              'matched_rows', 'matched_go_arounds']].to_csv(
        BASE / 'reports/04_decode_residuals.csv', index=False)
    # Показывать примеры без отбора по исходу и без переноса итоговых полей в предикторы.
    selectors = {
        'COR': features.correction_reported.eq(True),
        'CAVOK': features.cavok.eq(True),
        'VRB': features.wind_variable.eq(True),
        'gust': features.gust_group_reported.eq(True),
        'direction_variation': features.wind_variation_token.notna(),
        'trend_TEMPO': features.trend_text.str.contains(r'\bTEMPO\b', na=False),
        'trend_BECMG': features.trend_text.str.contains(r'\bBECMG\b', na=False),
        'midnight_transition': features.C40_CROSS_TIME.dt.date.ne(features.metar_utc.dt.date),
        'reduced_visibility': features.visibility_relation.eq('eq') & features.visibility_m.lt(5000),
        'vertical_visibility': features.vertical_visibility_reported.eq(True),
        'recent_weather': features.recent_weather_tokens.fillna('').ne(''),
        'automatic_cloud_unknowns': features.cloud_amount_unavailable.eq(True),
        'cloud_below_100ft': features.cloud_ceiling_relation.eq('lt'),
    }
    examples, absent = [], []
    for category, mask in selectors.items():
        chosen = features.loc[mask].sort_values(['C40_CROSS_TIME', 'source_row']).head(1)
        if len(chosen):
            row = chosen.iloc[0]
            examples.append(dict(category=category, source_row=int(row.source_row), airport=row.airport,
                c40_utc=row.C40_CROSS_TIME.isoformat(), metar_utc=row.metar_utc.isoformat(),
                age_minutes=float(row.age_minutes), metar=row.metar, observed_text=row.observed_text,
                trend_text=row.trend_text, wind_token=row.wind_token,
                wind_speed_kt=row.wind_speed_kt, wind_gust_kt=row.wind_gust_kt,
                visibility_token=row.visibility_token, visibility_m=row.visibility_m,
                visibility_relation=row.visibility_relation, cloud_tokens=row.cloud_tokens,
                cloud_ceiling_ft=row.cloud_ceiling_ft, cloud_ceiling_relation=row.cloud_ceiling_relation,
                present_weather_tokens=row.present_weather_tokens,
                recent_weather_tokens=row.recent_weather_tokens, qnh_hpa=row.qnh_hpa,
                parser_status=row.parser_status))
        else:
            absent.append(category)
    pd.DataFrame(examples).to_csv(BASE / 'reports/04_examples.csv', index=False)
    assert digest(input_matches) == matches_digest
    assert digest(INPUT) == manifest['input_sha256']
    for source in manifest['downloads']:
        assert digest(BASE / 'data/raw' / f"{source['airport']}_{source['month']}.csv") == source['sha256']
    journal = dict(checked_utc=datetime.now(timezone.utc).isoformat(),
        input_sha256=manifest['input_sha256'], input_matches_sha256=matches_digest,
        original_inputs_unchanged=True, raw_reports=len(decoded),
        decoded_ok=int(decoded.decoded_candidate.sum()), residual_reports=len(residual),
        residual_matched_rows=int(residual.matched_rows.sum()),
        residual_matched_go_arounds=int(residual.matched_go_arounds.sum()),
        residual_categories=residual.audit_category.value_counts().to_dict(),
        rows=len(features), go_arounds=int(features.GoA.sum()), rows_deleted=0,
        candidate_rows_60min=int(features.weather_candidate_60min.sum()),
        candidate_rows_60min_without_cor=int(features.weather_candidate_60min_without_cor.sum()),
        candidate_go_arounds_without_cor=int(features.loc[
            features.weather_candidate_60min_without_cor, 'GoA'].sum()),
        absent_example_categories=absent, availability_verified=False,
        parser='python-metar', parser_version=version('metar'),
        environment={name: version(name) for name in ['pandas', 'pyarrow', 'requests', 'metar']},
        python_version=platform.python_version(),
        decoder_sha256=digest(BASE / 'scripts/decode_metar.py'),
        pipeline_sha256=digest(BASE / 'scripts/decode_pilot.py'),
        limitations=['Observation time does not establish availability time',
                     'Destination airport may be outcome information',
                     'No runway wind components or operational thresholds inferred',
                     'No imputation; trend and remarks excluded from observed predictors'])
    (BASE / 'reports/04_decode_manifest.json').write_text(
        json.dumps(journal, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(journal, ensure_ascii=False, indent=2))
    print(counts.to_string())
    if len(residual):
        print(residual[['metar', 'decode_issues']].head(20).to_string(index=False))


if __name__ == '__main__':
    main()
