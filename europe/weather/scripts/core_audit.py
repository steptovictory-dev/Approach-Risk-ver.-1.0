"""Отдельная проверка основных полей без исправления полного сообщения."""
import math
from decode_metar import decode, split_sections, PRESSURE, TEMPERATURE

def audit_core(raw,station,stamp,original):
    tokens=split_sections(raw)[0].split()
    positions=[i for i,t in enumerate(tokens) if PRESSURE.fullmatch(t)]
    if len(positions)!=1:
        return False,'','qnh_missing_or_multiple'
    end=positions[0]
    if sum(bool(TEMPERATURE.fullmatch(t)) for t in tokens[:end])!=1:
        return False,'','temperature_missing_or_multiple'
    prefix=' '.join(tokens[:end+1])
    checked=decode(prefix,station,stamp.to_pydatetime())
    fields=[f for f in original if f.startswith(('wind_','visibility_','cloud_',
        'present_weather_','qnh_','vertical_visibility_')) or f in [
        'temperature_token','temperature_c','dewpoint_c','cavok','gust_group_reported',
        'cumulonimbus_reported','towering_cumulus_reported']]
    def missing(value):
        return value is None or (isinstance(value,float) and math.isnan(value))
    changed=[f for f in fields if not ((missing(original[f]) and missing(checked[f])) or
        (not missing(original[f]) and not missing(checked[f]) and original[f]==checked[f]))]
    issues='value_disagreement:'+','.join(changed) if changed else checked['decode_issues']
    return bool(checked['decoded_candidate'] and not changed),prefix,issues
