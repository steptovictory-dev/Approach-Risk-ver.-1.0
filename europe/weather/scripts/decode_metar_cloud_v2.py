"""Версия cloud-v2: короткие CB/TCU и сравнение облачных пропусков.

python-metar проверяет структуру; исходные токены задают единицы, пропуски
и границы. Его числовые значения не подменяют специальные коды видимости.
"""
import hashlib
import json
import math
import re
from metar import Metar
from decode_metar import decode as decode_v1

DECODER_VERSION = 'cloud-v2'

PARSER_VERSION = '2.0.1'
BOUNDARY = re.compile(r'\b(?:TEMPO|BECMG|NOSIG|FCST|INTER|PROB\d{2}|(?:FM|TL|AT)\d{4}|RMK)\b')
WIND = re.compile(r'^(?P<direction>\d{3}|VRB|///)(?P<speed>P?\d{2,3}|//)'
                  r'(?:G(?P<gust>P?\d{2,3}|//))?(?P<unit>KT|MPS|KMH)$')
VARIATION = re.compile(r'^(\d{3})V(\d{3})$')
CLOUD = re.compile(r'^(FEW|SCT|BKN|OVC|VV|///)(\d{3}|///)(CB|TCU|///)?$')
TEMPERATURE = re.compile(r'^(?P<temperature>M?\d{2}|//)/(?P<dewpoint>M?\d{2}|//)$')
PRESSURE = re.compile(r'^[QA](?:\d{4}|////)$')
WEATHER = re.compile(r'^(?:[+-]?(?:VC)?(?:MI|PR|BC|DR|BL|SH|TS|FZ)?'
                     r'(?:DZ|RA|SN|SG|IC|PL|GR|GS|UP|BR|FG|FU|VA|DU|SA|HZ|PY|PO|SQ|FC|SS|DS){1,3}'
                     r'|(?:VC)?TS|VCSH)$')


def split_sections(raw):
    # Текст источника хранится отдельно; нормализуется только рабочая копия.
    normalized = ' '.join(raw.strip().rstrip('=').split())
    remarks_match = re.search(r'\bRMK\b', normalized)
    if remarks_match:
        main = normalized[:remarks_match.start()].strip()
        remarks = normalized[remarks_match.start():]
    else:
        main, remarks = normalized, ''
    trend = BOUNDARY.search(main)
    if trend:
        return main[:trend.start()].strip(), main[trend.start():].strip(), remarks
    return main, '', remarks


def number_bound(token):
    if token is None or token == '//':
        return None, 'missing'
    return float(token.lstrip('P')), 'gt' if token.startswith('P') else 'eq'


def decode(raw, station, observation_time):
    observed, trend, remarks = split_sections(raw)
    tokens = observed.split()
    cloud_end=next((i for i,t in enumerate(tokens) if TEMPERATURE.fullmatch(t) or PRESSURE.fullmatch(t)),len(tokens))
    # Все неподлежащие этому исправлению сообщения воспроизводятся старой версией.
    if not any(t in ['///CB','///TCU','/////////'] for t in tokens[:cloud_end]):
        return decode_v1(raw,station,observation_time)
    issues = []
    result = dict(metar_id=hashlib.sha256(
        (station+'|'+observation_time.isoformat()+'|'+raw).encode('utf8')).hexdigest(),
        station=station, metar_utc=observation_time, metar=raw,
        observed_text=observed, trend_text=trend, remarks_text=remarks,
        has_trend=bool(trend), has_remarks=bool(remarks),
        correction_reported='COR' in tokens, auto_reported='AUTO' in tokens)
    # Основные погода/облака идут до температуры и QNH. Не переносить в них
    # подозрительные группы после QNH с отсутствующим индикатором тренда.
    main_end = next((i for i, t in enumerate(tokens)
                     if TEMPERATURE.fullmatch(t) or PRESSURE.fullmatch(t)), len(tokens))
    main_tokens = tokens[:main_end]
    pressure_index = next((i for i, t in enumerate(tokens) if PRESSURE.fullmatch(t)), None)
    tail = tokens[pressure_index+1:] if pressure_index is not None else []
    suspicious_tail = any(WEATHER.fullmatch(t) or CLOUD.fullmatch(t)
        or WIND.fullmatch(t) or re.fullmatch(r'\d{4}', t) for t in tail)
    result['unclassified_tail_text'] = ' '.join(tail) if suspicious_tail else ''
    if suspicious_tail:
        issues.append('unclassified_weather_after_qnh')
    try:
        parsed = Metar.Metar(observed, year=observation_time.year,
                             month=observation_time.month, strict=True)
        if parsed.station_id != station or parsed.time != observation_time.replace(tzinfo=None):
            issues.append('parser_station_or_time_mismatch')
        error = ''
    except Metar.ParserError as exc:
        parsed, error = None, str(exc)
        issues.append('parser_error')

    winds = [(i, WIND.fullmatch(t)) for i, t in enumerate(main_tokens) if WIND.fullmatch(t)]
    wind = winds[0][1] if len(winds) == 1 else None
    if len(winds) != 1:
        issues.append('wind_group_missing_or_multiple')
    direction = wind['direction'] if wind else None
    speed, speed_bound = number_bound(wind['speed'] if wind else None)
    gust, gust_bound = number_bound(wind['gust'] if wind else None)
    unit = wind['unit'] if wind else None
    factor = {'KT': 1.0, 'MPS': 3600/1852, 'KMH': 1000/1852}.get(unit)
    calm = wind is not None and direction == '000' and speed == 0
    direction_deg = float(direction) if direction and direction.isdigit() and not calm else None
    if direction_deg is not None and not 0 <= direction_deg <= 360:
        issues.append('wind_direction_out_of_code_range')
    # Условия отсутствия скорости не маскируются нулем.
    result.update(wind_token=wind[0] if wind else None, wind_direction_token=direction,
        wind_unit=unit, wind_speed_raw=speed, wind_speed_relation=speed_bound,
        wind_gust_raw=gust, wind_gust_relation=gust_bound,
        wind_speed_kt=speed*factor if speed is not None else None,
        wind_gust_kt=gust*factor if gust is not None else None,
        wind_direction_deg=direction_deg,
        wind_variable=direction == 'VRB' if direction and direction != '///' else None,
        wind_calm=calm if direction and direction != '///' and speed is not None else None,
        gust_group_reported=wind['gust'] is not None if wind else None)
    variations = [VARIATION.fullmatch(t) for t in tokens if VARIATION.fullmatch(t)]
    variation = variations[0] if len(variations) == 1 else None
    if len(variations) > 1:
        issues.append('multiple_wind_variation_groups')
    result.update(wind_variation_token=variation[0] if variation else None,
        wind_variable_from_deg=float(variation[1]) if variation else None,
        wind_variable_to_deg=float(variation[2]) if variation else None)

    # Положение сразу после ветра/переменности отличает prevailing visibility от RVR и QNH.
    vi = winds[0][0]+1 if len(winds) == 1 else len(tokens)
    if vi < len(tokens) and VARIATION.fullmatch(tokens[vi]):
        vi += 1
    vis_token = tokens[vi] if vi < len(tokens) else None
    vis_value, vis_relation, vis_unit = None, 'missing', None
    cavok = vis_token == 'CAVOK'
    # NDV не меняет числовую часть и границу специального кода видимости.
    vis_code = vis_token[:-3] if vis_token and vis_token.endswith('NDV') else vis_token
    if cavok or vis_code == '9999':
        vis_value, vis_relation, vis_unit = 10000.0, 'ge', 'M'
    elif vis_code == '0000':
        vis_value, vis_relation, vis_unit = 50.0, 'lt', 'M'
    elif vis_token and re.fullmatch(r'\d{4}(?:NDV)?', vis_token):
        vis_value, vis_relation, vis_unit = float(vis_token[:4]), 'eq', 'M'
    elif vis_code == '////':
        vis_relation, vis_unit = 'missing', 'M'
    else:
        # Дробные SM не встречались в европейском пилоте; не угадывать их по позиции.
        issues.append('visibility_format_outside_pilot_scope')
    result.update(visibility_token=vis_token, visibility_unit=vis_unit,
        visibility_m=vis_value, visibility_relation=vis_relation, cavok=cavok,
        visibility_ndv=vis_token.endswith('NDV') if vis_token else None)

    weather = [t for t in main_tokens if WEATHER.fullmatch(t)]
    recent = [t for t in tokens if t.startswith('RE') and WEATHER.fullmatch(t[2:])]
    result.update(present_weather_tokens=' '.join(weather), recent_weather_tokens=' '.join(recent),
        present_weather_status='reported' if weather else (
            'cavok_no_significant_weather' if cavok else (
                'missing_group' if '//' in tokens else 'no_group_reported')))
    layers = []
    cloud_tokens = []
    for token in main_tokens:
        if token in ['///CB','///TCU']:
            cloud_tokens.append(token)
            layers.append(dict(cover='///',height_ft=None,height_relation='missing',kind=token[3:],raw=token))
            continue
        cloud = CLOUD.fullmatch(token)
        if cloud:
            cloud_tokens.append(token)
            height = int(cloud[2])*100 if cloud[2].isdigit() else None
            relation = 'reported_step' if height is not None else 'missing'
            if height == 0:
                height, relation = 100, 'lt'
            layers.append(dict(cover=cloud[1], height_ft=height,
                height_relation=relation, kind=cloud[3] or '', raw=token))
    sky_codes = [t for t in main_tokens if t in ['NSC', 'NCD', 'SKC', 'CLR']]
    cloud_state = 'cavok' if cavok else ('layers' if layers else (
        sky_codes[0] if len(sky_codes) == 1 else 'not_reported'))
    if cavok and (layers or sky_codes or weather):
        issues.append('cavok_with_explicit_weather_or_cloud')
    known_layers = [x for x in layers if x['cover'] != 'VV' and x['height_ft'] is not None]
    lowest = min(known_layers, key=lambda x: x['height_ft']) if known_layers else None
    ceilings = [x for x in layers if x['cover'] in ['BKN', 'OVC']]
    unknown_amount = any(x['cover'] == '///' for x in layers)
    lowest_ceiling = min(ceilings, key=lambda x: x['height_ft']) if ceilings and all(
        x['height_ft'] is not None for x in ceilings) and not unknown_amount else None
    ceiling = lowest_ceiling['height_ft'] if lowest_ceiling else None
    vv = [x for x in layers if x['cover'] == 'VV']
    result.update(cloud_tokens=' '.join(cloud_tokens+sky_codes), cloud_state=cloud_state,
        cloud_layers_json=json.dumps(layers, ensure_ascii=False),
        lowest_known_cloud_base_ft=lowest['height_ft'] if lowest else None,
        lowest_known_cloud_base_relation=lowest['height_relation'] if lowest else 'missing',
        cloud_ceiling_ft=ceiling,
        cloud_ceiling_relation=lowest_ceiling['height_relation'] if lowest_ceiling else 'missing',
        cloud_ceiling_status=('reported' if ceiling is not None else (
            'amount_unknown' if unknown_amount else ('height_missing' if ceilings else 'no_bkn_ovc_group'))),
        vertical_visibility_ft=vv[0]['height_ft'] if len(vv) == 1 else None,
        vertical_visibility_relation=vv[0]['height_relation'] if len(vv) == 1 else 'missing',
        vertical_visibility_reported=bool(vv),
        cloud_type_unavailable=any(x['kind'] == '///' or x['raw'] == '//////' for x in layers),
        cloud_amount_unavailable=unknown_amount,
        cumulonimbus_reported=any(x['kind'] == 'CB' for x in layers),
        towering_cumulus_reported=any(x['kind'] == 'TCU' for x in layers))
    temperature_section = tokens[:pressure_index] if pressure_index is not None else tokens
    temp_tokens = [t for t in temperature_section if TEMPERATURE.fullmatch(t)]
    temp = temp_tokens[0] if len(temp_tokens) == 1 else None
    def temp_value(value):
        return None if value == '//' else float(value.replace('M', '-'))
    temperature_match = TEMPERATURE.fullmatch(temp) if temp else None
    result.update(temperature_token=temp,
        temperature_c=temp_value(temperature_match['temperature']) if temperature_match else None,
        dewpoint_c=temp_value(temperature_match['dewpoint']) if temperature_match else None)
    pressure_tokens = [t for t in tokens if PRESSURE.fullmatch(t)]
    pressure = pressure_tokens[0] if len(pressure_tokens) == 1 else None
    if len(pressure_tokens) != 1:
        issues.append('qnh_group_missing_or_multiple')
    qnh_raw = float(pressure[1:]) if pressure and pressure[1:].isdigit() else None
    pressure_unit = ('hPa' if pressure[0] == 'Q' else 'inHg') if pressure else None
    if pressure and pressure[0] == 'A' and qnh_raw is not None:
        qnh_raw /= 100
    qnh_hpa = qnh_raw*(33.8638866667 if pressure_unit == 'inHg' else 1) if qnh_raw is not None else None
    result.update(qnh_token=pressure, qnh_unit=pressure_unit, qnh_raw=qnh_raw, qnh_hpa=qnh_hpa)

    # Независимый от исходных токенов парсер должен согласоваться по обычным числам.
    if parsed:
        comparisons = [(parsed.wind_speed.value('KT') if parsed.wind_speed else None, result['wind_speed_kt'], 'wind_speed'),
                       (parsed.wind_gust.value('KT') if parsed.wind_gust else None, result['wind_gust_kt'], 'wind_gust'),
                       (parsed.temp.value('C') if parsed.temp else None, result['temperature_c'], 'temperature'),
                       (parsed.dewpt.value('C') if parsed.dewpt else None, result['dewpoint_c'], 'dewpoint'),
                       (parsed.press.value('HPA') if parsed.press else None, qnh_hpa, 'qnh')]
        # 0000, //// и 9999 проверяются по сырому коду, поскольку парсер теряет часть семантики.
        if vis_relation == 'eq':
            comparisons.append((parsed.vis.value('M') if parsed.vis else None, vis_value, 'visibility'))
        for left, right, name in comparisons:
            if (left is None) != (right is None) or (left is not None and
                    not math.isclose(left, right, rel_tol=1e-5, abs_tol=0.02)):
                issues.append('parser_value_disagreement_'+name)
        parsed_weather = [''.join(t or '' for t in group) for group in parsed.weather]
        # python-metar классифицирует голую ////// как погоду. Это код пропуска
        # облаков; он не должен превращаться в метеоявление.
        parsed_weather = [t for t in parsed_weather if t.strip('/')]
        if parsed_weather != weather:
            issues.append('parser_value_disagreement_weather')
        # Парсер не создает слой для полностью отсутствующих компонентов.
        # Сам токен и неизвестные характеристики остаются в результате v2.
        compared_layers = [x for x in layers if x['raw'].strip('/')]
        if len(parsed.sky) != len(compared_layers)+len(sky_codes):
            issues.append('parser_value_disagreement_cloud_layers')
        elif compared_layers:
            for layer, (cover, height, kind) in zip(compared_layers, parsed.sky):
                encoded_height = 0 if layer['height_relation'] == 'lt' else layer['height_ft']
                parser_height = height.value('FT') if height is not None else None
                expected_kind = '' if layer['kind'] == '///' else layer['kind']
                if (cover != layer['cover'] or (kind or '') != expected_kind
                        or parser_height != encoded_height):
                    issues.append('parser_value_disagreement_cloud_values')
                    break
    result.update(parser_status='ok' if not issues else ('error' if error else 'needs_review'),
        parser_error=error, decode_issues=';'.join(issues),
        decoded_candidate=not issues)
    return result
