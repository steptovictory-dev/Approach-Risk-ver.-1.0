"""Карта доступности полей и полноты C40 без фильтрации базы."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'europe/data/working/03_c40_diagnostic_flags.parquet'
REPORT = ROOT / 'europe/reports/analytical_sample'
SOURCE_URL = 'https://zenodo.org/records/10210178'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def field_rules():
    # Роли относятся к прогнозу до первой попытки, а не к любому возможному эксперименту
    rules = {}
    def assign(fields, role, status, reason, verification):
        for field in fields:
            rules[field] = dict(role=role, availability=status, reason=reason,
                                required_verification=verification)
    assign(['GoA'], 'target', 'не вход модели', 'Исходная целевая метка', 'Траекторная валидация метки')
    assign(['landing_time', 'attempt_times', 'ILS', 'n_attempts'], 'outcome', 'исключить из входов',
           'Поле завершенного наблюдения содержит информацию об исходе',
           'Можно использовать для аудита и определения метки, не для раннего прогноза')
    assign(['flight_id', 'callsign', 'icao24', 'registration', '__index_level_0__'],
           'identifier', 'не включать в базовый набор',
           'Идентификация, возможное запоминание бортов и рейсов',
           'Использовать для связей и проверки пересечения групп; flight_id не уникален')
    assign(['airport', 'country'], 'context_candidate', 'условно доступно',
           'В таблице указан аэропорт конечной посадки и его страна',
           'Подтвердить, что это целевой аэропорт в момент прогноза, а не результат диверсии')
    assign(['typecode', 'icaoaircrafttype', 'AC_CLASS', 'market_segment'],
           'metadata_candidate', 'условно доступно', 'Справочное обогащение без гарантии исторической доступности',
           'Источник, версия, дата действия и возможность получить поле до исхода')
    assign(['AP_C_RWY'], 'unresolved', 'не допускать до проверки',
           'Семантика и время присоединения ВПП не подтверждены',
           'Установить, выбранная ли это ВПП до захода или итоговая ВПП')
    assign(['C40_CROSS_TIME', 'C40_BEARING', 'C40_CROSS_LAT', 'C40_CROSS_LON'],
           'event_candidate', 'условно доступно в точке C40',
           'Измерение при входе в ASMA, но соответствие нужному полету требует проверки',
           'Порядок времен, полнота, единицы, сопоставление и доступность в реальном времени')
    assign(['registration_was_empty_string', 'typecode_was_empty_string', 'icaoaircrafttype_was_empty_string'],
           'provenance', 'не включать в базовый набор', 'Сохраненное представление пропуска',
           'Допуск индикаторов пропусков возможен отдельным экспериментом после проверки происхождения')
    assign(['aircraft_description_reference'], 'reference_candidate', 'не допускать до проверки',
           'Текущее справочное описание для части неоднозначных кодов',
           'Историческая применимость и единая семантика для всех наблюдений')
    assign(['aircraft_reference_status', 'aircraft_reference_source', 'aircraft_reference_checked_date'],
           'provenance', 'не включать в базовый набор', 'Следы выполненного справочного разбора',
           'Использовать для аудита, а не как характеристики полета')
    assign(['c40_time_missing'], 'diagnostic', 'не включать в базовый набор',
           'Пропуск в ретроспективном обогащении не равен отсутствию входа в ASMA',
           'Проверить механизм отсутствия и будущий способ получения данных')
    assign(['c40_after_first_attempt', 'c40_after_landing'], 'outcome_diagnostic', 'исключить из входов',
           'Флаг рассчитан относительно времен завершенного наблюдения', 'Использовать только для аудита')
    return rules


def main():
    previous = json.loads((ROOT / 'europe/reports/preparation/07_c40_flags_applied.json').read_text(encoding='utf8'))
    before = digest(INPUT)
    assert before == previous['output_sha256']
    reader = pq.ParquetFile(INPUT)
    rules = field_rules()
    assert set(rules) == set(reader.schema_arrow.names)
    nulls, empty, groups, positive = Counter(), Counter(), Counter(), Counter()
    rows = events = 0
    c40_fields = ['C40_CROSS_TIME', 'C40_BEARING', 'C40_CROSS_LAT', 'C40_CROSS_LON']
    for batch in reader.iter_batches(batch_size=100_000):
        table = pa.Table.from_batches([batch])
        for field in reader.schema_arrow:
            array = table[field.name]
            nulls[field.name] += array.null_count
            if pa.types.is_string(field.type):
                empty[field.name] += int(pc.sum(pc.cast(pc.fill_null(pc.equal(array, ''), False), pa.int64())).as_py())
        ga = table['GoA'].to_numpy(zero_copy_only=False)
        missing = table['c40_time_missing'].to_numpy(zero_copy_only=False)
        after = table['c40_after_first_attempt'].to_numpy(zero_copy_only=False)
        after_landing = table['c40_after_landing'].to_numpy(zero_copy_only=False)
        complete = np.ones(len(table), dtype=bool)
        for field in c40_fields:
            complete &= pc.is_valid(table[field]).to_numpy(zero_copy_only=False)
        finite = np.ones(len(table), dtype=bool)
        for field in c40_fields[1:]:
            finite &= np.isfinite(table[field].to_numpy(zero_copy_only=False))
        # Эта маска показывает последствия диагностического условия, не формирует выборку
        masks = {
            'all_rows': np.ones(len(table), dtype=bool),
            'c40_time_missing': missing,
            'c40_time_present': ~missing,
            'c40_time_present_not_after_first_attempt': ~missing & ~after,
            'c40_all_four_fields_present': complete,
            'c40_complete_finite_not_after_first_attempt': complete & finite & ~after,
            'c40_after_first_attempt': after,
            'c40_after_landing': after_landing,
        }
        for name, mask in masks.items():
            groups[name] += int(mask.sum())
            positive[name] += int(ga[mask].sum())
        rows += len(table)
        events += int(ga.sum())
    assert rows == previous['rows_after'] and events == previous['go_arounds_after']
    assert groups['c40_time_missing'] + groups['c40_time_present'] == rows
    assert groups['c40_time_present'] - groups['c40_time_present_not_after_first_attempt'] == groups['c40_after_first_attempt']
    assert digest(INPUT) == before
    mapping = pd.DataFrame([dict(field=field.name, physical_type=str(field.type),
                                null_rows=nulls[field.name], null_percent=100*nulls[field.name]/rows,
                                exact_empty_string_rows=empty[field.name], **rules[field.name])
                            for field in reader.schema_arrow])
    coverage = pd.DataFrame([dict(diagnostic_group=name, rows=groups[name], go_arounds=positive[name],
                                  go_around_percent=100*positive[name]/groups[name] if groups[name] else None,
                                  row_coverage_percent=100*groups[name]/rows,
                                  positive_coverage_percent=100*positive[name]/events)
                             for name in groups])
    REPORT.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(REPORT / '04_02_feature_availability.csv', index=False)
    coverage.to_csv(REPORT / '04_02_c40_coverage.csv', index=False)
    summary = dict(input=str(INPUT.relative_to(ROOT)), input_sha256=before, input_unchanged=True,
                   rows_checked=rows, go_arounds=events, physical_columns=len(mapping),
                   rows_changed=0, rows_deleted=0, prediction_context='before first observed attempt',
                   feature_set_finalized=False, c40_filter_applied=False,
                   timing_condition='C40 <= first attempt; equality retained, no lead time guaranteed',
                   role_counts=mapping.role.value_counts().to_dict(),
                   source_url=SOURCE_URL, source_checked_date='2026-10-08')
    (REPORT / '04_02_feature_audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return mapping, coverage


if __name__ == '__main__':
    main()
