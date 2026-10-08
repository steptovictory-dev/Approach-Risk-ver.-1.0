"""Исключение одной согласованной неоднозначной записи LSGG из эксперимента."""
import json
from datetime import datetime, timezone

import nbformat
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from audit_feature_availability import ROOT, digest

INPUT = ROOT / "europe/data/analytical/05_aircraft_types_reviewed.parquet"
OUTPUT = ROOT / "europe/data/analytical/06_02_lsgg_case_excluded.parquet"
EXCLUDED = ROOT / "europe/data/review/06_02_lsgg_excluded_record.parquet"
REPORT = ROOT / "europe/reports/attempts/LSGG"
KEY = dict(flight_id="TAP948_37820", icao24="495295", airport="LSGG", landing_time="2019-03-01T16:29:15+00:00")
REASON = "Исключение по решению пользователя: неразрешенная неоднозначность разметки, дальнейшая проверка этого случая прекращена. Ошибочность GoA не доказана."


def mask(table):
    result = pc.equal(table["flight_id"], KEY["flight_id"])
    for name in ("icao24", "airport"):
        result = pc.and_(result, pc.equal(table[name], KEY[name]))
    instant = pa.scalar(pd.Timestamp(KEY["landing_time"]).to_pydatetime(), type=table.schema.field("landing_time").type)
    return pc.fill_null(pc.and_(result, pc.equal(table["landing_time"], instant)), False)


def check_equal(left, right):
    assert left.schema.equals(right.schema, check_metadata=True)
    if not left.equals(right):
        # pandas считает NaN равными; все числа проверяются без допуска
        pd.testing.assert_frame_equal(left.to_pandas(ignore_metadata=True), right.to_pandas(ignore_metadata=True), check_exact=True)


def main():
    assert not OUTPUT.exists() and not EXCLUDED.exists(), "Выходные файлы уже существуют; автоматическая перезапись запрещена"
    prior = json.loads((ROOT / "europe/reports/aircraft_types/05_05_applied.json").read_text(encoding="utf8"))
    before = digest(INPUT)
    assert before == prior["output_sha256"]
    source = pq.ParquetFile(INPUT)
    removed = []
    rows = positives = airport_rows = airport_positives = 0
    # Сначала проверяем уникальность полного ключа и исходную структуру случая
    for batch in source.iter_batches(batch_size=100_000, columns=list(KEY) + ["GoA", "n_attempts", "ILS", "attempt_times"]):
        table = pa.Table.from_batches([batch])
        selected = table.filter(mask(table))
        removed.extend(selected.to_pylist())
    assert len(removed) == 1
    assert removed[0]["GoA"] is True and removed[0]["n_attempts"] == 2 and removed[0]["ILS"] == ["22R", "22"]
    assert (removed[0]["attempt_times"][1] - removed[0]["attempt_times"][0]).total_seconds() == 12
    temp = OUTPUT.with_suffix(".pending.parquet")
    assert not temp.exists()
    EXCLUDED.parent.mkdir(parents=True, exist_ok=True)
    removed_full = None
    with pq.ParquetWriter(temp, source.schema_arrow, compression="zstd") as writer:
        for batch in source.iter_batches(batch_size=100_000):
            table = pa.Table.from_batches([batch])
            selected = mask(table)
            if pc.any(selected).as_py():
                removed_full = table.filter(selected)
            retained = table.filter(pc.invert(selected))
            writer.write_table(retained, row_group_size=100_000)
            rows += len(retained)
            positives += pc.sum(pc.cast(retained["GoA"], pa.int64())).as_py()
            local = retained.filter(pc.equal(retained["airport"], "LSGG"))
            airport_rows += len(local)
            airport_positives += pc.sum(pc.cast(local["GoA"], pa.int64())).as_py() or 0
    assert rows == prior["rows_after"] - 1 and positives == prior["go_arounds_after"] - 1
    assert airport_rows == 205373 and airport_positives == 529
    saved = pq.ParquetFile(temp)
    groups = 0
    for index, batch in enumerate(source.iter_batches(batch_size=100_000)):
        table = pa.Table.from_batches([batch])
        retained = table.filter(pc.invert(mask(table)))
        check_equal(retained, saved.read_row_group(index))
        groups += 1
    assert groups == saved.num_row_groups and saved.metadata.num_rows == rows
    assert removed_full is not None and len(removed_full) == 1
    pq.write_table(removed_full, EXCLUDED, compression="zstd")
    check_equal(removed_full, pq.read_table(EXCLUDED))
    assert digest(INPUT) == before
    saved.close()
    temp.replace(OUTPUT)
    journal = dict(applied_utc=datetime.now(timezone.utc).isoformat(), input=str(INPUT.relative_to(ROOT)), input_sha256=before, input_unchanged=True, output=str(OUTPUT.relative_to(ROOT)), output_sha256=digest(OUTPUT), excluded=str(EXCLUDED.relative_to(ROOT)), excluded_sha256=digest(EXCLUDED), exclusion_key=KEY, reason=REASON, rows_before=prior["rows_after"], rows_after=rows, go_arounds_before=prior["go_arounds_after"], go_arounds_after=positives, lsgg_rows_after=airport_rows, lsgg_go_arounds_after=airport_positives, rows_excluded=1, retained_values_changed=0, all_retained_fields_and_order_verified=True, physical_columns=len(source.schema_arrow), label_corrected=False)
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "06_02_exclusion_applied.json").write_text(json.dumps(journal, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
    note = f"""## Решение об исключении – 09.10.2026

По прямому решению пользователя из дальнейшего эксперимента исключена одна запись: TAP948_37820, ICAO24 495295, LSGG, посадка 01.03.2019 16:29:15 UTC. Причина – неразрешенная неоднозначность разметки и прекращение дальнейшей проверки случая; это не подтверждение ошибочности GoA=True.

Актуальная выборка: `europe/data/analytical/06_02_lsgg_case_excluded.parquet` – {rows:,} строк, {positives:,} GoA=True. Для LSGG осталось {airport_rows:,} строк и {airport_positives} GoA=True. Полная исключенная строка сохранена в `europe/data/review/06_02_lsgg_excluded_record.parquet`. Вход этапа 05 сохранен; порядок, схема и значения всех остальных строк проверены после записи.

Исторические отчеты 06_01–06_03 описывают вход этапа 05 до исключения. Журнал решения: `europe/reports/attempts/LSGG/06_02_exclusion_applied.json`. Воспроизведение из корня: `.\\.venv\\Scripts\\python.exe europe/scripts/exclude_lsgg_ambiguous_case.py`.
"""
    (REPORT / "06_02_exclusion.md").write_text(note, encoding="utf8")
    path = ROOT / "europe/notebooks/06_attempts/06_02/LSGG/06_02_lsgg_runway_case.ipynb"
    nb = nbformat.read(path, as_version=4)
    nb.cells.append(nbformat.v4.new_markdown_cell(note))
    nbformat.validate(nb)
    nbformat.write(nb, path)
    for path in [ROOT / "europe/README.md", ROOT / "europe/reports/attempts/06_02_review.md"]:
        text = path.read_text(encoding="utf8")
        path.write_text(text + "\n" + note, encoding="utf8")
    print(json.dumps(journal, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
