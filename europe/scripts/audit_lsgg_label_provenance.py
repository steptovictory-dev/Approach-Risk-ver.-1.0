"""Проверка происхождения меток LSGG без изменения исходной таблицы."""
import ast
import hashlib
import json
from operator import attrgetter
from pathlib import Path
from types import SimpleNamespace

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "docs/sources/files/lsgg_label_provenance"
REPORT = ROOT / "europe/reports/attempts/LSGG"


class Segment:
    # Минимальный адаптер для выполнения сохраненной функции на искусственных данных
    def __init__(self, data):
        self.data = data

    @property
    def start(self):
        return self.data.timestamp.min()

    @property
    def stop(self):
        return self.data.timestamp.max()

    @property
    def altitude_min(self):
        return self.data.altitude.min()

    def bearing(self, threshold):
        return Segment(self.data.assign(bearing=threshold.bearing))

    def distance(self, threshold):
        return Segment(self.data.assign(distance=5.0))

    def assign(self, **kwargs):
        return Segment(self.data.assign(**kwargs))

    def query(self, expression):
        data = self.data.query(expression, engine="python")
        return None if data.empty else Segment(data)

    def split(self, interval):
        for _, data in self.data.groupby(self.data.timestamp.diff().gt(pd.Timedelta(interval)).cumsum()):
            yield Segment(data)

    def longer_than(self, duration):
        return self.stop - self.start > pd.Timedelta(duration.replace("T", "min"))

    def after(self, timestamp):
        data = self.data[self.data.timestamp.gt(timestamp)]
        return None if data.empty else Segment(data)

    def phases(self):
        return self.assign(phase="DESCENT")

    def aligned_on_ils(self, airport, **kwargs):
        return iter(self.detector(self, airport, **kwargs))


class Attempts:
    # go_around использует и len(), и next() для итератора сегментов
    def __init__(self, items):
        self.items = list(items)
        self.iterator = iter(self.items)

    def __len__(self):
        return len(self.items)

    def __next__(self):
        return next(self.iterator)


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    tree = ast.parse((SOURCES / "navigation.py").read_text(encoding="utf8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NavigationFeatures")
    methods = []
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in {"aligned_on_ils", "go_around"}:
            node.decorator_list = []
            node.body = [statement for statement in node.body if not isinstance(statement, ast.ImportFrom)]
            methods.append(node)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *methods], type_ignores=[])
    # Формулы и ветвления функции сохранены; импорты заменены минимальным окружением
    airport = SimpleNamespace(icao="SYNTH", runways=SimpleNamespace(shape=SimpleNamespace(is_empty=False), list=[SimpleNamespace(name="A", bearing=220.0), SimpleNamespace(name="B", bearing=220.0)]))
    scope = dict(np=np, pd=pd, attrgetter=attrgetter, cast=lambda kind, value: value, airports={"SYNTH": airport})
    exec(compile(ast.fix_missing_locations(module), "traffic_2_9_navigation", "exec"), scope)
    data = pd.DataFrame(dict(timestamp=pd.date_range("2000-01-01", periods=121, freq="s", tz="UTC"), altitude=np.linspace(3000, 1000, 121), track=220.0))
    flight = Segment(data)
    aligned = list(scope["aligned_on_ils"](flight, "SYNTH"))
    assert len(aligned) == 2 and aligned[0].start == aligned[1].start and aligned[0].stop == aligned[1].stop
    Segment.aligned_on_ils = lambda self, airport, **kwargs: Attempts(scope["aligned_on_ils"](self, airport, **kwargs))
    goarounds = list(scope["go_around"](flight, "SYNTH"))
    assert not goarounds
    notebook = json.loads((SOURCES / "OSN_paper_code.ipynb").read_text(encoding="utf8"))
    selected = []
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        if index in {3, 21, 29, 43}:
            selected.append(dict(cell_index_zero_based=index, source=source))
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in SOURCES.iterdir() if path.is_file()}
    result = dict(checked_date="2026-10-09", author_commit="f5aaaefc6ebeb2cff8048d9fa7b1f1da2d73851d", analysis_environment_traffic_version="2.9.0", dataset_generation_traffic_version_verified=False, generation_code_found_in_current_author_tree=False, original_runways_cache_available=False, original_trajectory_available=False, source_sha256=hashes, synthetic_check=dict(threshold_names=["A", "B"], identical_mock_geometry=True, single_descending_segment=True, aligned_segments=len(aligned), overlap_seconds=(aligned[0].stop-aligned[0].start).total_seconds(), go_around_segments=len(goarounds), limitation="Искусственные одинаковые оси; координаты LSGG и реальная траектория не воспроизведены"), case_decision=dict(flight_id="TAP948_37820", status="unresolved", change_ILS=False, change_n_attempts=False, change_GoA=False), author_cells=selected)
    (REPORT / "06_02_label_provenance.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
    note = (REPORT / "06_02_label_provenance.md").read_text(encoding="utf8")
    target = ROOT / "europe/notebooks/06_attempts/06_02/LSGG/06_02_lsgg_runway_case.ipynb"
    nb = nbformat.read(target, as_version=4)
    marker = "## Проверка кода разметки – 09.10.2026"
    nb.cells = [cell for cell in nb.cells if not (cell.cell_type == "markdown" and cell.source.startswith(marker))]
    nb.cells.append(nbformat.v4.new_markdown_cell(marker + "\n\n" + note.split("\n", 1)[1]))
    nbformat.validate(nb)
    nbformat.write(nb, target)
    print(json.dumps(result["synthetic_check"], ensure_ascii=False))
    print("Notebook valid; source data not opened or modified")


if __name__ == "__main__":
    main()
