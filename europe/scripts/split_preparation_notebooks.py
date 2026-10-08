"""Разделение подготовки по темам с сохранением результатов исходных ячеек."""
from copy import deepcopy
from pathlib import Path
import nbformat

TOPICS = {
    '03_01': ('03_01_missing_values.ipynb', 0, 5, 'Диагностика пропусков'),
    '03_02': ('03_02_normalize_empty_strings.ipynb', 5, 7, 'Нормализация пустых строк'),
    '03_03': ('03_03_aircraft_codes.ipynb', 7, 11, 'Диагностика классификаций'),
    '03_04': ('03_04_reference_review.ipynb', 11, 14, 'Справочные предложения'),
    '03_05': ('03_05_apply_aircraft_reference.ipynb', 14, 17, 'Справочное обогащение'),
    '03_06': ('03_06_unresolved_aircraft.ipynb', 17, 21, 'Неразрешенные классификации'),
    '03_07': ('03_07_c40_diagnostics.ipynb', 21, 26, 'Диагностика C40'),
    '03_08': ('03_08_apply_c40_flags.ipynb', 26, 29, 'Добавление флагов C40'),
    '03_09': ('03_09_flight_ids.ipynb', 29, 34, 'Повторяющиеся идентификаторы'),
    '03_10': ('03_10_attempts.ipynb', 34, 39, 'Несколько попыток захода'),
}


def export(notebook, region):
    destination = region / 'notebooks/03_data_preparation'
    destination.mkdir(exist_ok=True)
    assert len(notebook.cells) == 39, 'При добавлении шагов обновите границы тем'
    for filename, start, end, title in TOPICS.values():
        cells = deepcopy(notebook.cells[start:end])
        if filename.startswith('03_02'):
            # Восстанавливаем подготовку входа для самостоятельного запуска изменения
            cells[1:1] = [nbformat.v4.new_markdown_cell(
                '### Подготовка входа\n\nПовторяем проверку источника и подсчет пропусков '
                'из 03_01, чтобы выполнять изменение в отдельном ядре.'),
                deepcopy(notebook.cells[1]), deepcopy(notebook.cells[3])]
        for cell in cells:
            if cell.cell_type == 'markdown':
                cell.source = cell.source.replace('../reports/', '../../reports/')
            elif 'region = Path.cwd().parent if' in cell.source:
                cell.source = cell.source.replace(
                    "region = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd() / 'europe'",
                    "# Находим папку Европы из любой вложенной папки ноутбуков\n"
                    "region = next(p / 'europe' for p in [Path.cwd(), *Path.cwd().parents]\n"
                    "              if (p / 'europe/data/source/osn23_landings_merged.parquet.gzip').exists())")
        intro = nbformat.v4.new_markdown_cell(
            f'# {title}\n\nТематический этап подготовки Европы 2019–2023. '
            'Выполняйте ячейки сверху вниз в окружении проекта. '
            'Каждый ноутбук соответствует одному шагу; '
            'диагностика и применение согласованного изменения разделены. '
            'Входы, выходы и порядок описаны в [карте подготовки](README.md). '
            'Сохраненные результаты перенесены из выполненного общего ноутбука; '
            'сам перенос данных не преобразует.')
        result = nbformat.v4.new_notebook(cells=[intro, *cells], metadata=deepcopy(notebook.metadata))
        nbformat.validate(result)
        for cell in cells:
            if cell.cell_type == 'code':
                compile(cell.source, filename, 'exec')
        nbformat.write(result, destination / filename)
    print('Тематических ноутбуков:', len(TOPICS))


if __name__ == '__main__':
    region = Path(__file__).resolve().parents[1]
    export(nbformat.read(region / 'notebooks/03_data_preparation/archive/03_data_preparation.ipynb', as_version=4), region)
