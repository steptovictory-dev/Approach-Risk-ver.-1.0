"""Проверка сохраненных выводов и оформления без повторного расчета."""
from pathlib import Path
import ast
import base64
import io
import tokenize
import nbformat

STUDY = Path(__file__).resolve().parents[1]


def main():
    # Проверяем готовые результаты и сохраняем локальные изображения для просмотра
    previews = STUDY/'data/notebook_previews'
    previews.mkdir(parents=True, exist_ok=True)
    for path in sorted((STUDY/'notebooks').rglob('*.ipynb')):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        code = [cell for cell in notebook.cells if cell.cell_type == 'code']
        plots = 0
        for cell in code:
            ast.parse(cell.source)
            assert cell.execution_count is not None, path.name
            for token in tokenize.generate_tokens(io.StringIO(cell.source).readline):
                if token.type == tokenize.COMMENT:
                    assert not token.string.rstrip().endswith('.'), (path.name, token.string)
            for output in cell.outputs:
                assert output.output_type != 'error', path.name
                png = output.get('data', {}).get('image/png')
                if png:
                    plots += 1
                    (previews/f'{path.stem}_{plots}.png').write_bytes(base64.b64decode(png))
        assert plots > 0, path.name
        print(f'{path.name}: {len(code)} выполненных ячеек, {plots} графиков', flush=True)


if __name__ == '__main__':
    main()
