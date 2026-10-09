"""Проверка кода рабочих ноутбуков и ссылок на локальные результаты."""
from pathlib import Path
import ast
import nbformat

STUDY = Path(__file__).resolve().parents[1]


def main():
    for path in sorted((STUDY/'notebooks').rglob('*.ipynb')):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        namespace = {}
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                ast.parse(cell.source)
                exec(compile(cell.source,str(path),'exec'),namespace)
        print(f'Проверено: {path.name}',flush=True)


if __name__ == '__main__':
    main()
