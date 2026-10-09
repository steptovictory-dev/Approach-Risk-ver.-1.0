"""Сохранение старых обзорных ноутбуков с защитой оформленного исследования."""
from pathlib import Path
import nbformat


def write_overview(notebook, path):
    # Подготовительные скрипты обновляют отчеты, но не заменяют рабочие ноутбуки
    path = Path(path)
    if path.exists() and nbformat.read(path, as_version=4).metadata.get('curated_research'):
        return
    nbformat.write(notebook, path)
