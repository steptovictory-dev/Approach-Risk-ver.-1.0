# Качество и доступность исторических METAR на C40

Отдельное исследование для будущей статьи 04. Ветка:
`research/metar-quality-and-availability`, основа `main` на `91fb94d`.

[Научный вопрос и протокол](protocol.md). Исходная погодная подготовка
находится в [europe/weather](../../europe/weather/README.md) и сохраняется.
Исследование sample-selection остается самостоятельным завершенным этапом;
связь с ним - явные популяции и знаменатели оценки, не объединение результатов.

`notebooks/` содержит выполненные расчеты, формулы, таблицы и графики;
`reports/` - результаты и журналы; `figures/` - экспорты;
`scripts/` - воспроизведение; `literature/` - источники и позиционирование.
`data/` и `Checkpoint/` исключены из Git. Материалы будущей рукописи
хранятся локально в `Articles/article_04/`.

Обучение моделей и изменение исходных данных в этот этап не входят.

## Результаты

Начать с [ответа на вопрос](reports/06_research_answer.md), затем:

1. [Вопрос, формулы и методика](notebooks/01_question_and_methods.ipynb).
2. [Чувствительность доступности и признаков](notebooks/02_availability_sensitivity.ipynb).
3. [Проверка исходных токенов и разбор расхождений](notebooks/03_decode_validation.ipynb).

Семь сценариев, все 6 331 485 строк; 3 выполненных ноутбука и 7 графиков.
При сравнении задержек 0 и 15 минут покрытие меняется на 0,0855 п.п.,
но значения или пропуски хотя бы одного из 13 проверяемых полей меняются
для 48,88% событий. Это результат о чувствительности информации,
не измерение выигрыша модели или физической опасности.

[Источники и позиционирование](literature/01_positioning_and_sources.md),
[карта доказательств статьи](reports/07_article_evidence_map.csv).
Внешняя физическая верификация и фактическое время получения неизвестны.

## Воспроизведение

Из корня проекта, в окружении проекта с pandas, numpy, pyarrow, metar,
nbformat, nbclient, ipykernel и matplotlib:

```powershell
.\.venv\Scripts\python.exe research/metar-quality-and-availability/scripts/run_study.py
.\.venv\Scripts\python.exe research/metar-quality-and-availability/scripts/validate_and_bootstrap.py
.\.venv\Scripts\python.exe research/metar-quality-and-availability/scripts/write_evidence.py
.\.venv\Scripts\python.exe research/metar-quality-and-availability/scripts/build_notebooks.py
.\.venv\Scripts\python.exe research/metar-quality-and-availability/scripts/write_evidence.py
.\.venv\Scripts\python.exe research/metar-quality-and-availability/scripts/verify_artifacts.py
```

Повторный write_evidence обновляет экспорт графиков в локальном пакете статьи;
последний шаг проверяет согласованность результатов.
Точные версии расчетного окружения записаны в `02_study_manifest.json`.
Необходимы локальные архивы предыдущего погодного этапа; повторные загрузки
при расчете не выполняются. Публикация рукописи и депонирование данных
в рамках этого этапа не выполнялись.
