# Погодные ноутбуки

Выполненные ноутбуки содержат сохраненные выводы. Код, графики и таблицы доступны при просмотре без повторного запуска.

1. [01. Покрытие METAR](01_metar_coverage.ipynb): происхождение, выбор аэропортов/периодов, код повторного сопоставления, обе метки, пороги 30/60/90 минут, давность, COR и переход суток. 3 рисунка.
2. [02. Декодирование](02_metar_decoding.ipynb): разделение наблюдения/тренда, повторный разбор сообщений, заполненность и распределения, облачные коды, конкретные примеры, все 10 сообщений с замечаниями, диагностические сценарии и семантические тесты. 4 рисунка.

3. [03. Качество и доступность](03_quality_and_availability.ipynb): проверка канонических основных полей, все 10 остаточных сообщений, повторный выбор METAR при предполагаемой задержке 0/5/10/15 минут, COR и пороги давности. 3 рисунка. Задержки не подтверждают историческое время публикации.
4. [04. Протокол расширения пилота](04_expansion_protocol.ipynb): ранжирование ранее не исследованных станций без частоты GoA, проверка каталога IEM, группы по времени C40 UTC и ограничения дизайна. 3 рисунка. Сохранен дизайн до загрузки; результаты приведены в ноутбуке 05.
5. [05. Покрытие и декодирование расширенного пилота](05_expansion_coverage_and_decoding.ipynb): шесть выгрузок, повторная верификация сопоставления и всех декодированных полей, два сообщения с замечаниями, заполненность и сценарии качества. 3 рисунка.
6. [06. Качество и чувствительность покрытия шести аэропортов](06_combined_quality_and_availability.ipynb): проверка префикса до QNH, реестр 14 записей LEMD, повторный выбор при задержках 0/5/10/15 минут, сравнение шести станций и проверки сохранения данных. 4 рисунка.
7. [07. Протокол полного покрытия](07_full_coverage_protocol.ipynb): полный охват актуального входа, проверка 19 станций, месячные границы, план 93 запросов и 20 семантических тестов. 3 рисунка.

Для обозначения групп используются красный и зеленый цвета; значения обозначений приведены в легендах рисунков. Рисунки дополнительно сохранены в `../figures/` в PNG и SVG. Основные поля описаны в [словаре](../reports/04_field_dictionary.md).

Для повторного выполнения локальные игнорируемые файлы `../data/` должны быть на месте. Из корня проекта:

```powershell
.\.venv\Scripts\python.exe -m pip install -r europe/weather/requirements.txt
.\.venv\Scripts\python.exe europe/weather/scripts/build_weather_notebooks.py
.\.venv\Scripts\python.exe europe/weather/scripts/build_quality_notebook.py
.\.venv\Scripts\python.exe europe/weather/scripts/build_expansion_protocol.py
.\.venv\Scripts\python.exe europe/weather/scripts/run_expansion_pilot.py
.\.venv\Scripts\python.exe europe/weather/scripts/build_combined_quality_notebook.py
.\.venv\Scripts\python.exe europe/weather/scripts/full_weather_pipeline.py prepare
.\.venv\Scripts\python.exe europe/weather/scripts/build_full_weather_notebooks.py protocol
.\.venv\Scripts\python.exe europe/weather/scripts/full_weather_pipeline.py run
.\.venv\Scripts\python.exe europe/weather/scripts/build_full_weather_notebooks.py results
```

Скрипт сборки не скачивает данные и не обучает модели. С нуля сначала выполнить `audit_pilot_input.py`, `run_coverage_pilot.py`, `decode_pilot.py`. Сборка заново создает и выполняет оба ноутбука; редакторские изменения следует вносить в скрипт сборки.

8. [08. Полное погодное покрытие](08_full_weather_coverage.ipynb): 6 331 485 записей, целостность и сценарии доступности. 3 рисунка.
9. [09. Аудит числовых аномалий](09_numeric_anomaly_audit.ipynb): 18 кандидатов, соседние сообщения и почасовой архив. 1 рисунок.

Дополнительные команды воспроизведения из корня: verify_full_weather_results.py, audit_full_extreme_values.py, audit_numeric_anomalies.py, build_numeric_audit_notebook.py из europe/weather/scripts; запуск через .venv/Scripts/python.exe. Последний скрипт выполняет ноутбук, аудит загружает ограниченный архив при отсутствии кеша.

10. [10. Применение числового качества](10_numeric_quality_application.ipynb): принятые маски, журнал 84 ячеек в 44 строках, проверка сохранности всей базы. 1 рисунок.

Воспроизведение: запуск scripts/apply_numeric_quality.py, затем scripts/build_numeric_mask_notebook.py через .venv/Scripts/python.exe из корня проекта (пути скриптов начинаются с europe/weather/). Принятый протокол не перезаписывать сборщиком проекта протокола.

11. [11. Аудит остаточных форматов](11_residual_format_audit.ipynb): 2425 сообщений, основные категории, исходные примеры, дефект коротких CB/TCU и сохранность CP-10. 2 рисунка.

Воспроизведение: europe/weather/scripts/audit_residual_formats.py, затем europe/weather/scripts/build_residual_format_notebook.py через .venv/Scripts/python.exe из корня проекта.
