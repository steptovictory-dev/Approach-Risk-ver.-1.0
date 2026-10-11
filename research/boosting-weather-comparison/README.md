# Бустинговые классификаторы GoA на C40

Независимый этап от main f4454ae. Ветка research/boosting-weather-comparison.
Методика – обучение с учителем, бинарная классификация GoA градиентным
бустингом по деревьям. Первое семейство – CatBoostClassifier.
Статус: все 10 CatBoost обучены; оценка и независимые проверки завершены.
Исполненные ноутбуки: 01_catboost_training.ipynb, 02_catboost_diagnostics.ipynb.
Итог: reports/08_research_answer.md.
Пять заранее заданных конфигураций для каждого варианта сравниваются по
validation AP. Обе модели выбираются до расчета test; test уже известен из
предыдущего исследования, поэтому сравнение семейств остается исследовательским.

Используем версию europe_c40_model_v1 через approach_risk.data.registry.
Физические пути и контрольные суммы заданы в configs/datasets.json,
настройки моделей – configs/boosting_weather.json. Новый код не импортирует
сценарии соседнего исследования. Модели новых запусков –
artifacts/boosting-weather-comparison, вне Git.
Исходные данные и завершенные результаты не меняем и не дублируем.
Логистические модели остаются фиксированной исходной точкой сравнения.

Среда: CatBoost 1.2.10, CPU, 500 итераций, без подвыборки и early stopping.
Train упорядочен по C40, has_time сохраняет порядок при обработке категорий.
Ноутбук обучения исполняет fit, сохраняет параметры, прогнозы и кривые потерь.
Ноутбук notebooks/02_catboost_diagnostics.ipynb строит диагностику из сохраненных
прогнозов без обучения: precision–recall, калибровка, перенос порогов и важности.
Точные условия записаны в protocol.md до первого fit.
Архитектурная проверка scripts/verify_shared_architecture.py прошла по
93 файлам и воспроизвела сохраненные test-метрики старых моделей без fit.
Исполненный отчет – notebooks/00_architecture.ipynb.
Публичные артефакты: notebooks, scripts, reports, figures.
Локальные data (модели, прогнозы) и Checkpoint исключены из Git.

## Воспроизведение из корня проекта

Установить дополнительную зависимость CatBoost 1.2.10 из pyproject.toml.
Порядок исполнения:

```powershell
.venv/Scripts/python.exe research/boosting-weather-comparison/scripts/build_training_notebook.py
.venv/Scripts/python.exe research/boosting-weather-comparison/scripts/build_diagnostics_notebook.py
.venv/Scripts/python.exe research/boosting-weather-comparison/scripts/verify_bootstrap.py
.venv/Scripts/python.exe research/boosting-weather-comparison/scripts/verify_training.py
.venv/Scripts/python.exe research/boosting-weather-comparison/scripts/write_research_summary.py
```

Обучающий сценарий возобновляет завершенные кандидаты только при совпадении
fingerprint и хеша модели. Диагностический ноутбук явно вычисляет
PredictionValuesChange из сохраненных моделей и обновляет таблицы важностей;
его нужно исполнять после обучающего ноутбука. finish_training_notebook.py
сохранен как точечное восстановление двух отчетных ячеек первого запуска после
исправления обработки Series с датами; повторный fit при восстановлении не делался.
