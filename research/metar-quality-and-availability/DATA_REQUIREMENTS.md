# Входы и происхождение

Все пути относительно корня Approach Risk. Расчеты не требуют сетевой
загрузки; полный вход не хранится в Git.

| Путь | Требование |
|---|---|
| `europe/data/analytical/06_05_cases_excluded.parquet` | Фиксированный аналитический вход с SHA-256 из манифеста. |
| `europe/weather/data/full/events/*.parquet` | 93 исходных разбиения событий с source_row и C40. |
| `europe/weather/data/full_cloud_v2/reports/*_reports.parquet` | 93 версии наблюдаемых погодных сообщений cloud-v2. |
| `europe/weather/data/final_c40_weather_v1/*.parquet` | Основной принятый сценарий для независимого воспроизведения связей и значений. |
| `europe/weather/reports/17_numeric_quality_preview.csv` | Исторический реестр согласованных масок; старый статус proposed не отменяет их последующее применение. |
| `europe/weather/reports/23_final_weather_application.json` | Хеши, количество строк и меток. |

Производный аудит `data/audit_messages.parquet` создается run_study.py,
не является дополнительным источником и исключен из Git. Полные лексические
сравнения в reports/ воспроизводятся validate_and_bootstrap.py.

Нормативные источники, ограничения лицензий и происхождение исходного
Zenodo перечислены в literature/01_positioning_and_sources.md.
Наличие доступного архива не подтверждает фактическое время его получения
при историческом прогнозе. Данные не заменяются и не исправляются.
