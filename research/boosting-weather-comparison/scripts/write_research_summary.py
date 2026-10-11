"""Write a factual research summary and local continuity map after verification."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'research/boosting-weather-comparison'
R=BASE/'reports'
verification=json.loads((R/'06_independent_verification.json').read_text(encoding='utf8'))
assert verification['verified']
families=pd.read_csv(R/'03_family_comparison.csv')
selection=json.loads((R/'02_selection.json').read_text(encoding='utf8'))
intervals=pd.read_csv(R/'04_day_intervals.csv')
operating=pd.read_csv(R/'07_operating_points.csv')
table='| Семейство | Признаки | AP | ROC-AUC | Brier |\n|---|---|---:|---:|---:|\n'
for x in families.itertuples():
    table+=f'| {x.family} | {"С погодой" if x.variant=="weather" else "Без погоды"} | {x.AP:.6f} | {x.ROC_AUC:.4f} | {x.Brier:.6f} |\n'
ap=intervals[intervals.metric=='AP'].iloc[0]
ops=operating[operating.validation_FPR_budget==.01]
optext='\n'.join(f'- {x.variant}: полнота {x.test_recall:.2%}, фактический FPR {x.test_FPR:.2%}, точность {x.test_precision:.2%}, TP={x.TP}, FP={x.FP}.' for x in ops.itertuples())
claim='Интервал AP целиком выше нуля.' if ap.lower_95>0 else 'Интервал AP включает ноль; устойчивое преимущество погоды по этому интервалу не установлено.'
report=f'''# CatBoost на C40: результат исследовательского сравнения

Методика – обучение с учителем, бинарная классификация GoA градиентным
бустингом деревьев CatBoost 1.2.10, Logloss. Два набора признаков: без погоды
(8) и с погодой (34). Train 2019–2021, validation 2022, test 2023 по C40 UTC.
Train: 4 165 060 событий, 11 626 GoA; validation: 1 619 848 / 4 984;
test: 546 577 / 1 739. Доля GoA на test: {1739/546577:.4%}.

Каждый из десяти кандидатов обучен на train: 500 деревьев, CPU, без
подвыборки, class weights и early stopping. Train отсортирован по C40 UTC,
has_time=True. Глубины/скорости были заданы до первого fit. Выбор – максимум
validation AP отдельно для каждого набора; пороги также выбраны на validation.
Выбранные конфигурации: {selection['selected']}. Test не использовался для
выбора этих конфигураций. Все модели и прогнозы сохранены локально вне Git.

## Метрики на одинаковых событиях test

{table}
AP – average precision, выше лучше; ROC-AUC выше лучше; Brier ниже лучше.
Из-за редкости GoA ROC-AUC отдельно недостаточно. Это точечные оценки;
различия семейств здесь не сопровождаются отдельной проверкой значимости.

## Добавление погоды внутри выбранной пары CatBoost

Разность AP (с погодой минус без погоды): {ap.difference:.6f}; 95% дневной
парный bootstrap-интервал [{ap.lower_95:.6f}; {ap.upper_95:.6f}]. {claim}
2000 повторов по UTC-дням, seed=20261011. Интервалы условны на выбранных
моделях; не включают неопределенность train/выбора и не устраняют всю
междневную зависимость. Все три интервала: 04_day_intervals.csv.

## Порог с ограничением validation FPR ≤ 1%

{optext}

На test фактический FPR может отличаться от validation. Рост AP сам по себе
не доказывает пригодность для эксплуатации. Precision–recall, калибровка,
важности и срезы показаны в notebooks/02_catboost_diagnostics.ipynb.
Важности признаков не доказывают причинный эффект погоды.

## Воспроизводимость и ограничения

Входные хеши, временные границы, метки, объемы, одинаковые source_row,
сохраненные метрики и 1000 прогнозов каждой выбранной модели проверены.
Контрольные суммы {verification['hashed_files']} файлов сверены независимо.
Ноутбук обучения исполнен, ошибок ячеек нет. Ускоренный bootstrap отдельно
сверен с прямым пересэмплированием sklearn при наличии связок и без них.

Test уже просматривался в логистическом исследовании. Это исследовательское
сравнение, не независимое подтверждение. Проверки задержек METAR/COR для
новой пары, независимый период и независимая валидация разметки отсутствуют.
Не расширять подбор по этим test-результатам. Следующий научный вопрос –
переносимость выбранной пары и порогов при ограничениях доступности погоды;
его следует оформить отдельным протоколом до новых экспериментов.
'''
(R/'08_research_answer.md').write_text(report,encoding='utf8')
checkpoint=f'''# CP-03. CatBoost: обучение и исследовательская оценка завершены

11 октября 2026. Ветка research/boosting-weather-comparison.

{report}

## Карта текущего этапа

```mermaid
flowchart TD
    A["✓ Общая архитектура и версия europe_c40_model_v1"] --> B["✓ Методика: бинарная классификация GoA"]
    B --> C["✓ 10 CatBoost, выбор по validation"]
    C --> D["✓ Test, пороги, интервалы, диагностика"]
    D --> E["● МЫ ЗДЕСЬ: результат сохранен и проверен"]
    E --> F["→ Отдельный протокол переносимости и доступности METAR"]
    F --> G["→ Независимый период для подтверждения"]
    style E fill:#dbeafe,stroke:#2563eb,stroke-width:4px
```

## Карта всего проекта

```mermaid
flowchart TD
    P["Approach Risk"] --> EU["Европа: C40, траектории, METAR"]
    EU --> OLD["✓ Логистические модели и чувствительность погоды"]
    OLD --> NOW["● CatBoost: 10 моделей, оценка завершена"]
    NOW --> NEXT["→ Переносимость и независимое подтверждение"]
    P --> US["Америка"]
    US --> ARCH["Архив США/Канада 2019: сохранен"]
    US --> PILOT["Новый пилот США: отдельное направление"]
    P --> PUB["Статьи: локальные черновики и результаты"]
    NOW -.-> PUB
    P --> GH["GitHub: общий пакет и будущий c40_geometry"]
    GH --> IT["IT Город"]
    GH --> CV["Резюме и портфолио"]
    style NOW fill:#dbeafe,stroke:#2563eb,stroke-width:4px
```

Articles, Checkpoint, исходные данные, модели и прогнозы вне Git.
Старые исследования не переобучены и не перемещены. Main не изменен.
Следующие исследования не запускать без отдельной постановки. Не расширять
настройку по уже просмотренному test. Модели находятся в artifacts/
boosting-weather-comparison. Точные параметры и версии сохранены там и в reports.
'''
(BASE/'Checkpoint/CP-03_catboost.md').write_text(checkpoint,encoding='utf8')
readme=BASE/'README.md';text=readme.read_text(encoding='utf8')
text=text.replace('Статус: обучение CatBoost запущено в notebooks/01_catboost_training.ipynb.','Статус: все 10 CatBoost обучены; оценка и независимые проверки завершены.\nИсполненные ноутбуки: 01_catboost_training.ipynb, 02_catboost_diagnostics.ipynb.\nИтог: reports/08_research_answer.md.')
readme.write_text(text,encoding='utf8')
status=json.loads((R/'01_run_status.json').read_text(encoding='utf8'));status['status']='completed_verified'
(R/'01_run_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(report)
