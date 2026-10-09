"""Фиксирует решение пользователя: сохранить все записи погодного разбора."""
import json
import nbformat
from audit_feature_availability import ROOT, digest
from audit_airports import INPUT


def main():
    report=ROOT/'europe/reports/airports/february_2020_weather'
    before=digest(INPUT)
    manifest=json.loads((report/'07_03_weather_comparison_manifest.json').read_text(encoding='utf8'))
    assert before==manifest['input_sha256']
    decision=dict(decision_date='2026-10-09',authorized_by='user',scope='07_04 February 2020 weather comparison across airports',
        decision='retain_all_records_no_new_exclusions',input_sha256=before,rows_deleted=0,values_changed=0,
        accepted_interpretation='gusty wind explains the observed spike',interpretation_author='user',
        individual_event_causality_independently_verified=False,
        modeling_plan='evaluate weather association using both labels and only information available at C40; model learns GoA probability, not causal attribution',
        modeling_status='not_executed')
    text='''## Решение – 09.10.2026

**Пользователь принял решение сохранить все записи.** Новых исключений по результатам погодного разбора нет; значения, метки и состав актуальной выборки не изменяются. Предыдущие решения этапов 04–06 сохраняются: это решение не восстанавливает ранее исключенные строки.

Принятая пользователем интерпретация всплеска – **порывистый ветер**. Архивные METAR и совпадение пиков дают основание для этой интерпретации; причинная связь для каждой отдельной записи независимо не установлена.

При моделировании проверяем связь погодных условий с вероятностью GoA на обеих метках. Для прогноза на C40 используем только информацию, доступную к C40; METAR у более поздней попытки ухода служит аудиту и не переносится в модель как известный заранее признак. Модель оценивает вероятность GoA, а не устанавливает причину решения экипажа. Эксперименты еще не выполнены.

**Статус: решение по сохранению записей этого разбора принято.** Журнал: `07_04_user_decision.json`.
'''
    (report/'07_04_user_decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (report/'07_04_decision.md').write_text('# 07_04 – Решение по погодному разбору\n\n'+text,encoding='utf8')
    path=ROOT/'europe/notebooks/07_airports/07_04_weather_comparison.ipynb'
    nb=nbformat.read(path,as_version=4)
    cells=[c for c in nb.cells if c.cell_type=='markdown' and c.source.startswith('## Решение')]
    assert len(cells)==1
    cells[0].source=text
    for cell in nb.cells:
        if cell.cell_type=='markdown' and cell.source.startswith('## Анализ перед решением'):
            cell.source=cell.source.replace('## Анализ перед решением','## Анализ и основания решения').replace('Обсуждаем: достаточен ли погодный контекст для завершения описательного разбора, какие конкретные записи требуют проверки и нужен ли отдельный анализ условий у обоих классов.', 'Решение по сохранению записей принято ниже. Отдельная оценка погодной связи на обоих классах относится к последующему моделированию.')
    nbformat.validate(nb);nbformat.write(nb,path)
    readme=ROOT/'europe/notebooks/07_airports/README.md'
    note='\n\nРешение 07_04 от 09.10.2026: все записи погодного разбора сохранены, новых исключений нет. Принятая интерпретация всплеска – порывистый ветер; погодные признаки для модели проверяются на обеих метках с учетом доступности к C40.\n'
    current=readme.read_text(encoding='utf8')
    if note.strip() not in current:readme.write_text(current+note,encoding='utf8')
    assert digest(INPUT)==before
    print('Решение записано в 07_04, JSON-журнал и навигацию. SHA256 аналитической выборки не изменился.')


if __name__=='__main__':main()
