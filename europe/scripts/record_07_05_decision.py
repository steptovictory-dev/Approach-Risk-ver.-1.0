"""Сохраняет решение пользователя по составу и пропускам 07_05."""
import json
import nbformat
from audit_airports import ROOT, INPUT
from audit_feature_availability import digest


def main():
    folder=ROOT/'europe/reports/airports/composition'
    audit=json.loads((folder/'07_05_audit.json').read_text(encoding='utf8'))
    before=digest(INPUT);assert before==audit['input_sha256']
    decision=dict(decision_date='2026-10-09',authorized_by='user',decision='retain_all_records_no_exclusion_for_missing_reference_fields',input_sha256=before,rows_deleted=0,values_changed=0,feature_set_finalized=False,model_comparison='compare variants with and without reference fields; handle missing values as appropriate for each model',modeling_status='planned_not_executed')
    text='''## Решение – 09.10.2026

Пользователь решил **сохранить все записи**: пропуски справочных полей и различия состава не являются основанием для новых исключений. Состав выборки и исходные значения не изменены. Предыдущие решения этапов 04–06 сохраняются.

При моделировании сравнить варианты с этими справочными полями и без них, с подходящей для каждой модели обработкой пропусков. Использование полей в окончательной модели еще не определено. Проверка должна использовать одинаковую тестовую выборку и временные границы; доступность полей к C40 проверяется до включения. Эксперименты еще не выполнены.

Решение по сохранению записей в 07_05 принято. Следующий шаг – 07_06, распределения числовых полей отдельно для GoA=True и GoA=False.
'''
    (folder/'07_05_user_decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    path=folder/'07_05_review.md';current=path.read_text(encoding='utf8');current=current.split('## Решение')[0]+text;path.write_text(current,encoding='utf8')
    path=ROOT/'europe/notebooks/07_airports/07_05_composition_missing.ipynb'
    nb=nbformat.read(path,as_version=4)
    assert '## Решение' in nb.cells[0].source
    nb.cells[0].source=nb.cells[0].source.split('## Решение')[0]+text
    nbformat.validate(nb);nbformat.write(nb,path)
    assert digest(INPUT)==before
    print('Решение 07_05 сохранено в ноутбуке, отчете и JSON. Выборка не изменена.')


if __name__=='__main__':main()
