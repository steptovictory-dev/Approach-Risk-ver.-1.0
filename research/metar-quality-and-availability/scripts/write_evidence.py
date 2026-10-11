"""Проверяемый ответ на вопрос и пакет будущей статьи 04."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parents[1]
PROJECT=BASE.parents[1]
R=BASE/'reports'
article=PROJECT/'Articles/article_04'
summary=pd.read_csv(R/'02_scenario_summary.csv')
overall=summary.groupby('scenario')[['rows','usable','message_changed','information_changed']].sum()
for field in ['usable','message_changed','information_changed']:overall[field+'_percent']=100*overall[field]/overall.rows
assert overall.rows.eq(6331485).all()
assert summary.loc[summary.GoA.eq(True),'rows'].eq(18349).all()
assert overall.loc['primary','usable']==6322586
assert overall.loc['primary','information_changed']==0
fields=pd.read_csv(R/'03_field_differences.csv')
field_totals=fields.groupby(['scenario','field'])[['rows','paired_present','missing_changed','value_changed','threshold_changed','absolute_difference_sum']].sum().reset_index()
field_totals['value_changed_percent_of_pairs']=100*field_totals.value_changed/field_totals.paired_present.replace(0,np.nan)
field_totals['mean_absolute_difference']=field_totals.absolute_difference_sum/field_totals.paired_present.replace(0,np.nan)
field_totals.loc[field_totals.field.isin(['present_weather_tokens','cumulonimbus_reported','towering_cumulus_reported','cavok']),'mean_absolute_difference']=np.nan
field_totals.to_csv(R/'06_field_summary.csv',index=False)
overall.reset_index().to_csv(R/'06_overall_summary.csv',index=False)

validation=json.loads((R/'05_validation_manifest.json').read_text(encoding='utf8'))
sample=pd.read_parquet(BASE/'data/audit_messages.parquet')
audit=pd.read_csv(R/'05_lexical_validation.csv')
disagreement=sample.loc[sample.metar_id.isin(audit.loc[~audit.agrees,'metar_id'])].copy()
assert len(disagreement)==4
assert (~disagreement.decoded_candidate & ~disagreement.core_validated).all()
disagreement=disagreement[['metar_id','station','metar_utc','metar','decode_issues','decoded_candidate','core_validated','visibility_token']]
disagreement['adjudication']='readable_9999_after_unsupported_structure; conservative_core_block_retained; no_raw_repair'
disagreement.to_csv(R/'05_disagreement_adjudication.csv',index=False)
validation['disagreement_messages']=len(disagreement)
validation['all_disagreement_cores_blocked']=True
(R/'05_validation_manifest.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

primary=overall.loc['primary']
zero=overall.loc['delay0']
nocor=overall.loc['no_cor']
goa=summary.loc[summary.GoA.eq(True)].set_index('scenario')
temp=field_totals.loc[field_totals.scenario.eq('delay0')&field_totals.field.eq('temperature_c')].iloc[0]
bootstrap=pd.read_csv(R/'04_block_bootstrap.csv')
interval=bootstrap.loc[bootstrap.scenario.eq('no_cor')&bootstrap.GoA.eq(True)&bootstrap.block_days.eq(14)].iloc[0]
text=f'''# 06. Ответ на научный вопрос

В фиксированной европейской выборке высокий процент погодного покрытия
не гарантирует устойчивость выбранной погодной информации. Правила условной
доступности и обработки COR меняют ее заметно, даже при небольшом изменении
числа покрытых событий. Это вывод о подготовке данных, не о качестве ML.

## Измеренный результат

В основной конфигурации (60 минут / задержка 15 минут / COR включены)
допустимый погодный блок имеют **{int(primary.usable):,}** из **6 331 485**
событий: **{primary.usable_percent:.4f}%**. Среди GoA:
**{int(goa.loc['primary','usable']):,} / 18 349**, **{goa.loc['primary','usable_percent']:.4f}%**.

При нулевой задержке покрытие составляет **{zero.usable_percent:.4f}%**,
разность **{zero.usable_percent-primary.usable_percent:+.4f} процентного пункта**.
Однако смена выбранного сообщения происходит для **{int(zero.message_changed):,}**
событий (**{zero.message_changed_percent:.2f}%**), а значения или пропуски
хотя бы одного из 13 проверяемых полей изменяются для
**{int(zero.information_changed):,}** (**{zero.information_changed_percent:.2f}%**).
Нулевой сценарий - сравнение допущений, не доказанно правильный ориентир.

Температура меняется в **{int(temp.value_changed):,}** из
**{int(temp.paired_present):,}** пар с двумя заполненными значениями
(**{temp.value_changed_percent_of_pairs:.2f}%**). Разность не меньше 2 °C
найдена в **{int(temp.threshold_changed):,}** таких пар. Порог описательный,
не авиационный норматив; единица анализа - событие, повторно использующее METAR.

Полное исключение COR до сопоставления дает **{nocor.usable_percent:.4f}%**
покрытия всех событий и **{goa.loc['no_cor','usable_percent']:.4f}%** GoA.
Разность покрытия GoA: **{interval.difference_pp:+.4f} п.п.**, парный
bootstrap календарных дней с блоком 14 дней: **[{interval.lower_pp:.4f};
{interval.upper_pp:.4f}] п.п.** Это диапазон временной устойчивости в данном
корпусе, не независимая оценка будущих лет. Блоки 7 и 28 дней также сохранены.
Исключение COR не восстанавливает первоначальную версию исправленного сообщения.

Ограничение давности 30 минут при задержке 15 минут снижает покрытие до
**{overall.loc['age30','usable_percent']:.4f}%**; граница 90 минут повышает
его до **{overall.loc['age90','usable_percent']:.4f}%**. Это описывает обмен
между покрытием и свежестью, не устанавливает оптимальную границу.

## Проверка кодирования

Выборка содержит **{validation['sample_messages']:,}** сообщений:
детерминированные страты аэропорт × год и обогащение редкими форматами.
Отдельный алгоритм прямого чтения токенов выполнил
**{validation['lexical_comparisons']:,}** сравнений. **8** расхождений относятся
к **4** сообщениям с неподдержанной структурой перед читаемым 9999.
Все четыре основные погодные части уже заблокированы. Исходные сообщения
не исправлены. Разбор сохранен в 05_disagreement_adjudication.csv.

Все 8 явно заданных семантических примеров (21 проверка) прошли.
Совпадение числовых кодов не доказывает фактическую правильность погоды.
Доля расхождений обогащенной выборки не оценивает частоту ошибок всего архива.
Независимая внешняя человеческая разметка и фактическое время получения
исторических сообщений отсутствуют.

## Ответ и границы

Да, высокий показатель покрытия скрывает чувствительность самих значений:
изменение задержки с 15 до 0 минут почти сохраняет покрытие, но меняет
проверяемую информацию примерно у половины событий. COR дают значимый вклад
в покрытие, особенно GoA; это не основание считать каждое исправление
своевременно доступным. Поэтому публикация должна сообщать одновременно
покрытие, правила времени, смену сообщений, смену значений и причины пропусков.

Не установлены причинное влияние погоды на уходы, прирост точности модели,
опасность отдельных разностей, оптимальная задержка или пригодность
для эксплуатации. Выборка уже отобрана; test 2023 ранее просматривался.
Обучение не выполнялось. Источники и прежние версии не изменены.
'''
(R/'06_research_answer.md').write_text(text,encoding='utf8')

claims=[
 ('Высокое покрытие не гарантирует устойчивость информации','06_overall_summary.csv','03_coverage_and_information','Сравнение допущений, не фактического времени получения'),
 ('Изменение конкретных полей скрывается за покрытием','06_field_summary.csv','05_field_value_and_missingness','Парные заполненные значения и описательные пороги'),
 ('Чувствительность неоднородна по аэропортам','02_scenario_strata.csv','04_airport_information_sensitivity','Не стандартизовано по годам и составу'),
 ('COR влияет на доступный корпус','04_block_bootstrap.csv','06_block_bootstrap_goa','Не реконструкция исходной версии сообщения'),
 ('Кодирование проверено в ограниченном объеме','05_validation_manifest.json','07_lexical_validation','Не независимая физическая валидация')]
pd.DataFrame(claims,columns=['claim','evidence','figure','limitation']).to_csv(R/'07_article_evidence_map.csv',index=False)
for source in (BASE/'figures').glob('*'):
    if source.suffix in ['.png','.svg']:shutil.copy2(source,article/'figures'/source.name)
for name in ['06_overall_summary.csv','06_field_summary.csv','04_block_bootstrap.csv','05_lexical_summary.csv','07_article_evidence_map.csv']:
    shutil.copy2(R/name,article/'tables'/name)
shutil.copy2(R/'06_research_answer.md',article/'notes/research_answer.md')
print('Research answer and article 04 evidence saved',flush=True)
