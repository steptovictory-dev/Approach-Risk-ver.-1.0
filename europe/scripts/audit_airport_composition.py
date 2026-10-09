"""07_05: состав типов, сегментов и пропуски обеих меток без очистки."""
import json
from collections import Counter
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
from audit_airports import ROOT, INPUT, wilson
from audit_feature_availability import digest

REPORT=ROOT/'europe/reports/airports/composition'
FIGURES=ROOT/'europe/figures/07_airports'
STRINGS=['registration','typecode','icaoaircrafttype','aircraft_description_restored','market_segment','AC_CLASS','AP_C_RWY']
C40=['C40_CROSS_TIME','C40_BEARING','C40_CROSS_LAT','C40_CROSS_LON']
DISPLAY={'registration':'Регистрация','typecode':'Код типа','icaoaircrafttype':'Исходная классификация','aircraft_description_restored':'Восстановленное описание','market_segment':'Сегмент','AC_CLASS':'AC_CLASS','AP_C_RWY':'AP_C_RWY','classification_unavailable':'Классификация после 05',**{x:x for x in C40}}


def collapse(parts,keys):
    return pd.concat(parts,ignore_index=True).groupby(keys,dropna=False).sum(numeric_only=True).reset_index()


def md(table):
    return '| '+' | '.join(map(str,table.columns))+' |\n| '+' | '.join(['---']*len(table.columns))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in table.itertuples(index=False,name=None))


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    expected=json.loads((ROOT/'europe/reports/airports/07_01_audit.json').read_text(encoding='utf8'))
    before=digest(INPUT);assert before==expected['input_sha256']
    groups=[];miss=[];categories=[];periods=[];examples=[];selected=Counter()
    fields=['flight_id','icao24','airport','landing_time','GoA',*STRINGS,*C40]
    for batch in pq.ParquetFile(INPUT).iter_batches(columns=fields,batch_size=100000):
        frame=batch.to_pandas(ignore_metadata=True)
        groups.append(frame.groupby(['airport','GoA']).size().reset_index(name='rows'))
        for field in STRINGS+C40+['classification_unavailable']:
            if field=='classification_unavailable':
                source=frame.icaoaircrafttype
                available=source.notna()&source.ne('')&source.str.strip().ne('')
                values=source.where(available,frame.aircraft_description_restored)
            else:values=frame[field]
            flags=pd.DataFrame({'airport':frame.airport,'GoA':frame.GoA,'null':values.isna().astype(int),'empty':0,'whitespace':0})
            if field in STRINGS or field=='classification_unavailable':
                flags['empty']=values.eq('').fillna(False).astype(int)
                flags['whitespace']=(values.notna()&values.ne('')&values.str.strip().eq('')).astype(int)
            result=flags.groupby(['airport','GoA']).sum(numeric_only=True).reset_index();result['field']=field;miss.append(result)
            if field in ['typecode','market_segment','classification_unavailable',*C40]:
                bad=flags[['null','empty','whitespace']].sum(axis=1).gt(0)
                for (airport,label),part in frame[bad].groupby(['airport','GoA']):
                    key=(airport,bool(label),field);count=3-selected[key]
                    if count>0:
                        sample=part.head(count)[['flight_id','icao24','airport','landing_time','GoA','typecode','icaoaircrafttype','aircraft_description_restored','market_segment',*C40]].copy()
                        sample['review_field']=field;examples.append(sample);selected[key]+=len(sample)
        for field in ['typecode','market_segment']:
            group=frame[['airport','GoA',field]].copy()
            # Категории отсутствия отделены от сырых значений без их перезаписи.
            group['value_kind']=np.select([group[field].isna(),group[field].eq('').fillna(False),group[field].notna()&group[field].ne('')&group[field].str.strip().eq('')],['null','empty','whitespace'],default='value')
            group['value']=group[field].fillna('')
            cat=group.groupby(['airport','GoA','value_kind','value'],dropna=False).size().reset_index(name='rows');cat['field']=field;categories.append(cat)
        time=frame[['airport','GoA','typecode','market_segment']].copy();time['month']=frame.landing_time.dt.strftime('%Y-%m')
        time['type_missing']=time.typecode.isna()|time.typecode.eq('').fillna(False)|time.typecode.str.strip().eq('').fillna(False)
        time['segment_missing']=time.market_segment.isna()|time.market_segment.eq('').fillna(False)|time.market_segment.str.strip().eq('').fillna(False)
        periods.append(time.groupby(['airport','GoA','month']).agg(rows=('GoA','size'),type_missing=('type_missing','sum'),segment_missing=('segment_missing','sum')).reset_index())
    totals=collapse(groups,['airport','GoA']);missing=collapse(miss,['airport','GoA','field']).merge(totals,on=['airport','GoA'],validate='many_to_one')
    missing['missing']=missing['null']+missing['empty']+missing['whitespace']
    missing['missing_percent']=100*missing['missing']/missing.rows
    cats=collapse(categories,['airport','GoA','field','value_kind','value']).merge(totals.rename(columns={'rows':'label_rows'}),on=['airport','GoA'],validate='many_to_one')
    cats['within_label_percent']=100*cats.rows/cats.label_rows
    monthly=collapse(periods,['airport','GoA','month'])
    for field in ['type','segment']:monthly[field+'_missing_percent']=100*monthly[field+'_missing']/monthly.rows
    assert totals.rows.sum()==expected['rows'] and totals.loc[totals.GoA,'rows'].sum()==expected['go_arounds']
    shifts=missing.pivot(index=['airport','field'],columns='GoA',values='missing_percent').rename(columns={False:'false_percent',True:'true_percent'}).reset_index()
    shifts['gap_pp']=shifts.true_percent-shifts.false_percent
    summary=[]
    for airport in sorted(totals.airport.unique()):
        row={'airport':airport}
        for label,suffix in [(False,'false'),(True,'true')]:
            pop=totals[totals.airport.eq(airport)&totals.GoA.eq(label)].rows.iloc[0];row['rows_'+suffix]=int(pop)
            types=cats[cats.airport.eq(airport)&cats.GoA.eq(label)&cats.field.eq('typecode')&cats.value_kind.eq('value')].sort_values('rows',ascending=False)
            row['unique_types_'+suffix]=len(types)
            row['top5_types_percent_'+suffix]=100*types.head(5).rows.sum()/pop
            row['top_type_'+suffix]=types.iloc[0].value if len(types) else '(нет)'
            row['top_type_percent_'+suffix]=100*types.iloc[0].rows/pop if len(types) else 0
        for field in ['typecode','market_segment']:
            d=cats[cats.airport.eq(airport)&cats.field.eq(field)].pivot(index=['value_kind','value'],columns='GoA',values='within_label_percent').fillna(0)
            row[field+'_total_variation']=.5*(d[True]-d[False]).abs().sum()/100
        summary.append(row)
    summary=pd.DataFrame(summary)
    counts=cats.pivot(index=['airport','field','value_kind','value'],columns='GoA',values='rows').fillna(0).rename(columns={False:'goa_false',True:'go_arounds'}).reset_index()
    counts['rows']=counts.goa_false+counts.go_arounds;counts=wilson(counts)
    # Доли GoA по категориям являются описательными и смешивают календарь и состав.
    for data,name in [(totals,'label_totals'),(missing,'missing'),(cats,'categories'),(monthly,'monthly_missing'),(shifts,'missing_gaps'),(summary,'composition_summary'),(counts,'category_rates'),(pd.concat(examples,ignore_index=True),'review_records')]:
        data.to_csv(REPORT/f'07_05_{name}.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
    airports=summary.sort_values('rows_false',ascending=False).airport.tolist()
    selected_fields=['typecode','icaoaircrafttype','classification_unavailable','market_segment','registration','AC_CLASS','AP_C_RWY']
    fig,axes=plt.subplots(1,2,figsize=(14,9),sharey=True)
    for ax,label in zip(axes,[False,True]):
        values=missing[missing.GoA.eq(label)].pivot(index='airport',columns='field',values='missing_percent').reindex(index=airports,columns=selected_fields)
        im=ax.imshow(values,vmin=0,vmax=100,cmap='YlOrBr',aspect='auto');ax.set_xticks(range(len(selected_fields)),[DISPLAY[f] for f in selected_fields],rotation=45,ha='right');ax.set_yticks(range(len(airports)),airports);ax.set_title(f'GoA={label}')
        for y in range(len(airports)):
            for x in range(len(selected_fields)):
                val=values.iloc[y,x];ax.text(x,y,f'{val:.1f}',ha='center',va='center',fontsize=7,color='white' if val>60 else '#223344')
    fig.suptitle('07_05: пропуски по аэропортам и меткам, %');fig.subplots_adjust(bottom=.25,right=.9,wspace=.1)
    fig.colorbar(im,cax=fig.add_axes([.93,.25,.015,.6]),label='Пропуски, %')
    fig.text(.01,.02,'Null, пустые и пробельные строки учтены отдельно в CSV. Восстановление 05 оценивается совместно с исходным полем.',fontsize=9)
    for ext in ['png','svg','pdf']:fig.savefig(FIGURES/f'07_05_composition_missing.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,8));ordered=summary.set_index('airport').loc[airports];y=np.arange(len(airports))
    ax.barh(y-.18,ordered.top5_types_percent_false,height=.35,label='GoA=False',color='#537c99');ax.barh(y+.18,ordered.top5_types_percent_true,height=.35,label='GoA=True',color='#b45335')
    ax.set_yticks(y,airports);ax.invert_yaxis();ax.set(xlim=(0,100),xlabel='Доля всех записей метки, %',title='Пять наиболее частых известных типов внутри каждой метки');ax.legend();ax.grid(axis='x',alpha=.15);fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(FIGURES/f'07_05_type_concentration.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    short=summary[['airport','rows_false','rows_true','unique_types_false','unique_types_true','top_type_false','top_type_true']].rename(columns={'airport':'Аэропорт','rows_false':'n False','rows_true':'n True','unique_types_false':'Типов False','unique_types_true':'Типов True','top_type_false':'Первый тип False','top_type_true':'Первый тип True'})
    flagged=shifts[shifts.field.isin(['typecode','market_segment','classification_unavailable'])].assign(abs_gap=lambda x:x.gap_pp.abs()).sort_values('abs_gap',ascending=False).head(12)
    gaps=flagged[['airport','field','false_percent','true_percent','gap_pp']].copy()
    for field in ['false_percent','true_percent','gap_pp']:gaps[field]=gaps[field].map(lambda x:f'{x:.2f}')
    gaps=gaps.rename(columns={'airport':'Аэропорт','field':'Поле','false_percent':'Пропуски False, %','true_percent':'Пропуски True, %','gap_pp':'True − False, п.п.'})
    allmiss=missing.groupby('field')[['missing','rows']].sum();allmiss['percent']=100*allmiss['missing']/allmiss.rows
    before_restore=allmiss.loc['icaoaircrafttype','missing'];after_restore=allmiss.loc['classification_unavailable','missing']
    text=f'''# 07_05 – Состав типов и пропуски по аэропортам

Проверены все {int(totals.rows.sum()):,} записей актуальной выборки, включая {int(totals.loc[totals.GoA,'rows'].sum()):,} GoA=True. Аэропортов: {len(summary)}. Вход подтвержден SHA256 отчета 07_01. Состав и значения не изменены.

## Состав обеих меток

'''+md(short)+f'''
## Пропуски и восстановление

Классификация недоступна в исходном поле у {int(before_restore):,} записей; после дополнения описанием этапа 05 – у {int(after_restore):,}. Пропуск восстановленного описания сам по себе не означает отсутствие классификации: поле заполнялось только для части записей. Итоговая доступность не подтверждает корректность классификации.

Наибольшие различия доли пропусков между метками (описательное ранжирование):

'''+md(gaps)+'''
![Пропуски обеих меток](../../../figures/07_airports/07_05_composition_missing.png)

![Концентрация типов](../../../figures/07_airports/07_05_type_concentration.png)

## Интерпретация и ограничения

Количество известных типов и их концентрация зависят от объема группы; малый положительный класс нельзя считать сопоставимой по полноте описью флота. Первая категория и пять наиболее частых типов выбираются отдельно внутри каждой метки. Знаменатель включает записи с отсутствующим типом.

Различия состава и пропусков не доказывают влияние типа на GoA. Они могут смешивать аэропорт, календарь, сегмент и справочное обогащение. Total variation в CSV – половина суммы абсолютных различий долей категорий, от 0 до 1; категория пропуска включена, временное смешение не устранено. Это описательная дистанция, не тест значимости и не правило удаления.

Категориальные коды сохранены как есть: регистр и варианты кодов не объединены. Null, точные пустые строки и пробельные строки подсчитаны раздельно. Для итоговой классификации исходное непустое значение дополняется непустым восстановленным описанием; формат и историческая корректность здесь не подтверждены.

`07_05_monthly_missing.csv` показывает пропуски типа и сегмента по месяцам и обеим меткам. `07_05_review_records.csv` содержит до трех конкретных примеров каждой группы аэропорт × метка × проверяемое поле: исходные идентификаторы и значения сохранены, выбор первых записей не является случайной выборкой. Эти записи – материал для проверки происхождения пропуска, а не ошибки и не кандидаты автоматического удаления.

Метаданные C40 проверены на null. Доступность справочника до C40 и допустимость признаков для модели решаются позднее. GoA=False – отсутствие отмеченного ухода, а не независимое подтверждение нормального профиля. Старые исключения и пороги не переносятся.

## Решение

Новых исключений и исправлений не применено. Решение 07_04 сохранить записи погодного разбора действует. Отдельное решение по результатам 07_05 еще не принято.
'''
    (REPORT/'07_05_review.md').write_text(text,encoding='utf8')
    code="""from pathlib import Path
import pandas as pd
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'europe/reports/airports/composition/07_05_missing.csv').exists())
report = root/'europe/reports/airports/composition'
pd.read_csv(report/'07_05_composition_summary.csv')
"""
    inspect="""# Измените аэропорт, чтобы проверить конкретные исходные записи.
airport = 'EGCC'
records = pd.read_csv(report/'07_05_review_records.csv', dtype={'icao24': str}, keep_default_na=False)
records.loc[records.airport.eq(airport)]
"""
    notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(text.replace('../../../figures/','../../figures/')),nbformat.v4.new_code_cell(code),nbformat.v4.new_code_cell(inspect),nbformat.v4.new_markdown_cell('Воспроизведение из корня: `.\\.venv\\Scripts\\python.exe -u -X utf8 europe/scripts/audit_airport_composition.py`. [Полные таблицы и отчет](../../reports/airports/composition/07_05_review.md).')],metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
    nbformat.validate(notebook);path=ROOT/'europe/notebooks/07_airports/07_05_composition_missing.ipynb';nbformat.write(notebook,path)
    assert digest(INPUT)==before
    audit=dict(input_sha256=before,rows=int(totals.rows.sum()),go_arounds=int(totals.loc[totals.GoA,'rows'].sum()),airports=len(summary),rows_deleted=0,values_changed=0,input_unchanged=True,original_classification_missing=int(before_restore),effective_classification_missing=int(after_restore),c40_missing=int(missing[missing.field.isin(C40)]['missing'].sum()),null_empty_whitespace_separate=True)
    (REPORT/'07_05_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    for path,note in [(ROOT/'europe/notebooks/07_airports/README.md','\n\n- [07_05 – Состав типов и пропуски обеих меток](07_05_composition_missing.ipynb): все аэропорты, месячные пропуски и конкретные записи для проверки.\n'),(ROOT/'europe/reports/airports/README.md','\n\n[07_05 – Состав типов и пропуски](composition/07_05_review.md). Рабочий разбор – в общем ноутбуке папки 07.\n')]:
        current=path.read_text(encoding='utf8')
        if note.strip() not in current:path.write_text(current+note,encoding='utf8')
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)
    print(gaps.to_string(index=False),flush=True)


if __name__=='__main__':main()
