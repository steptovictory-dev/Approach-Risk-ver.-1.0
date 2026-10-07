"""Полный аудит европейского Parquet без фильтрации и исправления исходника."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import platform
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REGION = Path(__file__).resolve().parents[1]
ROOT = REGION.parent
SOURCE = REGION / 'data/source/osn23_landings_merged.parquet.gzip'
REPORTS = REGION / 'reports'
REVIEW = REGION / 'data/review'
BATCH_SIZE = 100_000
CITY = {'EDDB':'Берлин','EDDF':'Франкфурт-на-Майне','EDDM':'Мюнхен','EGCC':'Манчестер',
        'EGLL':'Лондон','EGSS':'Лондон','EHAM':'Амстердам','EIDW':'Дублин','EKCH':'Копенгаген',
        'EPWA':'Варшава','ESSA':'Стокгольм','LEMD':'Мадрид','LFPG':'Париж','LFPO':'Париж',
        'LIMC':'Милан','LIRF':'Рим','LOWW':'Вена','LSGG':'Женева','LSZH':'Цюрих','LTFJ':'Стамбул'}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(name, obj):
    (REPORTS / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str, allow_nan=False)+'\n', encoding='utf8')


def add_groups(counter, frame, columns):
    grouped = frame.groupby(columns, dropna=False, observed=True)['GoA'].agg(['size','sum'])
    for key, record in grouped.iterrows():
        key = key if isinstance(key, tuple) else (key,)
        key = tuple('<NULL>' if pd.isna(value) else str(value) for value in key)
        counter[(key, 'rows')] += int(record['size'])
        counter[(key, 'go_arounds')] += int(record['sum'])


def group_table(counter, columns):
    keys = sorted({key for key, metric in counter})
    result = pd.DataFrame([{**dict(zip(columns,key)), 'rows':counter[(key,'rows')],
                            'go_arounds':counter[(key,'go_arounds')]} for key in keys])
    result['ga_percent'] = result['go_arounds'] / result['rows'] * 100
    return result


def timestamp_lists(batch):
    values = batch.column('attempt_times')
    flat = pc.list_flatten(values).to_numpy(zero_copy_only=False).astype('datetime64[ns]')
    parents = pc.list_parent_indices(values).to_numpy(zero_copy_only=False)
    n = batch.num_rows
    first = np.full(n, np.datetime64('NaT', 'ns'), dtype='datetime64[ns]')
    last = first.copy()
    reversed_rows = np.zeros(n, dtype=bool)
    repeated_rows = reversed_rows.copy()
    null_rows = reversed_rows.copy()
    if len(flat):
        first_mask = np.r_[True, parents[1:] != parents[:-1]]
        last_mask = np.r_[parents[1:] != parents[:-1], True]
        first[parents[first_mask]] = flat[first_mask]
        last[parents[last_mask]] = flat[last_mask]
        same = parents[1:] == parents[:-1]
        valid = same & ~np.isnat(flat[1:]) & ~np.isnat(flat[:-1])
        reversed_rows[parents[1:][valid & (flat[1:] < flat[:-1])]] = True
        repeated_rows[parents[1:][valid & (flat[1:] == flat[:-1])]] = True
        null_rows[parents[np.isnat(flat)]] = True
    return flat, parents, first, last, reversed_rows, repeated_rows, null_rows


def profile_source():
    REPORTS.mkdir(parents=True, exist_ok=True)
    REVIEW.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(SOURCE)
    manifest = json.loads((ROOT/'docs/sources/download_manifest.json').read_text(encoding='utf8'))
    expected = next(f['sha256'] for f in manifest['files'] if f['path'] == SOURCE.relative_to(ROOT).as_posix())
    if source_hash != expected:
        raise ValueError('SHA256 исходника отличается от реестра источников')
    parquet = pq.ParquetFile(SOURCE)
    required = {'flight_id','callsign','icao24','landing_time','registration','typecode','icaoaircrafttype',
                'airport','country','attempt_times','ILS','n_attempts','GoA','market_segment','AC_CLASS',
                'AP_C_RWY','C40_CROSS_TIME','C40_BEARING','C40_CROSS_LAT','C40_CROSS_LON'}
    if not required.issubset(parquet.schema_arrow.names):
        raise ValueError('В исходнике отсутствуют необходимые поля аудита')
    if not pa.types.is_boolean(parquet.schema_arrow.field('GoA').type):
        raise ValueError('Неожиданный тип целевой метки')
    rows = 0
    target_true = 0
    missing = Counter()
    blank = Counter()
    whitespace = Counter()
    markers = Counter()
    unique = {f.name:set() for f in parquet.schema_arrow if pa.types.is_string(f.type) and f.name!='flight_id'}
    numeric = {name:{'count':0,'nonfinite':0,'min':None,'max':None} for name in ['n_attempts','C40_BEARING','C40_CROSS_LAT','C40_CROSS_LON']}
    groups = {name:Counter() for name in ['country','airport','month','year','airport_month','airport_day','attempts']}
    categories = {name:Counter() for name in ['market_segment','AC_CLASS','icaoaircrafttype','typecode','AP_C_RWY']}
    flag_counts = Counter()
    flag_ga = Counter()
    any_count = 0
    any_ga = 0
    examples = []
    first_time = last_time = None
    core_hashes = []
    duration_parts = []
    list_ranges = {'attempt_times':Counter(),'ILS':Counter()}
    scalar_names = [f.name for f in parquet.schema_arrow if not pa.types.is_list(f.type)]
    writer = None
    partial = REVIEW/'flagged_rows.parquet.partial'
    try:
        for batch in parquet.iter_batches(batch_size=BATCH_SIZE):
            start = rows
            rows += batch.num_rows
            frame = batch.select(scalar_names).to_pandas(ignore_metadata=True)
            ga = frame['GoA'].fillna(False).to_numpy(dtype=bool)
            target_true += int(ga.sum())
            for field in batch.schema:
                column = batch.column(field.name)
                missing[field.name] += column.null_count
                if pa.types.is_string(field.type):
                    blank[field.name] += int(pc.sum(pc.cast(pc.equal(column,''),pa.int64())).as_py() or 0)
                    trim = pc.utf8_trim_whitespace(column)
                    white = pc.and_(pc.equal(trim,''),pc.not_equal(column,''))
                    whitespace[field.name] += int(pc.sum(pc.cast(white,pa.int64())).as_py() or 0)
                    mark = pc.is_in(pc.utf8_lower(trim),value_set=pa.array(['none','null','nan','unknown','n/a']))
                    markers[field.name] += int(pc.sum(pc.cast(mark,pa.int64())).as_py() or 0)
                    if field.name in unique:
                        unique[field.name].update(v for v in pc.unique(column).to_pylist() if v is not None)
            times = frame['landing_time']
            lo, hi = times.min(), times.max()
            if pd.notna(lo):
                first_time = lo if first_time is None else min(first_time,lo)
                last_time = hi if last_time is None else max(last_time,hi)
            frame['month'] = times.dt.strftime('%Y-%m')
            frame['year'] = times.dt.strftime('%Y')
            frame['date'] = times.dt.strftime('%Y-%m-%d')
            for name, cols in [('country',['country']),('airport',['country','airport']),('month',['month']),
                               ('year',['year']),('airport_month',['airport','month']),
                               ('airport_day',['airport','date']),('attempts',['n_attempts'])]:
                add_groups(groups[name],frame,cols)
            for name in categories:
                add_groups(categories[name],frame,[name])
            core_hashes.append(pd.util.hash_pandas_object(frame[['landing_time','icao24','airport']],index=False).to_numpy())
            for name, stats in numeric.items():
                values = frame[name].to_numpy(dtype=float,na_value=np.nan)
                finite = np.isfinite(values)
                stats['count'] += int(finite.sum())
                stats['nonfinite'] += int((~finite & frame[name].notna().to_numpy()).sum())
                if finite.any():
                    lo, hi = float(values[finite].min()), float(values[finite].max())
                    stats['min'] = lo if stats['min'] is None else min(stats['min'],lo)
                    stats['max'] = hi if stats['max'] is None else max(stats['max'],hi)
            flat, parents, first, last, reverse, repeated, null_elements = timestamp_lists(batch)
            landing = times.to_numpy(dtype='datetime64[ns]')
            crossing = frame['C40_CROSS_TIME'].to_numpy(dtype='datetime64[ns]')
            length_attempt = pc.list_value_length(batch.column('attempt_times')).to_numpy(zero_copy_only=False)
            length_runway = pc.list_value_length(batch.column('ILS')).to_numpy(zero_copy_only=False)
            for name,lengths in [('attempt_times',length_attempt),('ILS',length_runway)]:
                list_ranges[name].update(int(v) for v in lengths[np.isfinite(lengths)])
            has_attempt = ~np.isnat(first)
            has_last = ~np.isnat(last)
            has_landing = ~np.isnat(landing)
            has_crossing = ~np.isnat(crossing)
            n_attempts = frame['n_attempts'].to_numpy(dtype=float,na_value=np.nan)
            flags = {
                'landing_time:null':~has_landing,
                'GoA:null':frame['GoA'].isna().to_numpy(),
                'n_attempts:null_or_nonpositive':~np.isfinite(n_attempts)|(n_attempts<1),
                'n_attempts:more_than_two':n_attempts>2,
                'GoA:inconsistent_with_n_attempts':frame['GoA'].notna().to_numpy() & np.isfinite(n_attempts) & (ga!=(n_attempts>1)),
                'attempt_times:null_or_empty':~np.isfinite(length_attempt)|(length_attempt==0),
                'attempt_times:length_vs_n_attempts':np.isfinite(length_attempt)&np.isfinite(n_attempts)&(length_attempt!=n_attempts),
                'attempt_times:null_element':null_elements,
                'attempt_times:descending_order':reverse,
                'attempt_times:repeated_adjacent_time':repeated,
                'attempt_times:last_after_landing':has_last&has_landing&(last>landing),
                'ILS:null_or_empty':~np.isfinite(length_runway)|(length_runway==0),
                'ILS:length_vs_n_attempts':np.isfinite(length_runway)&np.isfinite(n_attempts)&(length_runway!=n_attempts),
                'C40_CROSS_TIME:after_landing':has_crossing&has_landing&(crossing>landing),
                'C40_CROSS_TIME:after_first_attempt':has_crossing&has_attempt&(crossing>first),
                'icao24:unexpected_format':frame['icao24'].notna().to_numpy() & ~frame['icao24'].str.fullmatch(r'[0-9a-fA-F]{6}',na=True).to_numpy(dtype=bool),
                'airport:unexpected_format':frame['airport'].notna().to_numpy() & ~frame['airport'].str.fullmatch(r'[A-Z]{4}',na=True).to_numpy(dtype=bool),
                'icaoaircrafttype:outside_dominant_description_pattern':frame['icaoaircrafttype'].fillna('').str.strip().ne('').to_numpy(dtype=bool) & ~frame['icaoaircrafttype'].str.fullmatch(r'[A-Z][0-9][A-Z]',na=True).to_numpy(dtype=bool),
                'icaoaircrafttype:equals_typecode_ignoring_case':frame['icaoaircrafttype'].fillna('').str.strip().ne('').to_numpy(dtype=bool) & frame['icaoaircrafttype'].str.upper().eq(frame['typecode'].str.upper()).fillna(False).to_numpy(dtype=bool),
            }
            for name,(lower,upper,upper_exclusive) in {'C40_BEARING':(0,360,True),'C40_CROSS_LAT':(-90,90,False),'C40_CROSS_LON':(-180,180,False)}.items():
                values=frame[name].to_numpy(dtype=float,na_value=np.nan)
                present=frame[name].notna().to_numpy()
                outside=(values<lower)|(values>=upper if upper_exclusive else values>upper)|~np.isfinite(values)
                flags[f'{name}:outside_diagnostic_range']=present & outside
            coordinate_presence = frame[['C40_CROSS_LAT','C40_CROSS_LON']].notna().to_numpy()
            flags['C40_coordinates:only_one_coordinate'] = coordinate_presence[:,0] != coordinate_presence[:,1]
            flags['C40_metadata:coordinates_without_time'] = coordinate_presence.all(axis=1) & ~has_crossing
            flags['C40_metadata:time_without_coordinates'] = has_crossing & ~coordinate_presence.all(axis=1)
            # Формат обозначений ВПП является диагностикой, а не проверкой действовавших схем.
            runway_flat=pc.list_flatten(batch.column('ILS'))
            runway_parents=pc.list_parent_indices(batch.column('ILS')).to_numpy(zero_copy_only=False)
            valid_runway=pc.fill_null(pc.match_substring_regex(runway_flat,r'^(0[1-9]|[12][0-9]|3[0-6])[LCR]?$'),False).to_numpy(zero_copy_only=False)
            bad_runway=np.zeros(batch.num_rows,dtype=bool)
            bad_runway[runway_parents[~valid_runway]]=True
            flags['ILS:unexpected_designator_format']=bad_runway
            duration=(landing-crossing)/np.timedelta64(1,'m')
            duration_parts.append(duration[np.isfinite(duration)])
            flags_frame=pd.DataFrame(flags)
            any_flag=flags_frame.any(axis=1).to_numpy()
            any_count+=int(any_flag.sum())
            any_ga+=int((any_flag&ga).sum())
            for reason,mask in flags.items():
                flag_counts[reason]+=int(mask.sum())
                flag_ga[reason]+=int((mask&ga).sum())
                previous=sum(e['review_reason']==reason for e in examples)
                if previous<3:
                    for position in np.flatnonzero(mask)[:3-previous]:
                        example=batch.slice(int(position),1).to_pylist()[0]
                        example.update(source_row=start+int(position),review_reason=reason)
                        examples.append(example)
            if any_flag.any():
                selected=batch.filter(pa.array(any_flag))
                selected=selected.append_column('source_row',pa.array(start+np.flatnonzero(any_flag),type=pa.int64()))
                reasons=flags_frame.loc[any_flag].apply(lambda r:list(r.index[r]),axis=1).tolist()
                selected=selected.append_column('review_flags',pa.array(reasons,type=pa.list_(pa.string())))
                if writer is None: writer=pq.ParquetWriter(partial,selected.schema,compression='zstd')
                writer.write_batch(selected)
            if rows % 1_000_000 == 0:
                print(f'Проверено {rows:,} строк.',flush=True)
    finally:
        if writer is not None: writer.close()
    if writer is not None: partial.replace(REVIEW/'flagged_rows.parquet')
    schema_rows=[]
    for field in parquet.schema_arrow:
        schema_rows.append({'column':field.name,'arrow_type':str(field.type),'null_count':missing[field.name],
                            'null_percent':100*missing[field.name]/rows,'empty_string_count':blank[field.name],
                            'whitespace_only_count':whitespace[field.name],'text_missing_marker_count':markers[field.name],
                            'unique_nonnull_strings':len(unique[field.name]) if field.name in unique else None,
                            'role':'storage_index' if field.name=='__index_level_0__' else 'data'})
    pd.DataFrame(schema_rows).to_csv(REPORTS/'missingness.csv',index=False)
    pd.DataFrame([{'column':k,**v} for k,v in numeric.items()]).to_csv(REPORTS/'numeric_ranges.csv',index=False)
    table_columns={'country':['country'],'airport':['country','airport'],'month':['month'],'year':['year'],
                   'airport_month':['airport','month'],'airport_day':['airport','date'],'attempts':['n_attempts']}
    tables={name:group_table(groups[name],columns) for name,columns in table_columns.items()}
    tables['attempts']['n_attempts']=pd.to_numeric(tables['attempts']['n_attempts'],errors='coerce')
    tables['attempts']=tables['attempts'].sort_values('n_attempts')
    tables['airport'].insert(2,'city',tables['airport']['airport'].map(CITY))
    for name,table in tables.items():
        if int(table['rows'].sum())!=rows or int(table['go_arounds'].sum())!=target_true:
            raise ValueError(f'Не сходятся агрегаты: {name}')
        table.to_csv(REPORTS/f'{name}_summary.csv',index=False)
    for name,counter in categories.items():
        group_table(counter,[name]).sort_values('rows',ascending=False).to_csv(REPORTS/f'{name}_summary.csv',index=False)
    calendar=pd.date_range(first_time.normalize(),last_time.normalize(),freq='D').strftime('%Y-%m-%d')
    airport_codes=sorted(tables['airport']['airport'].unique())
    full_calendar=pd.MultiIndex.from_product([airport_codes,calendar],names=['airport','date'])
    daily=tables['airport_day'].set_index(['airport','date'])[['rows','go_arounds']].reindex(full_calendar,fill_value=0).reset_index()
    daily['has_observed_records']=daily['rows']>0
    daily.to_csv(REPORTS/'airport_calendar.csv',index=False)
    coverage=daily.groupby('airport').agg(observed_days=('has_observed_records','sum'),calendar_days=('date','size'),first_calendar_day=('date','min'),last_calendar_day=('date','max')).reset_index()
    coverage['days_without_records']=coverage['calendar_days']-coverage['observed_days']
    coverage['observed_day_percent']=100*coverage['observed_days']/coverage['calendar_days']
    coverage.to_csv(REPORTS/'coverage_summary.csv',index=False)
    flags_table=pd.DataFrame([{'reason':reason,'rows':count,'go_arounds':flag_ga[reason],'percent':100*count/rows} for reason,count in flag_counts.items()]).sort_values('rows',ascending=False)
    flags_table.to_csv(REPORTS/'review_flag_summary.csv',index=False)
    save_json('review_examples.json',examples)
    pd.DataFrame([{'field':name,'list_length':length,'rows':count} for name,counts in list_ranges.items() for length,count in sorted(counts.items())]).to_csv(REPORTS/'list_lengths.csv',index=False)
    durations=np.concatenate(duration_parts)
    duration_summary={'rows_with_both_times':len(durations),'unit':'minutes','min':float(durations.min()) if len(durations) else None,'max':float(durations.max()) if len(durations) else None,'quantiles':{str(q):float(np.quantile(durations,q)) for q in [.01,.5,.99]} if len(durations) else {}}
    save_json('c40_to_landing_duration.json',duration_summary)
    # Хеши только отбирают кандидатов; итоговые повторы проверяются по точным значениям.
    hashes=np.concatenate(core_hashes)
    values,counts=np.unique(hashes,return_counts=True)
    candidate_hashes=values[counts>1]
    del hashes,values,counts,core_hashes,duration_parts,durations,unique
    identifiers=pq.read_table(SOURCE,columns=['flight_id']).column('flight_id')
    id_counts=pc.value_counts(identifiers)
    repeated_ids=pc.filter(id_counts,pc.greater(id_counts.field('counts'),1))
    top_ids=sorted(zip(repeated_ids.field('values').to_pylist(),repeated_ids.field('counts').to_pylist()),key=lambda pair:pair[1],reverse=True)[:10]
    save_json('repeated_flight_ids.json',{'repeated_values':len(repeated_ids),'excess_rows':sum(repeated_ids.field('counts').to_pylist())-len(repeated_ids),'top_values':top_ids,'note':'Совпадение flight_id не является правилом удаления.'})
    candidate_rows=[]
    identifier_examples=[]
    row_offset=0
    top_values=pa.array([key for key,count in top_ids],type=pa.string())
    for batch in parquet.iter_batches(batch_size=BATCH_SIZE):
        core=batch.select(['landing_time','icao24','airport']).to_pandas(ignore_metadata=True)
        mask=np.isin(pd.util.hash_pandas_object(core,index=False).to_numpy(),candidate_hashes)
        if mask.any():
            selected=batch.filter(pa.array(mask)).append_column('source_row',pa.array(row_offset+np.flatnonzero(mask),type=pa.int64()))
            candidate_rows.extend(selected.to_pylist())
        if len(identifier_examples)<30:
            examples_batch=batch.filter(pc.is_in(batch.column('flight_id'),value_set=top_values))
            identifier_examples.extend(examples_batch.to_pylist()[:30-len(identifier_examples)])
        row_offset+=batch.num_rows
    key_counts=Counter((str(row['landing_time']),row['icao24'],row['airport']) for row in candidate_rows)
    duplicate_keys={key for key,count in key_counts.items() if count>1}
    duplicate_rows=[r for r in candidate_rows if (str(r['landing_time']),r['icao24'],r['airport']) in duplicate_keys]
    business_serialized=[json.dumps({k:v for k,v in r.items() if k not in ['source_row','__index_level_0__']},sort_keys=True,default=str) for r in duplicate_rows]
    business_duplicates=len(business_serialized)-len(set(business_serialized))
    duplicate_report={'core_key':['landing_time','icao24','airport'],'repeated_core_keys':len(duplicate_keys),
                      'rows_in_repeated_core_keys':len(duplicate_rows),'excess_core_key_rows':len(duplicate_rows)-len(duplicate_keys),
                      'full_duplicate_rows_excluding_storage_index':business_duplicates,'flight_id_repeated_values':len(repeated_ids),
                      'flight_id_excess_rows':sum(repeated_ids.field('counts').to_pylist())-len(repeated_ids),
                      'core_hash_candidate_rows':len(candidate_rows),'exact_core_key_examples':duplicate_rows[:10],
                      'repeated_identifier_examples':identifier_examples,'method':'64-bit screening of composite key followed by exact key and all business-field comparison; storage index excluded.',
                      'rows_removed':0}
    save_json('duplicate_check.json',duplicate_report)
    if duplicate_rows:
        pq.write_table(pa.Table.from_pylist(duplicate_rows),REVIEW/'duplicate_key_rows.parquet',compression='zstd')
    after_hash=sha256(SOURCE)
    if after_hash!=source_hash: raise ValueError('Исходник изменился во время аудита')
    if rows!=parquet.metadata.num_rows: raise ValueError('Проверено не все строки исходника')
    summary={'audit_date':'2026-10-08','source':SOURCE.relative_to(ROOT).as_posix(),'source_bytes':SOURCE.stat().st_size,
             'source_sha256':source_hash,'source_unchanged':True,'rows':rows,'physical_columns':len(parquet.schema_arrow),
             'business_columns':len([f for f in parquet.schema_arrow if f.name!='__index_level_0__']),
             'airports':len(airport_codes),'countries':int(tables['country'].shape[0]),'go_arounds':target_true,
             'ga_percent':100*target_true/rows,'min_landing_time':str(first_time),'max_landing_time':str(last_time),
             'unique_flagged_rows':any_count,'flagged_go_arounds':any_ga,'rows_deleted':0,'values_corrected':0,
             'repeated_flight_ids':duplicate_report['flight_id_repeated_values'],'excess_flight_id_rows':duplicate_report['flight_id_excess_rows'],
             'repeated_core_keys':len(duplicate_keys),'excess_core_key_rows':duplicate_report['excess_core_key_rows'],
             'full_duplicate_rows_excluding_storage_index':business_duplicates,'max_n_attempts':numeric['n_attempts']['max'],
             'rules_status':'diagnostic_not_cleaning_rules','runtime':{'python':platform.python_version(),'pandas':pd.__version__,'pyarrow':pa.__version__},
             'schema':[{'name':f.name,'type':str(f.type)} for f in parquet.schema_arrow]}
    save_json('profile_summary.json',summary)
    lines=['# Аудит базы данных Европы','',f"Проверено {rows:,} строк, {len(airport_codes)} аэропортов, {summary['countries']} стран. С уходом: {target_true:,} ({summary['ga_percent']:.4f}%).",'',
           f'Период посадок UTC: {first_time} – {last_time}.','',
           f"Исходник сохранен: SHA256 `{source_hash}` совпадает до и после аудита. Строк удалено: 0. Значений исправлено: 0.",'',
           '## Диагностические флаги','','Флаги обозначают случаи для разбора, а не доказанные ошибки и не правила исключения. Пропуски отдельно учитываются в missingness.csv; пустые строки и null не смешиваются.','',
           '| Причина | Кол-во записей | С уходом |','| --- | ---: | ---: |']
    for row in flags_table.loc[flags_table['rows']>0].itertuples(index=False):
        lines.append(f'| `{row.reason}` | {int(row.rows):,} | {int(row.go_arounds):,} |')
    lines+=['','## Повторы и временное покрытие','',
            f"Повторяющихся flight_id: {summary['repeated_flight_ids']:,}, дополнительных строк: {summary['excess_flight_id_rows']:,}. Повторяющихся составных ключей: {len(duplicate_keys):,}. Полных дополнительных дублей без служебного индекса: {business_duplicates:,}.",'',
            'Составной ключ – landing_time + icao24 + airport. Совпадение не дает разрешения удалить запись. Полные дубли проверены по всем содержательным полям, включая списки, среди кандидатов с повтором составного ключа.','',
            'Описание Zenodo указывает январь 2019 – июнь 2023; фактический максимум посадок – 30 мая 2023. Дни без записей не объявляются отсутствием полетов или отказом приемника. Наличие строк в месяце не доказывает полноту покрытия.','',
            '## Ограничения','',
            'Набор содержит метаданные и готовую разметку, но не полные траектории, погоду, высоты и скорости самолета. Причины ухода, назначение полета и соблюдение процедур по этой таблице не подтверждаются.','',
            'GoA и число попыток согласованы по логическому правилу, но это не независимая проверка метки. Больше двух попыток не доказывает учебный полет.','',
            'C40_CROSS_TIME и признаки входа в 40-мильную область требуют проверки полноты и положения относительно первой попытки. Признаки итоговых заходов и посадки нельзя использовать для раннего прогноза.','',
            'Очистка и отбор аэропортов еще не выполнялись. Критерии прежней выборки не перенесены автоматически. Следующий этап – разбор групп флагов и согласование обработки.','']
    (REPORTS/'database_audit.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps({k:v for k,v in summary.items() if k!='schema'},ensure_ascii=False,indent=2),flush=True)
    return summary


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf8')
    profile_source()
