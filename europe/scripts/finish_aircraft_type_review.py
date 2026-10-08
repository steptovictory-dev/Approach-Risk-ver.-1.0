"""Ограниченный завершающий разбор семейств и подтвержденных подтипов ВС."""
import json
import re
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from audit_aircraft_types import REPORT
from audit_feature_availability import ROOT, digest
from restore_aircraft_classification import REFERENCE, SOURCE, ADDED

INPUT=ROOT/'europe/data/analytical/05_aircraft_model_names_enriched.parquet'
OUTPUT=ROOT/'europe/data/analytical/05_aircraft_types_reviewed.parquet'
MODEL_TOKENS=['A320-251N','A321-231','A320-214','A330','A350','A380','B777','G500','G400','G600']
EASA_SOURCE='https://ad.easa.europa.eu/ad/2024-0231/print'


def main():
    # Проверяем происхождение обоих входов по сохраненным контрольным суммам
    prior=json.loads((REPORT/'05_04_applied.json').read_text(encoding='utf8'))
    before=digest(INPUT)
    ref_hash=digest(REFERENCE)
    assert before==prior['output_sha256'] and ref_hash==prior['reference_sha256']
    reference=pd.DataFrame(json.loads(REFERENCE.read_text(encoding='utf8')))
    reference=reference[~reference.ShowInPart3Only.fillna(False)]
    variants=reference.groupby('Designator').Description.agg(lambda x:sorted(set(x.dropna())))
    direct={k:v[0] for k,v in variants.items() if len(v)==1 and v[0]}
    # Реестр ограничен разобранными подтипами и явно указанными семействами
    selectors = {
        'A320-251N': reference.Designator.isin(['A320','A20N']),
        'A321-231': reference.Designator.eq('A321'),
        'A320-214': reference.Designator.eq('A320'),
        'A330': reference.ModelFullName.str.startswith('A-330-',na=False),
        'A350': reference.ModelFullName.str.startswith('A-350-',na=False),
        'A380': reference.ModelFullName.str.startswith('A-380-',na=False),
        'B777': reference.ModelFullName.str.match(r'777-(200|300|8|9)',na=False),
    }
    for token in ['G500','G400','G600']:
        selectors[token]=reference.ModelFullName.str.contains(r'(?<![A-Z0-9])'+token+r'(?![A-Z0-9])',case=False,regex=True,na=False)
    model_map={}
    evidence=[]
    for token,mask in selectors.items():
        selected=reference[mask]
        descriptions=selected.Description.dropna().unique()
        assert len(selected)>0 and len(descriptions)==1
        # Разные designators допустимы только при едином широком описании
        model_map[token]=('|'.join(sorted(selected.Designator.unique())),descriptions[0])
        for record in selected[['ModelFullName','Designator','Description']].drop_duplicates().to_dict('records'):
            evidence.append(dict(source_token=token,reference_url=SOURCE,
                subtype_source=EASA_SOURCE if token in ['A320-251N','A321-231','A320-214'] else None,
                matching_level='classification_only',**record))
    pd.DataFrame(evidence).to_csv(REPORT/'05_05_reference_evidence.csv',index=False)
    reader=pq.ParquetFile(INPUT)
    if OUTPUT.exists(): raise FileExistsError('Выход существует; перезапись запрещена')
    schema=reader.schema_arrow
    temporary=OUTPUT.with_suffix('.parquet.part')
    pieces=[]
    rows=positives=changed=changed_positive=0
    with pq.ParquetWriter(temporary,schema,compression='gzip',compression_level=1) as writer:
        for batch in reader.iter_batches(batch_size=100_000):
            table=pa.Table.from_batches([batch])
            frame=table.select(['typecode','icaoaircrafttype','GoA']+ADDED).to_pandas(ignore_metadata=True)
            key=frame.typecode.str.strip().str.upper()
            needed=frame.icaoaircrafttype.isna() & frame.aircraft_description_restored.isna() & frame.typecode.notna()
            descriptions=key.map(direct)
            trimmed=needed & descriptions.notna()
            models=needed & ~trimmed & key.isin(model_map)
            for token,(_,description) in model_map.items(): descriptions.loc[models & key.eq(token)]=description
            mask=trimmed|models
            frame.loc[mask,'aircraft_description_restored']=descriptions[mask]
            frame.loc[trimmed,'aircraft_restoration_status']='restored_from_trimmed_designator'
            frame.loc[models,'aircraft_restoration_status']='restored_from_reviewed_family_description'
            frame.loc[mask,'aircraft_restoration_source']=SOURCE
            frame.loc[mask,'aircraft_restoration_checked_date']=prior['reference_checked_date']
            for field in ADDED:
                table=table.set_column(schema.get_field_index(field),schema.field(field),pa.array(frame[field],type=pa.string(),from_pandas=True))
            writer.write_table(table)
            if mask.any():
                selected=frame[mask].copy()
                selected['lookup_key']=key[mask]
                pieces.append(selected.groupby(['typecode','lookup_key','aircraft_description_restored','aircraft_restoration_status'],dropna=False).agg(rows=('GoA','size'),go_arounds=('GoA','sum')).reset_index())
            rows+=len(table)
            positives+=int(frame.GoA.sum())
            changed+=int(mask.sum())
            changed_positive+=int(frame.loc[mask,'GoA'].sum())
    # Проверяем все неизменяемые поля и точное правило каждой новой классификации
    candidate=pq.ParquetFile(temporary)
    assert candidate.schema_arrow.equals(schema)
    originals=reader.iter_batches(batch_size=100_000)
    verified=0
    original_fields=[f for f in schema.names if f not in ADDED]
    for batch in candidate.iter_batches(batch_size=100_000):
        actual=pa.Table.from_batches([batch])
        source=pa.Table.from_batches([next(originals)])
        assert actual.select(original_fields).equals(source.select(original_fields))
        old=source.select(['typecode','icaoaircrafttype']+ADDED).to_pandas(ignore_metadata=True)
        new=actual.select(['typecode','icaoaircrafttype']+ADDED).to_pandas(ignore_metadata=True)
        key=old.typecode.str.strip().str.upper()
        eligible=old.icaoaircrafttype.isna() & old.aircraft_description_restored.isna() & old.typecode.notna()
        allowed=eligible & (key.isin(direct)|key.isin(model_map))
        assert actual.select(ADDED).filter(pa.array(~allowed)).equals(source.select(ADDED).filter(pa.array(~allowed)))
        expected=key.map(direct)
        for token,(_,description) in model_map.items(): expected.loc[key.eq(token)]=description
        assert new.loc[allowed,'aircraft_description_restored'].eq(expected[allowed]).all()
        verified+=len(actual)
    assert next(originals,None) is None and verified==rows==prior['rows_after']
    assert positives==prior['go_arounds_after']
    candidate.close()
    assert digest(INPUT)==before and digest(REFERENCE)==ref_hash
    temporary.replace(OUTPUT)
    detail=pd.concat(pieces).groupby(['typecode','lookup_key','aircraft_description_restored','aircraft_restoration_status'],dropna=False)[['rows','go_arounds']].sum().reset_index()
    assert int(detail.rows.sum())==changed and int(detail.go_arounds.sum())==changed_positive
    detail.to_csv(REPORT/'05_05_applied_matches.csv',index=False)
    result=dict(input=str(INPUT.relative_to(ROOT)),input_sha256=before,input_unchanged=True,
        output=str(OUTPUT.relative_to(ROOT)),output_sha256=digest(OUTPUT),rows_after=rows,go_arounds_after=positives,
        classifications_added=changed,positive_classifications_added=changed_positive,
        classifications_restored_total=prior['classifications_restored_total']+changed,
        positive_classifications_restored_total=prior['positive_classifications_restored_total']+changed_positive,
        missing_classification_after=prior['missing_classification_after']-changed,
        positive_missing_classification_after=prior['positive_missing_classification_after']-changed_positive,
        rows_deleted=0,source_values_changed=0,physical_columns=len(schema),
        all_original_fields_verified=True,prior_derived_values_outside_rule_verified=True,
        reference_sha256=ref_hash,reference_url=SOURCE,reference_checked_date=prior['reference_checked_date'],
        model_tokens=MODEL_TOKENS,rule='Reviewed subtype or explicit family with one ICAO Description; specific designator not inferred',
        historical_identity_verified=False,feature_availability_verified=False,
        subtype_source=EASA_SOURCE, source_checked_date='2026-10-08',
        unresolved_records_retained=True, typecode_not_reconstructed=True, bounded_aircraft_type_review_complete=True)
    remaining=residual_report()
    assert int(remaining.go_arounds.sum())==911-result['positive_classifications_restored_total']
    result['remaining_missing_positive']=int(remaining.go_arounds.sum())
    result['remaining_missing_rows']=int(remaining.rows.sum())
    (REPORT/'05_05_applied.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return detail,result


def residual_report():
    # Закрываем остаток без удаления и без замены пропусков догадками
    pieces=[]
    for batch in pq.ParquetFile(OUTPUT).iter_batches(batch_size=100_000,
            columns=['typecode','icaoaircrafttype','aircraft_description_restored','GoA']):
        frame=pa.Table.from_batches([batch]).to_pandas(ignore_metadata=True)
        unresolved=frame.icaoaircrafttype.isna() & frame.aircraft_description_restored.isna()
        frame['remaining_issue']='classification_unresolved'
        frame.loc[unresolved & frame.typecode.isna(),'remaining_issue']='both_type_fields_missing'
        frame.loc[~unresolved & frame.typecode.isna(),'remaining_issue']='typecode_only_missing'
        selected=frame[unresolved | frame.typecode.isna()].copy()
        selected['lookup_key']=selected.typecode.str.strip().str.upper()
        pieces.append(selected.groupby(['lookup_key','remaining_issue'],dropna=False).agg(
            rows=('GoA','size'),go_arounds=('GoA','sum')).reset_index())
    remaining=pd.concat(pieces).groupby(['lookup_key','remaining_issue'],dropna=False)[['rows','go_arounds']].sum().reset_index()
    remaining['decision']='retain_unknown; no further historical aircraft lookup in this bounded review'
    remaining.loc[remaining.remaining_issue.eq('typecode_only_missing'),'decision']='retain; broad classification cannot uniquely determine typecode'
    remaining.to_csv(REPORT/'05_05_unresolved_remaining.csv',index=False)
    return remaining
