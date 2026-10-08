"""Дополнительное восстановление по очищенному ключу и явному имени модели ICAO."""
import json
import re
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from audit_aircraft_types import REPORT
from audit_feature_availability import ROOT, digest
from restore_aircraft_classification import REFERENCE, SOURCE, ADDED

INPUT=ROOT/'europe/data/analytical/05_aircraft_classification_enriched.parquet'
OUTPUT=ROOT/'europe/data/analytical/05_aircraft_model_names_enriched.parquet'
MODEL_TOKENS=['G550','G650','G450','G350','G100']


def main():
    # Проверяем происхождение обоих входов по сохраненным контрольным суммам
    prior=json.loads((REPORT/'05_03_restoration_applied.json').read_text(encoding='utf8'))
    before=digest(INPUT)
    ref_hash=digest(REFERENCE)
    assert before==prior['output_sha256'] and ref_hash==prior['reference_sha256']
    reference=pd.DataFrame(json.loads(REFERENCE.read_text(encoding='utf8')))
    reference=reference[~reference.ShowInPart3Only.fillna(False)]
    variants=reference.groupby('Designator').Description.agg(lambda x:sorted(set(x.dropna())))
    direct={k:v[0] for k,v in variants.items() if len(v)==1 and v[0]}
    # Проверяем только заранее разобранные буквальные обозначения моделей
    model_map={}
    evidence=[]
    for token in MODEL_TOKENS:
        selected=reference[reference.ModelFullName.str.contains(r'(?<![A-Z0-9])'+re.escape(token)+r'(?![A-Z0-9])',case=False,regex=True,na=False)]
        pairs=selected[['Designator','Description']].drop_duplicates()
        assert len(pairs)==1 and pairs.iloc[0].notna().all()
        model_map[token]=(pairs.iloc[0].Designator,pairs.iloc[0].Description)
        for record in selected[['ModelFullName','Designator','Description']].drop_duplicates().to_dict('records'):
            evidence.append(dict(source_token=token,reference_url=SOURCE,**record))
    pd.DataFrame(evidence).to_csv(REPORT/'05_04_model_reference_evidence.csv',index=False)
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
            frame.loc[models,'aircraft_restoration_status']='restored_from_explicit_model_name'
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
    detail.to_csv(REPORT/'05_04_applied_matches.csv',index=False)
    result=dict(input=str(INPUT.relative_to(ROOT)),input_sha256=before,input_unchanged=True,
        output=str(OUTPUT.relative_to(ROOT)),output_sha256=digest(OUTPUT),rows_after=rows,go_arounds_after=positives,
        classifications_added=changed,positive_classifications_added=changed_positive,
        classifications_restored_total=prior['classifications_restored']+changed,
        positive_classifications_restored_total=prior['positive_classifications_restored']+changed_positive,
        missing_classification_after=prior['classification_missing_after']-changed,
        positive_missing_classification_after=prior['positive_classification_missing_after']-changed_positive,
        rows_deleted=0,source_values_changed=0,physical_columns=len(schema),
        all_original_fields_verified=True,prior_derived_values_outside_rule_verified=True,
        reference_sha256=ref_hash,reference_url=SOURCE,reference_checked_date=prior['reference_checked_date'],
        model_tokens=MODEL_TOKENS,rule='Trim and uppercase search key; exact ICAO Designator or whitelisted explicit model token with one Designator and Description',
        historical_identity_verified=False,feature_availability_verified=False,
        deferred_airbus_certificate_url='https://www.easa.europa.eu/en/downloads/16507/en',
        deferred_airbus_certificate_status='Full PDF unavailable through web tool: HTTP 429; no subtype mapping applied')
    (REPORT/'05_04_applied.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return detail,result
