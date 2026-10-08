"""Восстановление отсутствующей классификации по прямому справочному коду."""
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from audit_aircraft_types import INPUT, REPORT
from audit_feature_availability import ROOT, digest

REFERENCE = ROOT/'docs/sources/files/aircraft_code_review/icao_aircraft_types.json'
OUTPUT = ROOT/'europe/data/analytical/05_aircraft_classification_enriched.parquet'
SOURCE = 'https://www.icao.int/operational-safety/doc-8643-aircraft-type-designators/search'
ADDED = ['aircraft_description_restored','aircraft_restoration_status','aircraft_restoration_source','aircraft_restoration_checked_date']


def main():
    # Проверяем ранее сохраненный официальный снимок и актуальную выборку
    manifest=json.loads((ROOT/'europe/reports/preparation/aircraft_reference_sources.json').read_text(encoding='utf8'))
    entry=next(x for x in manifest['sources'] if x['name']=='icao_aircraft_types.json')
    ref_hash=digest(REFERENCE)
    assert ref_hash==entry['sha256']
    before=digest(INPUT)
    previous=json.loads((REPORT/'05_01_audit.json').read_text(encoding='utf8'))
    assert before==previous['input_sha256']
    reference=pd.DataFrame(json.loads(REFERENCE.read_text(encoding='utf8')))
    reference=reference[~reference.ShowInPart3Only.fillna(False)]
    variants=reference.groupby('Designator').Description.agg(lambda values:sorted(set(values.dropna())))
    mapping={k:v[0] for k,v in variants.items() if len(v)==1 and v[0]}
    if OUTPUT.exists(): raise FileExistsError('Выход уже существует; перезапись запрещена')
    reader=pq.ParquetFile(INPUT)
    schema=reader.schema_arrow
    out_schema=schema
    for field in ADDED: out_schema=out_schema.append(pa.field(field,pa.string()))
    temporary=OUTPUT.with_suffix('.parquet.part')
    pieces=[]
    rows=positive=restored=restored_positive=0
    with pq.ParquetWriter(temporary,out_schema,compression='gzip',compression_level=1) as writer:
        for batch in reader.iter_batches(batch_size=100_000):
            table=pa.Table.from_batches([batch])
            frame=table.select(['typecode','icaoaircrafttype','GoA']).to_pandas(ignore_metadata=True)
            # Меняем регистр только поискового ключа, без неявных псевдонимов и подстрок
            keys=frame.typecode.str.upper()
            description=keys.map(mapping)
            needed=frame.icaoaircrafttype.isna()
            mask=needed & frame.typecode.notna() & description.notna()
            status=pd.Series('source_classification_present',index=frame.index)
            status[needed & frame.typecode.isna()]='both_type_fields_missing'
            status[needed & frame.typecode.notna()]='no_unique_direct_reference_match'
            status[mask]='restored_from_unique_designator'
            derived=description.where(mask,None)
            source=pd.Series(None,index=frame.index,dtype=object)
            checked=source.copy()
            source[mask]=SOURCE
            checked[mask]=manifest['checked_date']
            for name,values in zip(ADDED,[derived,status,source,checked]):
                table=table.append_column(name,pa.array(values,type=pa.string(),from_pandas=True))
            writer.write_table(table)
            summary=frame.copy()
            summary['restoration_status']=status
            summary['description_restored']=derived
            pieces.append(summary.groupby(['typecode','restoration_status','description_restored'],dropna=False).agg(rows=('GoA','size'),go_arounds=('GoA','sum')).reset_index())
            rows+=len(table)
            positive+=int(frame.GoA.sum())
            restored+=int(mask.sum())
            restored_positive+=int(frame.loc[mask,'GoA'].sum())
    assert rows==previous['rows_checked'] and positive==previous['go_arounds']
    # Повторно читаем все строки и сверяем исходные поля без изменений
    candidate=pq.ParquetFile(temporary)
    assert candidate.schema_arrow.equals(out_schema)
    source_batches=reader.iter_batches(batch_size=100_000)
    checked_rows=0
    for batch in candidate.iter_batches(batch_size=100_000):
        original=next(source_batches)
        actual=pa.Table.from_batches([batch])
        assert actual.select(schema.names).equals(pa.Table.from_batches([original]))
        # Проверяем соответствие каждого добавленного описания справочнику
        f=actual.select(['typecode','icaoaircrafttype']+ADDED).to_pandas(ignore_metadata=True)
        populated=f.aircraft_description_restored.notna()
        assert f.loc[populated,'icaoaircrafttype'].isna().all()
        assert f.loc[populated,'typecode'].str.upper().map(mapping).eq(f.loc[populated,'aircraft_description_restored']).all()
        assert populated.eq(f.aircraft_restoration_status.eq('restored_from_unique_designator')).all()
        checked_rows+=len(actual)
    assert next(source_batches,None) is None and checked_rows==rows
    candidate.close()
    assert digest(INPUT)==before and digest(REFERENCE)==ref_hash
    temporary.replace(OUTPUT)
    details=pd.concat(pieces).groupby(['typecode','restoration_status','description_restored'],dropna=False)[['rows','go_arounds']].sum().reset_index()
    assert details.rows.sum()==rows and details.go_arounds.sum()==positive
    details.to_csv(REPORT/'05_03_restoration_by_code.csv',index=False)
    totals=details.groupby('restoration_status')[['rows','go_arounds']].sum().reset_index()
    totals.to_csv(REPORT/'05_03_restoration_summary.csv',index=False)
    result=dict(input=str(INPUT.relative_to(ROOT)),input_sha256=before,input_unchanged=True,
        output=str(OUTPUT.relative_to(ROOT)),output_sha256=digest(OUTPUT),rows_before=rows,rows_after=rows,
        go_arounds_before=positive,go_arounds_after=positive,rows_deleted=0,source_values_changed=0,
        classifications_restored=restored,positive_classifications_restored=restored_positive,
        classification_missing_after=281738-restored,positive_classification_missing_after=898-restored_positive,
        added_fields=ADDED,original_fields_exact_comparison_passed=True,
        rule='Missing icaoaircrafttype only; uppercase typecode exactly matches an ordinary ICAO Designator with one Description',
        reference_url=SOURCE,reference_endpoint=entry['url'],reference_sha256=ref_hash,
        reference_checked_date=manifest['checked_date'],reference_row_count=entry['reference_rows'],
        historical_reference_validity_verified=False,aircraft_identity_verified=False,
        training_feature_availability_verified=False,typecode_restored=False)
    (REPORT/'05_03_restoration_applied.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return totals,details,result
