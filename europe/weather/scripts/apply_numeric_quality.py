"""Отдельная копия числовых признаков; исходные файлы не изменяются."""
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from audit_pilot_input import BASE, INPUT, digest

R = BASE / 'reports'
DEST = BASE / 'data/numeric_quality_v1'
FIELDS = ['wind_speed_kt','wind_gust_kt','visibility_m','cloud_ceiling_ft',
          'lowest_known_cloud_base_ft','temperature_c','dewpoint_c','qnh_hpa']
IDENTITY = ['source_row','airport','GoA','C40_CROSS_TIME','metar_id','metar','metar_utc',
            'age_minutes','status','decoded_candidate','core_validated',
            'correction_reported','availability_verified']

def main():
    decision=json.loads((R/'17_numeric_quality_protocol.json').read_text(encoding='utf8'))
    assert decision['status']=='accepted' and decision['user_decision']=='accept_proposed_field_masks'
    registry=pd.read_csv(R/'17_numeric_quality_preview.csv').set_index('metar_id')
    assert registry.index.is_unique and len(registry)==18
    integrity=json.loads((R/'12_full_integrity.json').read_text(encoding='utf8'))
    hashes={row['name']:row['sha256'] for row in integrity['feature_files']}
    assert digest(INPUT)==decision['input_sha256']==integrity['input_sha256']
    progress=json.loads((R/'12_full_progress.json').read_text(encoding='utf8'))
    seen=np.zeros(integrity['rows'],dtype=bool)
    DEST.mkdir(parents=True,exist_ok=True)
    files, journal, coverage = [], [], []
    labels=0
    for job in progress['jobs']:
        name=f"{job['airport']}_{job['year']}_features.parquet"
        source=BASE/'data/full/results'/name
        assert digest(source)==hashes[name]
        frame=pd.read_parquet(source,columns=IDENTITY+FIELDS)
        ids=frame.source_row.to_numpy()
        assert frame.source_row.is_unique and not seen[ids].any()
        seen[ids]=True
        labels+=int(frame.GoA.sum())
        frame['numeric_quality_reasons']=frame.metar_id.map(registry.numeric_quality_reasons).fillna('not_flagged_by_current_checks')
        frame['numeric_mask_fields']=frame.metar_id.map(registry.proposed_mask_fields).fillna('')
        frame['numeric_quality_protocol_version']='numeric-quality-v1'
        any_mask=frame.numeric_mask_fields.ne('')
        for field in FIELDS:
            mask=frame.numeric_mask_fields.str.split(';').apply(lambda values:field in values)
            frame[field+'_model']=frame[field].mask(mask)
            part=frame.loc[mask,['source_row','airport','GoA','metar_id','status','numeric_quality_reasons',field]].copy()
            part=part.rename(columns={field:'original_value'})
            part['field']=field
            part['model_value']=np.nan
            part['original_value_present']=part.original_value.notna()
            journal.append(part)
        output=DEST/name
        frame.to_parquet(output,index=False)
        saved=pd.read_parquet(output)
        # Проверяется сохраненный файл, включая исходные поля, порядок и кратность.
        pd.testing.assert_frame_equal(saved[IDENTITY+FIELDS],pd.read_parquet(source,columns=IDENTITY+FIELDS))
        for field in FIELDS:
            mask=saved.numeric_mask_fields.str.split(';').apply(lambda values:field in values)
            assert saved.loc[mask,field+'_model'].isna().all()
            pd.testing.assert_series_equal(saved.loc[~mask,field+'_model'],saved.loc[~mask,field],check_names=False)
        assert digest(source)==hashes[name]
        for label,part in frame.groupby('GoA'):
            affected=part.numeric_mask_fields.ne('')
            coverage.append(dict(airport=job['airport'],year=job['year'],GoA=bool(label),
                rows=len(part),flagged_rows=int(affected.sum()),
                flagged_rows_matched_60min=int((affected&part.status.eq('matched')).sum())))
        files.append(dict(name=name,source_sha256=hashes[name],output_sha256=digest(output),rows=len(saved),
            flagged_rows=int(any_mask.sum())))
        print('verified copy',job['airport'],job['year'],len(saved),flush=True)
    assert seen.all() and labels==integrity['go_arounds']
    assert digest(INPUT)==decision['input_sha256']
    changes=pd.concat(journal,ignore_index=True)
    assert not changes.duplicated(['source_row','field']).any()
    changes.to_csv(R/'18_numeric_mask_journal.csv',index=False)
    counts=pd.DataFrame(coverage)
    counts.to_csv(R/'18_numeric_mask_coverage.csv',index=False)
    baseline=changes.loc[changes.status.eq('matched')]
    assert baseline.source_row.nunique()==44 and baseline.GoA.sum()==0
    field_counts=changes.groupby('field').agg(masked_cells=('source_row','size'),
        previously_present=('original_value_present','sum'),affected_rows=('source_row','nunique')).reset_index()
    field_counts.to_csv(R/'18_numeric_mask_fields.csv',index=False)
    report=dict(version='numeric-quality-v1',decision='accepted',applied_utc=datetime.now(timezone.utc).isoformat(),
        input_sha256=decision['input_sha256'],registry_sha256=digest(R/'17_numeric_quality_preview.csv'),
        implementation_sha256=digest(BASE/'scripts/apply_numeric_quality.py'),
        rows=integrity['rows'],go_arounds=labels,all_source_rows_present_once=True,
        raw_fields_unchanged=True,source_files_unchanged=True,unmasked_model_values_unchanged=True,
        rows_deleted=0,training_executed=False,numeric_copy_only=True,
        matched_60min_flagged_rows=int(baseline.source_row.nunique()),
        all_status_flagged_rows=int(changes.source_row.nunique()),masked_cells=len(changes),files=files)
    (R/'18_numeric_mask_application.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print({key:value for key,value in report.items() if key!='files'},flush=True)

if __name__=='__main__':main()
