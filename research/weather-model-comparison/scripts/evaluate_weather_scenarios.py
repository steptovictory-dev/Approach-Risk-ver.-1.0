"""Fixed binary GoA logistic classifier: delay/COR sensitivity, no fit."""
import hashlib,json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from comparison_pipeline import BASE,prepare_features,metrics,threshold_metrics,calibration_table,feature_registry

PROJECT=BASE.parents[1]; W=PROJECT/'europe/weather'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    print('Method: binary classification, fixed logistic L2 classifier; no training or threshold tuning.',flush=True)
    training=json.loads((BASE/'reports/04_training_manifest.json').read_text(encoding='utf8'))
    application=json.loads((W/'reports/23_final_weather_application.json').read_text(encoding='utf8'))
    table_manifest=json.loads((BASE/'reports/02_model_table_manifest.json').read_text(encoding='utf8'))
    files={x['file']:x for x in table_manifest['files']}
    modelpath=BASE/'data/primary_models_v1/weather.joblib'
    assert sha(modelpath)==next(x['sha256'] for x in training['files'] if x['file']=='weather.joblib')
    model=joblib.load(modelpath)
    primary=pd.read_parquet(BASE/'data/primary_models_v1/weather_test_predictions.parquet').set_index('source_row')
    masks=pd.read_csv(W/'reports/17_numeric_quality_preview.csv',keep_default_na=False).set_index('metar_id').proposed_mask_fields
    registry=feature_registry();fields=list(registry.loc[registry.block.eq('weather'),'field'])
    kinds=registry.set_index('field').kind
    dest=BASE/'data/weather_sensitivity_v1';dest.mkdir(parents=True,exist_ok=True)
    scenarios=[('primary_reproduction',15,True),('delay0',0,True),('delay30',30,True),('no_cor',15,False)]
    chunks={s:[] for s,_,_ in scenarios};checks=[]
    for entry in application['files']:
        name=entry['name']
        frame=pd.read_parquet(BASE/'data/model_table_v1'/name)
        frame=frame[frame.temporal_split.eq('test')].sort_values('C40_CROSS_TIME').reset_index(drop=True)
        if frame.empty:continue
        assert sha(BASE/'data/model_table_v1'/name)==files[name]['sha256']
        path=W/'data/full_cloud_v2/reports'/name.replace('.parquet','_reports.parquet')
        assert sha(path)==entry['source_reports_sha256']
        reports=pd.read_parquet(path)
        assert reports.metar_id.is_unique
        reports['ambiguous']=reports.duplicated('metar_utc',keep=False)
        reports['numeric_mask_fields']=reports.metar_id.map(masks).fillna('')
        for scenario,delay,include_cor in scenarios:
            r=reports if include_cor else reports[~reports.correction_reported.eq(True)]
            r=r.sort_values('metar_utc').drop_duplicates('metar_utc').copy()
            r['assumed_available']=r.metar_utc+pd.Timedelta(minutes=delay)
            linked=pd.merge_asof(frame[['source_row','C40_CROSS_TIME']],r,left_on='C40_CROSS_TIME',right_on='assumed_available',direction='backward',allow_exact_matches=False)
            known=linked.metar_utc.notna()
            assert linked.loc[known,'assumed_available'].lt(linked.loc[known,'C40_CROSS_TIME']).all()
            age=(linked.C40_CROSS_TIME-linked.metar_utc).dt.total_seconds()/60
            usable=known&age.le(60)&~linked.ambiguous.eq(True)&(linked.decoded_candidate.eq(True)|linked.core_validated.eq(True))
            x=frame.copy();changed=np.zeros(len(frame),dtype=bool)
            for field in fields:
                raw=field.removesuffix('_model')
                masked=linked.numeric_mask_fields.fillna('').str.split(';').map(lambda a:raw in a)
                value=linked[raw].mask(~usable|masked)
                if kinds[field]=='numeric':value=pd.to_numeric(value,errors='raise').astype('Float64')
                elif kinds[field]=='boolean':value=value.astype('boolean')
                else:value=value.astype('string')
                equal=value.eq(frame[field]).fillna(False)|(value.isna()&frame[field].isna())
                changed|=~equal.to_numpy(dtype=bool)
                x[field]=value
            if scenario=='primary_reproduction':assert not changed.any(),f'Failed primary reconstruction: {name}'
            with threadpool_limits(limits=6):p=model.predict_proba(prepare_features(x,weather=True))[:,1]
            original=primary.loc[frame.source_row,'probability'].to_numpy()
            if scenario=='primary_reproduction':np.testing.assert_allclose(p,original,rtol=1e-12,atol=1e-12)
            audit=frame[['source_row','GoA','C40_CROSS_TIME','airport']].copy()
            audit['probability']=p;audit['weather_usable']=usable.to_numpy();audit['information_changed']=changed
            audit['probability_changed']=np.abs(p-original)>1e-12
            chunks[scenario].append(audit)
        assert sha(path)==entry['source_reports_sha256']
        checks.append({'file':name,'rows':len(frame),'reports_sha256':entry['source_reports_sha256']})
        print('Scenarios verified',name,len(frame),flush=True)
    summaries=[];thresholds=[]
    for scenario,_,_ in scenarios:
        f=pd.concat(chunks[scenario],ignore_index=True).sort_values('source_row').reset_index(drop=True)
        assert len(f)==546577 and f.GoA.sum()==1739 and f.source_row.is_unique
        assert np.array_equal(f.source_row,primary.index.sort_values())
        f.to_parquet(dest/f'{scenario}_predictions.parquet',index=False)
        summaries.append({'scenario':scenario,**metrics(f.GoA,f.probability),'weather_usable':int(f.weather_usable.sum()),'information_changed':int(f.information_changed.sum()),'probability_changed':int(f.probability_changed.sum())})
        for limit,t in training['thresholds']['weather'].items():
            thresholds.append({'scenario':scenario,'target_validation_FPR':float(limit),**threshold_metrics(f.GoA,f.probability,t)})
        calibration_table(f.GoA,f.probability).to_csv(BASE/f'reports/06_{scenario}_calibration.csv',index=False)
    pd.DataFrame(summaries).to_csv(BASE/'reports/06_weather_scenario_summary.csv',index=False)
    pd.DataFrame(thresholds).to_csv(BASE/'reports/06_weather_scenario_thresholds.csv',index=False)
    manifest={'verified':True,'fit_executed':False,'thresholds_retuned':False,'model_sha256':sha(modelpath),'primary_reproduced':True,'source_checks':checks,
              'files':[{'file':p.name,'sha256':sha(p)} for p in dest.glob('*.parquet')]}
    (BASE/'reports/06_weather_scenario_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)
if __name__=='__main__':main()
