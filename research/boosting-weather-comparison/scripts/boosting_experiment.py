"""Binary classification protocol: train candidates, select validation, then test."""
from pathlib import Path
import gc,hashlib,json,time,importlib.metadata
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from approach_risk.paths import project_root
from approach_risk.data.registry import dataset,sha
from approach_risk.models.boosting import pool,classifier
from approach_risk.evaluation.classification import metrics,choose_thresholds,evaluate_pair,calibration_table

ROOT=project_root();BASE=ROOT/'research/boosting-weather-comparison'
CONFIG=json.loads((ROOT/'configs/boosting_weather.json').read_text())
DEST=ROOT/CONFIG['artifacts'];R=BASE/'reports'
DEST.mkdir(parents=True,exist_ok=True)

def write_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(path)

class Experiment:
    def __init__(self):
        self.ds=dataset(CONFIG['dataset']);self.input_manifest=self.ds.manifest()
        self.fingerprint=hashlib.sha256(((ROOT/'configs/boosting_weather.json').read_text()+
            (BASE/'protocol.md').read_text()+sha(Path(__file__))+self.ds.manifest_sha256+
            sha(ROOT/'src/approach_risk/models/boosting.py')+sha(ROOT/'src/approach_risk/features/c40.py')).encode()).hexdigest()
        self.candidates=[('fixed',6,.05)]+[(f'd{d}_lr{lr}',d,lr) for d in [4,8] for lr in [.03,.1]]
        self.rows=[];self.selected={}
        write_json(R/'01_environment.json',{'versions':{x:importlib.metadata.version(x) for x in ['catboost','numpy','pandas','pyarrow','scikit-learn','nbformat','nbclient']},'fingerprint':self.fingerprint,'task':'supervised binary classification','family':'CatBoostClassifier','dataset_manifest_sha256':self.ds.manifest_sha256})

    def train_variant(self,name,weather):
        # Method is printed immediately before every fit; only train enters fitting.
        train=pd.concat(list(self.ds.iter_split('train')),ignore_index=True).sort_values(['C40_CROSS_TIME','source_row']).reset_index(drop=True)
        validation=pd.concat(list(self.ds.iter_split('validation')),ignore_index=True).sort_values('source_row').reset_index(drop=True)
        assert len(train)==4165060 and train.GoA.sum()==11626
        assert len(validation)==1619848 and validation.GoA.sum()==4984
        assert train.C40_CROSS_TIME.max()<validation.C40_CROSS_TIME.min()
        train_pool=pool(train,weather,True);valid_pool=pool(validation,weather)
        print(f'{name}: train={len(train)}, validation={len(validation)}, predictors={train_pool.num_col()}',flush=True)
        for candidate,depth,lr in self.candidates:
            directory=DEST/name/candidate;directory.mkdir(parents=True,exist_ok=True)
            marker=directory/'complete.json'
            if marker.exists():
                row=json.loads(marker.read_text(encoding='utf8'))
                if row['fingerprint']!=self.fingerprint or sha(directory/'model.cbm')!=row['model_sha256']:raise ValueError('Resume provenance mismatch')
                print('Verified completed candidate:',name,candidate,flush=True)
            else:
                parameters={**CONFIG['parameters'],'depth':depth,'learning_rate':lr}
                print(f'Before fit: supervised binary classification GoA, CatBoost Logloss; {name}/{candidate}; train only, no eval_set.',flush=True)
                write_json(R/'01_run_status.json',{'status':'training','variant':name,'candidate':candidate,'fingerprint':self.fingerprint})
                model=classifier(parameters,directory/'logs');start=time.time();model.fit(train_pool)
                assert model.tree_count_==500
                model.save_model(str(directory/'model.cbm'))
                p=model.predict_proba(valid_pool)[:,1]
                audit=validation[['source_row','GoA','C40_CROSS_TIME','airport','weather_core_usable']].copy();audit['probability']=p
                audit.to_parquet(directory/'validation_predictions.parquet',index=False)
                pd.DataFrame({'iteration':np.arange(1,501),'train_Logloss':model.get_evals_result()['learn']['Logloss']}).to_csv(R/f'01_{name}_{candidate}_learning.csv',index=False)
                write_json(directory/'actual_parameters.json',model.get_all_params())
                row={'variant':name,'candidate':candidate,'depth':depth,'learning_rate':lr,'iterations':model.tree_count_,
                     'seconds':time.time()-start,'fingerprint':self.fingerprint,'model_sha256':sha(directory/'model.cbm'),
                     'validation_predictions_sha256':sha(directory/'validation_predictions.parquet'),**metrics(validation.GoA,p)}
                write_json(marker,row);del model,p,audit
            self.rows.append(row);pd.DataFrame(self.rows).to_csv(R/'01_validation_candidates.csv',index=False)
            print('Validation:',row['variant'],row['candidate'],row['AP'],row['ROC_AUC'],flush=True)
        del train_pool,valid_pool,train,validation;gc.collect()
        return pd.DataFrame([x for x in self.rows if x['variant']==name])

    def select(self):
        assert len(self.rows)==10
        for name in ['baseline','weather']:
            rows=pd.DataFrame([x for x in self.rows if x['variant']==name])
            best=rows.sort_values(['AP','depth','learning_rate'],ascending=[False,True,True]).iloc[0]
            self.selected[name]=best['candidate']
        write_json(R/'02_selection.json',{'fingerprint':self.fingerprint,'criterion':'validation AP, then smaller depth, then learning rate','selected':self.selected,'test_used_for_selection':False})
        return pd.DataFrame([x for x in self.rows if self.selected[x['variant']]==x['candidate']])

    def test(self):
        # This method is called only after immutable selection of both variants.
        assert len(self.selected)==2 and len(self.rows)==10
        predictions={};thresholds={}
        for name,weather in [('baseline',False),('weather',True)]:
            directory=DEST/name/self.selected[name]
            valid=pd.read_parquet(directory/'validation_predictions.parquet')
            thresholds[name]=choose_thresholds(valid.GoA,valid.probability)
            model=CatBoostClassifier();model.load_model(str(directory/'model.cbm'));frames=[]
            for f in self.ds.iter_split('test'):
                audit=f[['source_row','GoA','C40_CROSS_TIME','airport','weather_core_usable']].copy()
                audit['probability']=model.predict_proba(pool(f,weather))[:,1];frames.append(audit)
            pred=pd.concat(frames,ignore_index=True).sort_values('source_row').reset_index(drop=True)
            assert len(pred)==546577 and pred.GoA.sum()==1739
            pred.to_parquet(DEST/f'{name}_test_predictions.parquet',index=False);predictions[name]=pred
            calibration_table(pred.GoA,pred.probability).to_csv(R/f'03_{name}_calibration.csv',index=False)
            pd.DataFrame({'feature':model.feature_names_,'importance':model.feature_importances_}).to_csv(R/f'03_{name}_importance.csv',index=False)
        evaluation=evaluate_pair(predictions['baseline'],predictions['weather'],thresholds['baseline'],thresholds['weather'])
        evaluation.to_csv(R/'03_test_evaluation.csv',index=False)
        write_json(R/'03_thresholds.json',{name:{str(k):v for k,v in values.items()} for name,values in thresholds.items()})
        write_json(R/'01_run_status.json',{'status':'test_evaluated','selected':self.selected,'fingerprint':self.fingerprint})
        return evaluation
