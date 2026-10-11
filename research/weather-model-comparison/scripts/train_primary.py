"""Supervised binary classification of GoA using logistic regression.

Both models fit transformations on train only. Validation selects thresholds;
test measures performance. See methodology.md before every training run.
"""
from pathlib import Path
import hashlib, json, time, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from comparison_pipeline import BASE, iter_model_split, prepare_features, build_pipeline, choose_thresholds, metrics, calibration_table, evaluate_pair

def main():
    print('Методика: бинарная классификация GoA; логистический классификатор L2, C=1; fit только train.',flush=True)
    dest=BASE/'data/primary_models_v1'; dest.mkdir(parents=True,exist_ok=True)
    report=BASE/'reports'; predictions={}; results=[]
    train=pd.concat(list(iter_model_split('train')),ignore_index=True)
    assert len(train)==4165060 and train.GoA.sum()==11626
    y=train.GoA.to_numpy(dtype=int)
    for weather,name in [(False,'baseline'),(True,'weather')]:
        print('Перед обучением '+name+': бинарная классификация, логистическая регрессия; train 2019–2021.',flush=True)
        pipe=build_pipeline(weather=weather); x=prepare_features(train,weather=weather)
        start=time.time()
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=6):
            warnings.simplefilter('always',ConvergenceWarning)
            pipe.fit(x,y)
        convergence=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
        joblib.dump(pipe,dest/f'{name}.joblib')
        info={'model':name,'method':'supervised binary classification / logistic regression L2 C=1',
              'train_rows':len(train),'train_GoA':int(y.sum()),'converged':convergence,
              'iterations':int(pipe.named_steps['model'].n_iter_.max()),'seconds':time.time()-start,
              'warnings':[str(w.message) for w in caught]}
        results.append(info)
        print(json.dumps(info,ensure_ascii=False),flush=True)
        if not convergence: raise RuntimeError('Classifier did not converge; results not final')
        del x
        predictions[name]={}
        for split in ['validation','test']:
            frames=[]
            for frame in iter_model_split(split):
                audit=frame[['source_row','GoA','C40_CROSS_TIME','airport','weather_core_usable']].copy()
                with threadpool_limits(limits=6): audit['probability']=pipe.predict_proba(prepare_features(frame,weather=weather))[:,1]
                frames.append(audit)
            pred=pd.concat(frames,ignore_index=True).sort_values('source_row').reset_index(drop=True)
            pred.to_parquet(dest/f'{name}_{split}_predictions.parquet',index=False)
            predictions[name][split]=pred
            calibration_table(pred.GoA,pred.probability).to_csv(report/f'04_{name}_{split}_calibration.csv',index=False)
            print(name+' '+split+': '+json.dumps(metrics(pred.GoA,pred.probability)),flush=True)
        del pipe
    thresholds={name:choose_thresholds(p['validation'].GoA,p['validation'].probability) for name,p in predictions.items()}
    for split in ['validation','test']:
        evaluate_pair(predictions['baseline'][split],predictions['weather'][split],thresholds['baseline'],thresholds['weather']).to_csv(report/f'04_{split}_evaluation.csv',index=False)
    manifest={'verified':True,'training_executed':True,'methodology':'binary classification; logistic regression',
              'methodology_sha256':hashlib.sha256((BASE/'methodology.md').read_bytes()).hexdigest(),
              'models':results,'thresholds':{n:{str(k):v for k,v in t.items()} for n,t in thresholds.items()},
              'input_manifest_sha256':hashlib.sha256((report/'02_model_table_manifest.json').read_bytes()).hexdigest(),
              'pending':['no-aircraft pair','gap mask pair','day bootstrap','delay/COR sensitivity']}
    manifest['files']=[{'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in dest.iterdir() if p.is_file()]
    (report/'04_training_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('Основная пара обучена и оценена.',flush=True)

if __name__=='__main__': main()
