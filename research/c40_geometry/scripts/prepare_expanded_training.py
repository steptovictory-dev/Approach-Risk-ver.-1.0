"""Новые даты для проверки влияния объема обучения."""
from pathlib import Path
import hashlib
import json
from prepare_external_day import main as fetch_day

STUDY = Path(__file__).resolve().parents[1]


def main():
    # Распределяем даты заранее и не меняем их после получения ошибок
    plan = {'version':'11-v1','train_days':['2019-11-04','2019-11-11','2019-11-18'],
            'validation_day':'2019-11-25','test_day':'2019-12-02',
            'existing_train':'six train flights from 08 plus all ten flights from inspected 10',
            'selection':'first 30 chronological eligible EGCC/EPWA cases, hour 22 UTC',
            'alphas':[0.1,1.,10.,100.,1000.],
            'comparison':'same Ridge features; old 08, six-flight refit, expanded refit; one new test day',
            'old_model_sha256':hashlib.sha256((STUDY/'data/public_pilot/08_ridge_model.npz').read_bytes()).hexdigest()}
    path = STUDY/'reports/11_expanded_protocol.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf8')) == plan
    else:
        path.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    for day in plan['train_days']+[plan['validation_day'],plan['test_day']]:
        print('Подготовка даты:',day,flush=True)
        fetch_day(day,folder=STUDY/f'data/expanded_training/{day}',
                  report_prefix='11_'+day,max_cases=30)


if __name__ == '__main__': main()
