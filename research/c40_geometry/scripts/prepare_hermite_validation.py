"""Две новые даты для проверки обучаемой поправки к Эрмиту."""
from pathlib import Path
import hashlib
import json
from prepare_external_day import main as fetch_day

STUDY = Path(__file__).resolve().parents[1]


def main():
    # Даты и метод фиксируем до просмотра ошибок новых рейсов
    plan = {'version':'12-v1','train_days':['2019-11-04','2019-11-11','2019-11-18'],
            'validation_day':'2019-11-25','test_days':['2019-12-09','2019-12-16'],
            'train_source':'same 41 flights as 11, validation same 8 flights',
            'features':'same 19 endpoint features as 08-11',
            'target':'observed local position minus cubic Hermite local position',
            'alphas':[0.1,1.,10.,100.,1000.],
            'selection_metric':'macro mean WGS84 validation error, equal flight weights',
            'selection':'first 30 chronological eligible EGCC/EPWA cases, hour 22 UTC',
            'model11_sha256':hashlib.sha256((STUDY/'data/expanded_training/11_ridge_model.npz').read_bytes()).hexdigest()}
    path = STUDY/'reports/12_hermite_protocol.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf8')) == plan
    else:
        path.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    for day in plan['test_days']:
        print('Подготовка нового test:',day,flush=True)
        fetch_day(day,folder=STUDY/f'data/hermite_validation/{day}',report_prefix='12_'+day,max_cases=30)


if __name__ == '__main__': main()
