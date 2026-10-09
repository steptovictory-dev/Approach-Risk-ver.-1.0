"""Пилот восстановления скрытых наблюдений; обучение здесь не выполняется."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'research/c40_geometry'
sys.path.insert(0, str(ROOT / 'europe/scripts'))
from audit_c40_radial_cases import ellipsoid_distance


def forward(lat, lon, heading, meters):
    # Кинематический baseline на сфере радиуса 6371000 м.
    phi, lam, az = np.radians([lat, lon, heading])
    arc = meters / 6371000
    p = np.arcsin(np.sin(phi)*np.cos(arc) + np.cos(phi)*np.sin(arc)*np.cos(az))
    l = lam + np.arctan2(np.sin(az)*np.sin(arc)*np.cos(phi), np.cos(arc)-np.sin(phi)*np.sin(p))
    return float(np.degrees(p)), float((np.degrees(l)+180)%360-180)


def main(source=None, prefix='07'):
    source = Path(source).resolve() if source is not None else STUDY / 'data/public_pilot/EPWA_control_geometry.csv'
    frame = pd.read_csv(source)
    rows, audits = [], []
    for flight_id, raw in frame.groupby('case_flight_id', sort=True):
        raw = raw.sort_values('time')
        # Дубликаты времени неоднозначны: прекращаем расчет, не выбираем ответ молча.
        if raw.time.duplicated().any():
            raise ValueError(f'{flight_id}: повтор времени; требуется аудит')
        valid = raw[['time', 'lat', 'lon', 'lastposupdate']].notna().all(axis=1)
        valid &= raw.lat.between(-90, 90) & raw.lon.between(-180, 180)
        valid &= (raw.time-raw.lastposupdate).between(0, 2)
        valid &= raw.onground.eq(False)
        audits.append(dict(flight_id=flight_id, raw_rows=len(raw), eligible_rows=int(valid.sum())))
        # Сохраняем исходные индексы: не соединяем через исключенные наблюдения.
        track = raw.reset_index(drop=True)
        ok = valid.to_numpy()
        for gap in [20, 30, 60, 120]:
            for left in range(len(track)-2):
                a = track.iloc[left]
                candidates = np.flatnonzero(track.time.to_numpy() == a.time+gap)
                if len(candidates) != 1:
                    continue
                right = int(candidates[0])
                if right <= left+1 or not ok[left:right+1].all():
                    continue
                if not np.all(np.diff(track.time.iloc[left:right+1]) == 10):
                    continue
                b = track.iloc[right]
                if not np.isfinite([a.velocity, a.heading]).all() or a.velocity < 0:
                    continue
                turn = abs((b.heading-a.heading+180)%360-180) if pd.notna(b.heading) else np.nan
                for target in track.iloc[left+1:right].itertuples():
                    dt = target.time-a.time
                    w = dt/gap
                    near = a if w <= .5 else b
                    predictions = {
                        'nearest': (near.lat, near.lon),
                        'linear_lat_lon': (a.lat+w*(b.lat-a.lat), a.lon+w*((b.lon-a.lon+180)%360-180)),
                        'constant_velocity_sphere': forward(a.lat, a.lon, a.heading, a.velocity*dt),
                    }
                    for method, (lat, lon) in predictions.items():
                        error = float(1000*ellipsoid_distance(target.lat, target.lon, np.array([lat]), np.array([lon]))[0])
                        rows.append(dict(flight_id=flight_id, gap_s=gap, left_time=a.time, target_time=target.time,
                            method=method, error_m=error, endpoint_heading_change_deg=turn,
                            target_position_age_s=target.time-target.lastposupdate))
    errors = pd.DataFrame(rows)
    if errors.empty:
        raise ValueError('Нет пригодных интервалов')
    summary = errors.groupby(['flight_id', 'gap_s', 'method']).error_m.agg(
        targets='size', median_m='median', mean_m='mean', p95_m=lambda x: x.quantile(.95), max_m='max').reset_index()
    errors.to_csv(source.parent/f'{prefix}_hidden_point_errors.csv', index=False)
    summary.to_csv(STUDY/f'reports/{prefix}_hidden_point_summary.csv', index=False)
    result = dict(source=str(source.relative_to(ROOT)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        flights=audits, gaps_s=[20,30,60,120], max_position_age_s=2,
        overlapping_windows=True, training_performed=False, evaluation_rows=len(errors),
        truth='held-out OpenSky observations, not independently measured true positions')
    (STUDY/f'reports/{prefix}_hidden_point_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(summary.to_string(index=False))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
