# Публичная траектория EPWA – первый пилот

Из публичного часового архива OpenSky извлечен борт 489221. Сопоставление выполнено по ICAO24, времени и callsign. Использован восстановленный центр EPWA; исторический ARP еще не подтвержден независимо.

Пересечение 40 NM выделено по соседним наблюдениям снаружи и внутри границы. Линейная интерполяция расстояния во времени дает оценку, а не измеренный момент пересечения. Проверяем интервал и свежесть позиции. Это OpenSky ADS-B; совпадение с исходными CPR EUROCONTROL не предполагается. Результат этого рейса не переносится автоматически на EGCC.

Исходные наблюдения и геометрия сохранены локально в data/public_pilot, которая исключена из Git. Контрольная сумма и URL архива – в 05_public_pilot_download.json. Все записи основной базы сохраняются.

В момент метаданных существует наблюдение OpenSky с тем же ICAO24 и callsign, но расстояние между позициями около 43 км. Определение радиальной границы само по себе не объясняет пространственное несоответствие. Причина и источник расхождения пока не установлены. Проверку дополнить независимым историческим ARP и другими случаями.

```json
{
  "flight_id": "ENT73JH_21100",
  "icao24": "489221",
  "airport": "EPWA",
  "rows": 386,
  "local_rows": 61,
  "center_status": "recovered_center_historical_ARP_pending",
  "c40_utc": "2019-10-21 22:49:20+00:00",
  "metadata_sphere_km": 66.95418607324106,
  "metadata_wgs84_km": 67.080558734863,
  "closest_time_seconds_offset": 0.0,
  "closest_time_point_distance_m": 43035.862589326345,
  "closest_position_seconds_offset": 180.0,
  "closest_position_distance_m": 29782.061005509328,
  "median_observed_interval_s": 10.0,
  "crossings": [
    {
      "model": "sphere_km",
      "before_utc": "2019-10-21 22:49:40+00:00",
      "after_utc": "2019-10-21 22:49:50+00:00",
      "before_radius_km": 74.1743838584967,
      "after_radius_km": 72.48951869701217,
      "gap_s": 10,
      "linear_estimate_utc": "2019-10-21 22:49:40.560186386+00:00",
      "estimated_seconds_before_metadata": -20.5601863861084,
      "max_position_age_s": 0.009999990463256836
    },
    {
      "model": "wgs84_km",
      "before_utc": "2019-10-21 22:49:40+00:00",
      "after_utc": "2019-10-21 22:49:50+00:00",
      "before_radius_km": 74.40287814414137,
      "after_radius_km": 72.71227035250857,
      "gap_s": 10,
      "linear_estimate_utc": "2019-10-21 22:49:41.909834623+00:00",
      "estimated_seconds_before_metadata": -21.909834623336792,
      "max_position_age_s": 0.009999990463256836
    }
  ],
  "next_hour_downloaded": true,
  "callsigns": [
    "",
    "ENT73JH"
  ],
  "observed_at_c40": {
    "lat": 52.01621362718485,
    "lon": 19.8584943498884,
    "sphere_km": 77.5521324738327,
    "wgs84_km": 77.79210439412596,
    "position_age_s": 0.04800009727478027
  },
  "metadata_coordinates": {
    "lat": 51.73416666666665,
    "lon": 20.286111111111108
  },
  "near_landing": {
    "utc": "2019-10-21 23:02:30+00:00",
    "time_offset_s": -5.0,
    "radius_wgs84_km": 1.5670873695709933,
    "baroaltitude_m": 53.34,
    "onground": false
  }
}
```
