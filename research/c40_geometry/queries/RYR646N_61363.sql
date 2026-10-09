-- План запроса; не выполнялся. Время UTC, исходные позиции не фильтруем по свежести.
-- RYR646N_61363; C40 2019-10-15T22:17:37+00:00
SELECT time, icao24, callsign, lat, lon, velocity, heading,
       baroaltitude, geoaltitude, vertrate, onground,
       lastposupdate, lastcontact, hour
FROM minio.osky.state_vectors_data4
WHERE hour IN (1571176800)
  AND icao24 = '4ca75f'
  AND time >= 1571177557 AND time < 1571178395
ORDER BY time;
