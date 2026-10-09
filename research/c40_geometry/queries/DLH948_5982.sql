-- План запроса; не выполнялся. Время UTC, исходные позиции не фильтруем по свежести.
-- DLH948_5982; C40 2019-04-30T21:41:48+00:00
SELECT time, icao24, callsign, lat, lon, velocity, heading,
       baroaltitude, geoaltitude, vertrate, onground,
       lastposupdate, lastcontact, hour
FROM minio.osky.state_vectors_data4
WHERE hour IN (1556658000)
  AND icao24 = '3c6645'
  AND time >= 1556660208 AND time < 1556661174
ORDER BY time;
