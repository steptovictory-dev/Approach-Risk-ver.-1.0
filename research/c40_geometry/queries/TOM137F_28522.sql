-- План запроса; не выполнялся. Время UTC, исходные позиции не фильтруем по свежести.
-- TOM137F_28522; C40 2019-11-07T12:14:59+00:00
SELECT time, icao24, callsign, lat, lon, velocity, heading,
       baroaltitude, geoaltitude, vertrate, onground,
       lastposupdate, lastcontact, hour
FROM minio.osky.state_vectors_data4
WHERE hour IN (1573128000)
  AND icao24 = '406d70'
  AND time >= 1573128599 AND time < 1573130281
ORDER BY time;
