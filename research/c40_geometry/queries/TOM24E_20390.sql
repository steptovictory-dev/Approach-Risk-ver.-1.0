-- План запроса; не выполнялся. Время UTC, исходные позиции не фильтруем по свежести.
-- TOM24E_20390; C40 2019-02-05T22:42:00+00:00
SELECT time, icao24, callsign, lat, lon, velocity, heading,
       baroaltitude, geoaltitude, vertrate, onground,
       lastposupdate, lastcontact, hour
FROM minio.osky.state_vectors_data4
WHERE hour IN (1549404000)
  AND icao24 = '40647f'
  AND time >= 1549406220 AND time < 1549407334
ORDER BY time;
