-- План запроса; не выполнялся. Время UTC, исходные позиции не фильтруем по свежести.
-- SHT2N_3852; C40 2020-09-18T17:18:00+00:00
SELECT time, icao24, callsign, lat, lon, velocity, heading,
       baroaltitude, geoaltitude, vertrate, onground,
       lastposupdate, lastcontact, hour
FROM minio.osky.state_vectors_data4
WHERE hour IN (1600448400)
  AND icao24 = '4009d8'
  AND time >= 1600449180 AND time < 1600450227
ORDER BY time;
