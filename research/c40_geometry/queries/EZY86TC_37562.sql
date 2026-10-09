-- План запроса; не выполнялся. Время UTC, исходные позиции не фильтруем по свежести.
-- EZY86TC_37562; C40 2019-08-20T22:38:40+00:00
SELECT time, icao24, callsign, lat, lon, velocity, heading,
       baroaltitude, geoaltitude, vertrate, onground,
       lastposupdate, lastcontact, hour
FROM minio.osky.state_vectors_data4
WHERE hour IN (1566338400)
  AND icao24 = '40756d'
  AND time >= 1566340420 AND time < 1566341409
ORDER BY time;
