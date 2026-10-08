-- UTC; raw state vectors retained, stale records evaluated separately
SELECT time, icao24, lat, lon, velocity, heading, vertrate, callsign,
       onground, baroaltitude, geoaltitude, lastposupdate, lastcontact
FROM minio.osky.state_vectors_data4
WHERE icao24 = '3c6605'
  AND hour IN (1546434000, 1546437600)
  AND time BETWEEN 1546434873 AND 1546440363
ORDER BY time;
