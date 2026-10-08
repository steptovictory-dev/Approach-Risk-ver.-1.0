-- UTC; raw state vectors retained, stale records evaluated separately
SELECT time, icao24, lat, lon, velocity, heading, vertrate, callsign,
       onground, baroaltitude, geoaltitude, lastposupdate, lastcontact
FROM minio.osky.state_vectors_data4
WHERE icao24 = '3c65c1'
  AND hour IN (1556604000, 1556607600)
  AND time BETWEEN 1556604388 AND 1556610013
ORDER BY time;
