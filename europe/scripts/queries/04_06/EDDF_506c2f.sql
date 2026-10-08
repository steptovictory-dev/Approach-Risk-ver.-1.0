-- UTC; raw state vectors retained, stale records evaluated separately
SELECT time, icao24, lat, lon, velocity, heading, vertrate, callsign,
       onground, baroaltitude, geoaltitude, lastposupdate, lastcontact
FROM minio.osky.state_vectors_data4
WHERE icao24 = '506c2f'
  AND hour IN (1565625600, 1565629200)
  AND time BETWEEN 1565626413 AND 1565631624
ORDER BY time;
