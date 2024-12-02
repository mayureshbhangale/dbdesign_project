WITH RECURSIVE RouteHierarchy AS (
    SELECT
        f1.origin_airport AS origin_airport,
        f1.destination_airport AS destination_airport,
        ARRAY[f1.origin_airport, f1.destination_airport] AS route_path,
        1 AS level
    FROM flights f1
    WHERE f1.departure_delay > 50 
      AND f1.month BETWEEN 1 AND 3
      AND f1.distance > 800 

    UNION ALL

    
    SELECT
        rh.origin_airport AS origin_airport,
        f2.destination_airport AS destination_airport,
        rh.route_path || f2.destination_airport AS route_path,
        rh.level + 1 AS level
    FROM RouteHierarchy rh
    JOIN flights f2 ON rh.destination_airport = f2.origin_airport
    WHERE NOT f2.destination_airport = ANY(rh.route_path)
      AND rh.level < 2 
      AND f2.departure_delay > 50 
      AND f2.month BETWEEN 1 AND 3 
      AND f2.distance > 800 
),
AggregatedRoutes AS (
    SELECT
        rh.origin_airport,
        rh.destination_airport,
        rh.level AS connection_depth,
        COUNT(*) AS total_routes
    FROM RouteHierarchy rh
    GROUP BY rh.origin_airport, rh.destination_airport, rh.level
    HAVING COUNT(*) > 5 
SELECT
    ar.origin_airport AS origin_airport,
    ar.destination_airport AS destination_airport,
    ar.connection_depth AS connection_depth,
    ar.total_routes AS total_routes
FROM AggregatedRoutes ar
ORDER BY connection_depth DESC, total_routes DESC
LIMIT 20000;
