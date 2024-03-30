DO $$
DECLARE
    r record;
    outputsql text;
BEGIN
    -- Temporary table to store assignments
    CREATE TEMP TABLE IF NOT EXISTS temp_assignments (
        khate_number INT,
        full_name TEXT,
        total_holding_area NUMERIC,
        key INT,
        point_geom geometry(point, 32643),
        farm_geom geometry(multipolygon, 32643),
        distance FLOAT
    );
    
    -- Loop through each point
    FOR r IN
        SELECT
            khate_number,
            full_name,
            total_holding_area,
            geom
        FROM
            filtered_amravati_pahani
        GROUP BY
            khate_number,
            full_name,
            total_holding_area,
            geom
    LOOP
        -- Attempt to find the closest matching farmplot that hasn't been assigned yet
        WITH eligible_farmplots AS (
            SELECT
                f.geom,
                f.key,
                ST_Distance(r.geom, f.geom) AS distance
            FROM 
                s2.filtered_farmplots_amravati as f
            WHERE 
                ST_intersects(st_buffer(r.geom, 50), f.geom)
            AND
                ST_Area(f.geom)
            BETWEEN
                r.total_holding_area*9500 AND r.total_holding_area*10500
            AND
                NOT EXISTS (
                    SELECT
                        1
                    FROM
                        temp_assignments
                    WHERE
                        key = f.key
                )
            ORDER BY 
                distance
            LIMIT 1
        )
        INSERT INTO temp_assignments (
            khate_number,
            full_name,
            total_holding_area,
            key,
            point_geom,
            farm_geom,
            distance
        )
        SELECT
            r.khate_number,
            r.full_name,
            r.total_holding_area,
            eligible_farmplots.key,
            r.geom,
            eligible_farmplots.geom,
            eligible_farmplots.distance
        FROM
            eligible_farmplots;
    END LOOP;

    -- Output the assignments
    outputsql := '
    DROP TABLE IF EXISTS nearest_farmplots_map;
    CREATE TABLE nearest_farmplots_map as
    SELECT * FROM temp_assignments;
    ';
    EXECUTE(outputsql);
    
    -- Cleanup: Drop the temporary table if desired
    DROP TABLE temp_assignments;
END $$;
