CREATE OR REPLACE VIEW tank_parsed AS
SELECT 
    tank_id,
    org_id,
    TRY_CAST(json_extract_string(obj, '$.d') AS DOUBLE) as d,
    TRY_CAST(json_extract_string(obj, '$.yearStart') AS INTEGER) as yearStart,
    TRY_CAST(json_extract_string(obj, '$.beltQuantity') AS INTEGER) as beltQuantity,
    json_extract_string(obj, '$.tankType') as tankType,
    json_extract_string(obj, '$.tankFluid') as tankFluid,
    json_extract_string(obj, '$.fluidCorr') as fluidCorr,
    json_extract_string(obj, '$.steelGradeWall') as steelGradeWall,
    json_extract_string(obj, '$.baseType') as baseType,
    json_extract_string(obj, '$.roof_type') as roof_type,
    TRY_CAST(json_extract_string(obj, '$.maxWallT') AS DOUBLE) as maxWallT,
    TRY_CAST(json_extract_string(obj, '$.minWallT') AS DOUBLE) as minWallT
FROM read_csv_auto('{{normalized_path}}/../tank.csv', header=true, ignore_errors=true);

CREATE OR REPLACE VIEW element_parsed AS
SELECT 
    element_id,
    tank_id,
    tag,
    COALESCE(json_extract_string(obj, '$.rPart'), '') as rPart,
    COALESCE(json_extract_string(obj, '$.rSubpart'), '') as rSubpart,
    COALESCE(json_extract_string(obj, '$.controlType'), '') as controlType,
    COALESCE(json_extract_string(obj, '$.weldFinish'), '') as weldFinish,
    COALESCE(json_extract_string(obj, '$.weldMethod'), '') as weldMethod,
    COALESCE(json_extract_string(obj, '$.weldType'), '') as weldType
FROM read_csv_auto('{{normalized_path}}/../element.csv', header=true, ignore_errors=true);

CREATE OR REPLACE VIEW element_data_parsed AS
SELECT 
    id,
    element_id,
    Activity_id as activityId,
    TRY_CAST(json_extract_string(obj, '$.delta') AS DOUBLE) as delta,
    json_extract_string(obj, '$.createdAt') as createdAt,
    TRY_CAST(json_extract_string(obj, '$.H') AS DOUBLE) as H,
    TRY_CAST(json_extract_string(obj, '$.L') AS DOUBLE) as L,
    TRY_CAST(json_extract_string(obj, '$.W') AS DOUBLE) as W,
    COALESCE(json_extract_string(obj, '$.repair'), 'false') as repair,
    COALESCE(json_extract_string(obj, '$.defectKind'), '') as defectKind,
    COALESCE(json_extract_string(obj, '$.defectType'), '') as defectType,
    COALESCE(json_extract_string(obj, '$.geometryDefect'), '') as geometryDefect
FROM read_csv_auto('{{normalized_path}}/../ElementData.csv', header=true, ignore_errors=true)
WHERE json_extract_string(obj, '$.delta') IS NOT NULL;

CREATE OR REPLACE VIEW activity_parsed AS
SELECT 
    id,
    tank_id,
    COALESCE(json_extract_string(obj_new, '$.description'), '') as description,
    COALESCE(json_extract_string(obj_new, '$.activityType'), '') as activityType,
    COALESCE(json_extract_string(obj_new, '$.act_subtype'), '') as act_subtype
FROM read_csv_auto('{{normalized_path}}/../Activity.csv', header=true, ignore_errors=true);

COPY (
    WITH LaggedData AS (
        SELECT
            *,
            LAG(try_strptime(createdAt, ['%Y-%m-%dT%H:%M:%S.%f%z', '%Y-%m-%dT%H:%M:%S%z']), 1) 
                OVER (PARTITION BY element_id ORDER BY try_strptime(createdAt, ['%Y-%m-%dT%H:%M:%S.%f%z', '%Y-%m-%dT%H:%M:%S%z'])) as prev_measurement_date,
            LAG(delta, 1) 
                OVER (PARTITION BY element_id ORDER BY try_strptime(createdAt, ['%Y-%m-%dT%H:%M:%S.%f%z', '%Y-%m-%dT%H:%M:%S%z'])) as prev_delta
        FROM element_data_parsed
        WHERE delta IS NOT NULL
    )
    SELECT
        ed.delta AS delta,

        YEAR(try_strptime(ed.createdAt, ['%Y-%m-%dT%H:%M:%S.%f%z', '%Y-%m-%dT%H:%M:%S%z'])) AS measurement_year,
        (YEAR(try_strptime(ed.createdAt, ['%Y-%m-%dT%H:%M:%S.%f%z', '%Y-%m-%dT%H:%M:%S%z'])) - t.yearStart) AS tank_age,
        DATE_DIFF('day', ed.prev_measurement_date, try_strptime(ed.createdAt, ['%Y-%m-%dT%H:%M:%S.%f%z', '%Y-%m-%dT%H:%M:%S%z'])) as days_since_prev_measurement,
        CASE 
            WHEN ed.prev_delta IS NOT NULL AND DATE_DIFF('day', ed.prev_measurement_date, try_strptime(ed.createdAt, ['%Y-%m-%dT%H:%M:%S.%f%z', '%Y-%m-%dT%H:%M:%S%z'])) > 0
            THEN (ed.delta - ed.prev_delta) / DATE_DIFF('day', ed.prev_measurement_date, try_strptime(ed.createdAt, ['%Y-%m-%dT%H:%M:%S.%f%z', '%Y-%m-%dT%H:%M:%S%z']))
            ELSE NULL
        END as corrosion_rate,
        
        ed.H as defect_H,
        ed.L as defect_L,
        ed.W as defect_W,
        ed.repair,
        ed.defectKind,
        ed.defectType,
        ed.geometryDefect,

        e.rPart AS r_part,
        e.rSubpart AS r_subpart,
        e.controlType,
        e.weldFinish,
        e.weldMethod,
        e.weldType,

        t.d AS diameter,
        t.beltQuantity AS belt_quantity,
        t.tankType,
        t.tankFluid,
        t.fluidCorr,
        t.steelGradeWall,
        t.baseType,
        t.roof_type,
        t.maxWallT as max_wall_temp,
        t.minWallT as min_wall_temp,
        
        a.description AS activity_description,
        a.activityType,
        a.act_subtype

    FROM LaggedData ed
    LEFT JOIN element_parsed e ON ed.element_id = e.element_id
    LEFT JOIN tank_parsed t ON e.tank_id = t.tank_id
    LEFT JOIN activity_parsed a ON ed.activityId = a.id
    WHERE ed.delta IS NOT NULL
        AND ed.delta > 0
        AND ed.delta < 100
) TO '{{output_path}}' (FORMAT CSV, HEADER);
