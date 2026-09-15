from __future__ import annotations

import math
from typing import Any


STUDY_AREA = {
    "study_area_id": "DEHRADUN-CHANDRABANI-PS26192",
    "name": "Chandrabani focused micro-catchment study area",
    "district": "Dehradun",
    "state": "Uttarakhand",
    "public_crs": "EPSG:4326",
    "metric_crs": "EPSG:32643",
    "bounding_box": {
        "west": 77.968,
        "south": 30.270,
        "east": 78.000,
        "north": 30.300,
    },
    "center": {
        "longitude": 77.978689,
        "latitude": 30.285029,
    },
    "approx_area_km2": 10.25,
}

TERRAIN = {
    "source_name": "OpenTopoData SRTM 30m elevation API",
    "source_status": "OPEN_REAL_DATA",
    "derived_status": "DERIVED_FROM_REAL_DATA",
    "dataset": "SRTM 30m",
    "spatial_resolution_m": 30,
    "min_elevation_m": 594.0,
    "max_elevation_m": 646.0,
    "mean_elevation_m": 605.1,
    "mean_slope_deg": 0.46,
    "mean_slope_fraction": 0.0079,
    "nodata_count": 0,
    "hand_m": None,
    "twi": None,
}

TERRAIN_GRID = (
    (77.968, 30.270, 646.0),
    (77.976, 30.270, 634.0),
    (77.984, 30.270, 617.0),
    (77.992, 30.270, 610.0),
    (78.000, 30.270, 600.0),
    (77.968, 30.2775, 609.0),
    (77.976, 30.2775, 605.0),
    (77.984, 30.2775, 600.0),
    (77.992, 30.2775, 599.0),
    (78.000, 30.2775, 595.0),
    (77.968, 30.285, 597.0),
    (77.976, 30.285, 595.0),
    (77.984, 30.285, 596.0),
    (77.992, 30.285, 603.0),
    (78.000, 30.285, 599.0),
    (77.968, 30.2925, 594.0),
    (77.976, 30.2925, 595.0),
    (77.984, 30.2925, 600.0),
    (77.992, 30.2925, 599.0),
    (78.000, 30.2925, 609.0),
    (77.968, 30.300, 600.0),
    (77.976, 30.300, 601.0),
    (77.984, 30.300, 600.0),
    (77.992, 30.300, 609.0),
    (78.000, 30.300, 616.0),
)

ROAD_DIRECT_COORDS = (
    (77.9921844, 30.2815189),
    (77.9912487, 30.2822485),
    (77.9919438, 30.2838838),
    (77.9926018, 30.2854228),
    (77.9936385, 30.2876034),
    (77.9948480, 30.2898648),
    (77.9974537, 30.2940510),
)
ROAD_BYPASS_COORDS = (
    (77.9889999, 30.2752512),
    (77.9894658, 30.2750446),
    (77.9902560, 30.2745811),
    (77.9918576, 30.2736417),
    (77.9945152, 30.2720267),
    (77.9965798, 30.2700555),
)
ROAD_CLOSED_COORDS = (
    (77.9766333, 30.2810359),
    (77.9767205, 30.2809907),
)
STREAM_COORDS = (
    (77.9883529, 30.2996786),
    (77.9864617, 30.2978559),
    (77.9831561, 30.2950073),
    (77.9815021, 30.2941741),
    (77.9807278, 30.2910940),
    (77.9787451, 30.2892119),
    (77.9768518, 30.2877356),
)

ROADS = {
    "ROAD-SHELTER-CORRIDOR": {
        "name": "Transport Nagar Road",
        "road_class": "tertiary",
        "source": "OpenStreetMap",
        "source_status": "OPEN_REAL_DATA",
        "source_osm_id": 114099376,
        "dataset_version": "2026-05-06T03:25:00Z",
        "coordinates": ROAD_DIRECT_COORDS,
    },
    "ROAD-HIGHER-GROUND-BYPASS": {
        "name": "Post Office Road",
        "road_class": "tertiary",
        "source": "OpenStreetMap",
        "source_status": "OPEN_REAL_DATA",
        "source_osm_id": 101971528,
        "dataset_version": "2026-05-06T03:25:00Z",
        "coordinates": ROAD_BYPASS_COORDS,
    },
    "ROAD-BRIDGE-APPROACH": {
        "name": "Unnamed Road",
        "road_class": "unclassified",
        "source": "OpenStreetMap",
        "source_status": "OPEN_REAL_DATA",
        "source_osm_id": 1095906636,
        "dataset_version": "2026-05-06T03:25:00Z",
        "coordinates": ROAD_CLOSED_COORDS,
        "status_basis": "DEMO_AUTHORITY_CLOSURE_ONLY",
    },
}

STREAMS = {
    "OSM-STREAM-234936176": {
        "name": "Unnamed stream",
        "source": "OpenStreetMap",
        "source_status": "OPEN_REAL_DATA",
        "source_osm_id": 234936176,
        "dataset_version": "2026-07-24T11:04:51Z",
        "coordinates": STREAM_COORDS,
    }
}

SETTLEMENTS = {
    "VILLAGE-CHANDRABANI": {
        "name": "Chandrabani",
        "admin_level": "SETTLEMENT_POINT",
        "source": "OpenStreetMap",
        "source_status": "OPEN_REAL_DATA",
        "source_osm_id": 2379693646,
        "dataset_version": "2026-07-24T11:04:51Z",
        "coordinates": (77.978689, 30.285029),
    },
    "OSM-SETTLEMENT-SUBHASHNAGAR": {
        "name": "Subhashnagar",
        "admin_level": "SETTLEMENT_POINT",
        "source": "OpenStreetMap",
        "source_status": "OPEN_REAL_DATA",
        "source_osm_id": 2379693754,
        "dataset_version": "2026-07-24T11:04:51Z",
        "coordinates": (77.990146, 30.277767),
    },
}

CRITICAL_ASSETS = {
    "OSM-POI-4254504794": {
        "name": "Rajaram Mohan Roy Academy",
        "amenity": "school",
        "source": "OpenStreetMap",
        "source_status": "OPEN_REAL_DATA",
        "source_osm_id": 4254504794,
        "dataset_version": "2026-07-24T11:04:51Z",
        "coordinates": (77.9940942, 30.28497),
    },
    "OSM-POI-6950253649": {
        "name": "Baunthiyal Nursing Home",
        "amenity": "clinic",
        "source": "OpenStreetMap",
        "source_status": "OPEN_REAL_DATA",
        "source_osm_id": 6950253649,
        "dataset_version": "2026-07-24T11:04:51Z",
        "coordinates": (77.997745, 30.2837296),
    },
    "OSM-POI-6956686347": {
        "name": "Minocha Hospital and Maternity Home",
        "amenity": "hospital",
        "source": "OpenStreetMap",
        "source_status": "OPEN_REAL_DATA",
        "source_osm_id": 6956686347,
        "dataset_version": "2026-07-24T11:04:51Z",
        "coordinates": (77.9986905, 30.2919587),
    },
}

SHELTER = {
    "id": "SHELTER-SCHOOL-01",
    "name": "Demo shelter at Rajaram Mohan Roy Academy POI",
    "coordinates": CRITICAL_ASSETS["OSM-POI-4254504794"]["coordinates"],
    "source_status": "DEMO",
    "poi_source_status": "OPEN_REAL_DATA",
    "designation_status": "DEMO_EVACUATION_SHELTER_NOT_AUTHORITY_VERIFIED",
}


def bbox_polygon_coordinates() -> list[list[list[float]]]:
    bbox = STUDY_AREA["bounding_box"]
    return [
        [
            [bbox["west"], bbox["south"]],
            [bbox["east"], bbox["south"]],
            [bbox["east"], bbox["north"]],
            [bbox["west"], bbox["north"]],
            [bbox["west"], bbox["south"]],
        ]
    ]


def point_feature(
    feature_id: str,
    entity_type: str,
    coordinates: tuple[float, float],
    properties: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "Feature",
        "id": feature_id,
        "geometry": {
            "type": "Point",
            "coordinates": [coordinates[0], coordinates[1]],
        },
        "properties": {
            "id": feature_id,
            "entityType": entity_type,
            **properties,
        },
    }


def line_feature(
    feature_id: str,
    entity_type: str,
    coordinates: tuple[tuple[float, float], ...],
    properties: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "Feature",
        "id": feature_id,
        "geometry": {
            "type": "LineString",
            "coordinates": [[lon, lat] for lon, lat in coordinates],
        },
        "properties": {
            "id": feature_id,
            "entityType": entity_type,
            **properties,
        },
    }


def nearest_named(
    longitude: float,
    latitude: float,
    entities: dict[str, dict[str, Any]],
) -> tuple[str, float] | None:
    best: tuple[str, float] | None = None
    for entity_id, entity in entities.items():
        coordinates = entity["coordinates"]
        if isinstance(coordinates[0], tuple):
            distance = min(
                haversine_m(latitude, longitude, lat, lon)
                for lon, lat in coordinates
            )
        else:
            lon, lat = coordinates
            distance = haversine_m(latitude, longitude, lat, lon)
        if best is None or distance < best[1]:
            best = (entity_id, distance)
    return best


def nearest_elevation_m(longitude: float, latitude: float) -> float:
    return min(
        TERRAIN_GRID,
        key=lambda point: haversine_m(latitude, longitude, point[1], point[0]),
    )[2]


def within_study_area(longitude: float, latitude: float) -> bool:
    bbox = STUDY_AREA["bounding_box"]
    return (
        bbox["west"] <= longitude <= bbox["east"]
        and bbox["south"] <= latitude <= bbox["north"]
    )


def haversine_m(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    radius_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * radius_m * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
