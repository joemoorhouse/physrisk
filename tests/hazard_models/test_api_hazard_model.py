from unittest.mock import patch

import pytest
import shapely.wkt
from physrisk.kernel.hazard_model import HazardDataRequest, HazardEventDataResponse
from physrisk.kernel.hazards import (
    CoastalInundation,
    PluvialInundation,
    RiverineInundation,
)

from physrisk.data.geocode import Geocoder
from physrisk.hazard_models.credentials_provider import EnvCredentialsProvider
from physrisk.hazard_models.hazard_cache import GeometryH3BasedCache, MemoryStore
from physrisk.hazard_models.jba_hazard_model import JBACacheKey, JBAHazardModel

from tests.conftest import cache_store_tests


def lats_lons():
    latitudes = [
        22.3022371,
        22.2968475,
        22.3314947,
    ]
    longitudes = [
        114.1867006,
        114.1733945,
        114.1777367,
    ]
    return latitudes, longitudes


def test_geocoding():
    latitudes, longitudes = lats_lons()
    geocoder = Geocoder()
    countries = geocoder.get_countries(latitudes, longitudes)
    assert (
        countries[0] == "HK"
    )  # note Hong Kong as Special Administrative Region of China


def test_continent_from_country_code():
    continent_and_country_from_code_iso_3166 = (
        Geocoder.get_continent_and_country_from_code_iso_3166(
            country_codes=["USA", "FR", 56]
        )
    )
    assert (
        continent_and_country_from_code_iso_3166["Continent"]["USA"] == "North America"
    )
    assert continent_and_country_from_code_iso_3166["Continent"]["FR"] == "Europe"
    assert continent_and_country_from_code_iso_3166["Continent"][56] == "Europe"


def test_spatial_keys():
    latitudes, longitudes = lats_lons()
    geometries = [
        shapely.wkt.loads(f"POINT({lon} {lat})")
        for lat, lon in zip(latitudes, longitudes)
    ]
    store = GeometryH3BasedCache(MemoryStore())
    spatial_key_lat_lon = store.spatial_key(latitudes[0], longitudes[0])
    spatial_key_geom = store.spatial_key(
        latitudes[0], longitudes[0], geometry=geometries[0]
    )
    assert spatial_key_lat_lon == "8c411c8691b0bff"
    assert spatial_key_geom == "wkbhash_C2amiZX82h4Ga8RxL7PO"


def test_jba_hazard_model(load_credentials, hazard_dir, update_inputs):
    """JBA test, using made-up cached data."""
    # latitudes, longitudes = lats_lons()
    latitudes, longitudes = [43.264209], [5.386365]

    # store = LMDBStore(str((Path(hazard_dir) / "temp" / "hazard_cache.db").absolute()))
    # cache = H3BasedCache(MemoryStore()) # in order to test this 'live', use the MemoryStore and enable API calls
    # cache = H3BasedCache(store)
    credentials = EnvCredentialsProvider(disable_api_calls=False)

    with cache_store_tests(__name__, update_inputs) as cache_store:
        model = JBAHazardModel(
            GeometryH3BasedCache(cache_store), credentials, max_requests=5
        )
        requests_riv = [
            HazardDataRequest(
                hazard_type=RiverineInundation,
                longitude=lon,
                latitude=lat,
                indicator_id="flood_depth",
                scenario="ssp585",
                year=2050,
                geometry=shapely.wkt.loads(f"POINT({lon} {lat})"),
            )
            for lat, lon in zip(latitudes, longitudes)
        ]
        requests_pluv = [
            HazardDataRequest(
                hazard_type=PluvialInundation,
                longitude=lon,
                latitude=lat,
                indicator_id="flood_depth",
                scenario="ssp585",
                year=2050,
                geometry=shapely.wkt.loads(f"POINT({lon} {lat})"),
            )
            for lat, lon in zip(latitudes, longitudes)
        ]
        response = model.get_hazard_data(requests_riv + requests_pluv)
        assert response is not None


def _stats_block(depth: float):
    # several return periods, deliberately not in a set-hash-friendly order, so a
    # regression that stops sorting the merged STSU_U return periods gets caught
    return {
        "rp_1500": {"max1500": depth * 6},
        "rp_20": {"max20": depth},
        "rp_500": {"max500": depth * 5},
        "rp_50": {"max50": depth * 1.5},
        "rp_200": {"max200": depth * 3},
        "rp_100": {"max100": depth * 2},
    }


async def _fake_flood_depth_no_future_storm_surge(
    self, api_request, access_token, session
):
    """Mocks JBA's flood-depths API as it behaves today: FLRF_U/FLSW_U/STSU_U are all
    returned for the historical baseline, but STSU_U (storm surge) is missing for every
    future scenario - the gap that the GMSLR backfill is meant to fill."""
    result = {}
    for spatial_key in api_request.spatial_keys:
        for jba_scenario in ["historical"] + self.jba_scenarios:
            is_historical = jba_scenario == "historical"
            result[JBACacheKey(jba_scenario, spatial_key)] = {
                "stats": {
                    "FLRF_U": _stats_block(0.5),
                    "STSU_U": _stats_block(0.5) if is_historical else {},
                }
            }
    return result


async def _fake_storm_surge_slr(self, api_request, access_token, session):
    """Mocks JBA's GMSLR API: each bucket returns a distinct storm-surge depth so that
    interpolation between buckets can be verified precisely."""
    bucket_depths_m = {"GMSLR1": 1.0, "GMSLR2": 2.0, "GMSLR4": 4.0, "GMSLR8": 8.0}
    depth = bucket_depths_m[api_request.country_code]
    return {
        JBACacheKey(api_request.country_code, spatial_key): {
            "stats": {"STSU_U": _stats_block(depth)}
        }
        for spatial_key in api_request.spatial_keys
    }


def test_coastal_storm_surge_backfill():
    """Storm surge (STSU_U) is not yet returned by JBA for future scenarios; the model
    backfills it from JBA's separate GMSLR API, linearly interpolated between the two
    buckets that bracket the scenario/year's expected sea level rise. ssp585/2050 maps to
    0.23m, which falls between the GMSLR2 (0.2m) and GMSLR4 (0.4m) buckets, giving weights
    of 0.85 and 0.15 respectively."""
    latitude, longitude = 43.264209, 5.386365
    point = shapely.wkt.loads(f"POINT({longitude} {latitude})")
    model = JBAHazardModel(
        GeometryH3BasedCache(MemoryStore()),
        credentials=EnvCredentialsProvider(disable_api_calls=False),
        max_requests=1000,
        restrict_coverage=True,  # fewer scenarios requested, keeps the mock small
    )
    coastal_request = HazardDataRequest(
        hazard_type=CoastalInundation,
        longitude=longitude,
        latitude=latitude,
        indicator_id="flood_depth",
        scenario="ssp585",
        year=2050,
        geometry=point,
    )
    riverine_request = HazardDataRequest(
        hazard_type=RiverineInundation,
        longitude=longitude,
        latitude=latitude,
        indicator_id="flood_depth",
        scenario="ssp585",
        year=2050,
        geometry=point,
    )
    historical_request = HazardDataRequest(
        hazard_type=CoastalInundation,
        longitude=longitude,
        latitude=latitude,
        indicator_id="flood_depth",
        scenario="historical",
        year=2020,
        geometry=point,
    )

    with (
        patch.object(
            JBAHazardModel, "flood_depth", _fake_flood_depth_no_future_storm_surge
        ),
        patch.object(JBAHazardModel, "storm_surge_slr", _fake_storm_surge_slr),
    ):
        result = model.get_hazard_data(
            [coastal_request, riverine_request, historical_request]
        )

    coastal_response = result[coastal_request]
    assert isinstance(coastal_response, HazardEventDataResponse)
    # return periods from the merged/interpolated STSU_U must come out in ascending
    # order - a regression here previously broke downstream return-period interpolation
    assert list(coastal_response.return_periods) == sorted(
        coastal_response.return_periods
    )
    coastal_by_rp = dict(
        zip(coastal_response.return_periods, coastal_response.intensities)
    )
    assert coastal_by_rp[20.0] == pytest.approx(0.85 * 2.0 + 0.15 * 4.0)
    assert coastal_by_rp[100.0] == pytest.approx(0.85 * 4.0 + 0.15 * 8.0)
    assert coastal_by_rp[1500.0] == pytest.approx(0.85 * 12.0 + 0.15 * 24.0)

    # riverine data (FLRF_U) is untouched by the storm-surge backfill
    riverine_response = result[riverine_request]
    assert isinstance(riverine_response, HazardEventDataResponse)
    riverine_by_rp = dict(
        zip(riverine_response.return_periods, riverine_response.intensities)
    )
    assert riverine_by_rp[20.0] == pytest.approx(0.5)
    assert riverine_by_rp[100.0] == pytest.approx(1.0)

    # historical STSU_U comes straight from the main API, not the GMSLR backfill
    historical_response = result[historical_request]
    assert isinstance(historical_response, HazardEventDataResponse)
    historical_by_rp = dict(
        zip(historical_response.return_periods, historical_response.intensities)
    )
    assert historical_by_rp[20.0] == pytest.approx(0.5)
    assert historical_by_rp[100.0] == pytest.approx(1.0)
