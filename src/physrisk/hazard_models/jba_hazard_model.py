import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import aiohttp
import numpy as np
from shapely.geometry.base import BaseGeometry

from physrisk.data.hazard_data_provider import HazardDataProvider, ScenarioYear
from physrisk.kernel.hazard_model import (
    HazardDataFailedResponse,
    HazardDataRequest,
    HazardDataResponse,
    HazardEventDataResponse,
    HazardModel,
    HazardParameterDataResponse,
)
from physrisk.kernel.hazards import (
    CoastalInundation,
    PluvialInundation,
    RiverineInundation,
)
from physrisk.data.geocode import Geocoder
from physrisk.utils.event_loop import get_loop, run
from physrisk.hazard_models.credentials_provider import (
    CredentialsProvider,
    EnvCredentialsProvider,
)
from physrisk.hazard_models.hazard_cache import GeometryH3BasedCache

logger = logging.getLogger(__name__)


class Indicator(NamedTuple):
    hazard_type: str
    indicator_id: str


class ItemType(str, Enum):
    request = "request"
    response = "response"


class JBACacheKey(NamedTuple):
    # JBA responses for a given lat/lon contain all hazards but a single scenario, hence the key comprises:
    jba_scenario: str  # the JBA combination of scenario and year
    spatial_key: str


class RequestKey(NamedTuple):
    country_code: str  # 2 letter code
    # jba_scenario: str


@dataclass
class APIRequest:
    spatial_keys: Sequence[str]  # spatial keys for latitudes and longitudes in order
    latitudes: Sequence[float]
    longitudes: Sequence[float]
    geometries: Sequence[Optional[BaseGeometry]]
    country_code: str
    location_cache_keys: Dict[
        str, List[JBACacheKey]
    ]  # for each spatial_keys list of cache keys that is requested


@dataclass
class RequestWeights:
    request: HazardDataRequest
    weights: List[Tuple[JBACacheKey, float]]


class JBAHazardModel(HazardModel):
    # GMSLR buckets available for the storm-surge backfill, in metres of global mean sea
    # level rise - see the comment above _gmslr_interpolation_weights.
    _GMSLR_BUCKET_METRES: Tuple[Tuple[str, float], ...] = (
        ("GMSLR1", 0.1),
        ("GMSLR2", 0.2),
        ("GMSLR4", 0.4),
        ("GMSLR8", 0.8),
    )

    def __init__(
        self,
        cache_store: GeometryH3BasedCache,
        credentials: Optional[CredentialsProvider] = None,
        geocoder: Optional[Geocoder] = None,
        max_requests: int = 5,
        cmip: int = 6,
        batch_size: int = 100,
        restrict_coverage: bool = False,
        only_request_required: bool = False,
        default_buffer: int = 10,
        backfill_storm_surge_slr: bool = True,
    ):
        """JBAHazardModel retrieves data via the JBA API.
        https://api.jbarisk.com/docs/index.html
        Note that JBA requests are changed per location, therefore a good policy is to request all scenarios (e.g. SSPs)
        and years likely to be needed and cache against future need. This is the default but can be turned off via:
        only_request_required.

        Args:
            cache_store (H3BasedCache): Results caching store.
            credentials (Optional[CredentialsProvider], optional): Credentials provider. Defaults to None.
            geocoder (Optional[Geocoder], optional): Geocoder (needed to identify country in current JBA API version).
                Defaults to None.
            max_requests (int, optional): Maximum number of requests permitted; if exceeded an exception is raised.
                Defaults to 5.
            cmip (int, optional): Can take values 5 (CMIP5, i.e. RCPs) or 6 (CMIP6, i.e. SSPs).
                Defaults to 6.
            batch_size (int, optional): Number of spatial locations in each JBA API call.
                Defaults to 100.
            restrict_coverage (bool, optional): If True, restrict the number of scenarios for performance reasons.
            only_request_required (bool, optional): If True, request only scenarios and years required (i.e. not extra ones to cache)
            default_buffer (int, optional): Default buffer in metres to apply around points if no geometry provided. Defaults to 10.
            backfill_storm_surge_slr (bool, optional): If True, backfill STSU_U (storm surge) for future
                scenarios via JBA's GMSLR API, since the main flood-depths API does not yet return it for
                future climates. Interim measure: remove once JBA populate STSU_U for future scenarios
                directly. Defaults to True.
        """
        self.cache_store = cache_store
        self.credentials = (
            credentials if credentials is not None else EnvCredentialsProvider()
        )
        self.default_buffer = default_buffer
        self.geocoder = geocoder if geocoder is not None else Geocoder.instance()
        self.indicators = set(
            [
                Indicator(hazard_type="RiverineInundation", indicator_id="flood_depth"),
                Indicator(hazard_type="PluvialInundation", indicator_id="flood_depth"),
                Indicator(hazard_type="CoastalInundation", indicator_id="flood_depth"),
                Indicator(hazard_type="RiverineInundation", indicator_id="flood_sop"),
                Indicator(hazard_type="PluvialInundation", indicator_id="flood_sop"),
                Indicator(hazard_type="CoastalInundation", indicator_id="flood_sop"),
            ]
        )
        self.year_ranges = {
            2030: "2016-2045",
            2040: "2026-2055",
            2050: "2036-2065",
            2080: "2066-2095",
            2100: "2086-2115",
        }
        # median global mean sea level rise, in metres relative to a 1995-2014-ish
        # baseline, by (scenario, year). Used to backfill STSU_U (storm surge) for future
        # scenarios by linear interpolation across the GMSLR buckets - see the comment
        # above _gmslr_interpolation_weights.
        self._gmslr_mapping: Dict[Tuple[str, int], float] = {
            ("ssp126", 2030): 0.09,
            ("ssp245", 2030): 0.09,
            ("ssp585", 2030): 0.1,
            ("ssp126", 2050): 0.19,
            ("ssp245", 2050): 0.2,
            ("ssp585", 2050): 0.23,
            ("ssp126", 2080): 0.34,
            ("ssp245", 2080): 0.4,
            ("ssp585", 2080): 0.51,
        }
        # for any API call we request these 15 scenarios
        self.pillar_years = [
            2030,
            2050,
            2080,
        ]  # we could have included 2040 and 2100, but removed for better performance
        self.historical_year = 2025
        self.lock = Lock()
        self.max_requests = max_requests
        self.cmip = cmip
        self.batch_size = batch_size
        self.jba_scenarios = [
            self.jba_scenario(s, y)
            for y in self.pillar_years
            for s in (
                ["ssp245", "ssp585"]
                if restrict_coverage
                else ["ssp126", "ssp245", "ssp585"]
            )
        ]
        self.restrict_coverage = restrict_coverage
        self.only_request_required = only_request_required  # here in case needed in future. If true, only scenarios that are explicitly asked for are
        # included in API request; improves performance.
        self.backfill_storm_surge_slr = backfill_storm_surge_slr
        # interim storm-surge backfill: maps each jba_scenario string (e.g. "ssp585_2066-2095")
        # to linear interpolation weights across the GMSLR buckets that bracket its expected
        # sea level rise (self._gmslr_mapping). gmslr_buckets is the small fixed set of
        # buckets that ever need fetching. See the comment above _gmslr_interpolation_weights.
        self._jba_scenario_to_gmslr_weights: Dict[str, List[Tuple[str, float]]] = {
            self.jba_scenario(scenario, year): self._gmslr_interpolation_weights(slr_m)
            for (scenario, year), slr_m in self._gmslr_mapping.items()
        }
        self.gmslr_buckets: List[str] = sorted(
            {
                bucket
                for weights in self._jba_scenario_to_gmslr_weights.values()
                for bucket, _ in weights
            }
        )

    def check_requests(self, requests: Sequence[HazardDataRequest]):
        if any(
            r
            for r in requests
            if Indicator(
                hazard_type=r.hazard_type.__name__, indicator_id=r.indicator_id
            )
            not in self.indicators
        ):
            raise ValueError("invalid request")

    def get_hazard_data(
        self, requests: Sequence[HazardDataRequest]
    ) -> Mapping[HazardDataRequest, HazardDataResponse]:
        # noqa:C90
        with self.lock:
            # why don't we want this to be accessed by more than one thread at the same time?
            # 1) JBA API returns Riverine and Pluvial hazards at the same time. We want to make sure that
            # we get these and cache just once: otherwise a risk that we request the same data multiple times!
            # 2) We are already maxing out the number of requests to JBA using async for a single thread.
            if not self.geocoder:
                self.geocoder = Geocoder()
            cache_key_country: Dict[JBACacheKey, str] = {}  # cache item to requests
            request_groups: Dict[RequestKey, List[JBACacheKey]] = defaultdict(
                list
            )  # request to cache items
            self.check_requests(requests)
            # some deviations from 2-letter country codes:
            country_mapping = {
                "AU": "AUC",  # Australian model including coastal inundation
                "ES-ML": "ES",  # Melilla as ES
                "FR": "FR5C",  # France 5m model including coastal inundation
                "GG": "GB",  # Guernsey uses GB map
                "HK": "CN",
                "IE": "IE30",
                "JE": "FR5C",  # Jersey uses France map
                "MC": "FR5C",  # Monaco uses France map
                "NI": "NIC",  # Nicaragua uses NIC
                "US": "US5",  # US 5 m model
            }
            result: MutableMapping[HazardDataRequest, HazardDataResponse] = {}
            # group requests by common location
            requests_by_location: Dict[str, List[HazardDataRequest]] = defaultdict(list)
            all_years: set[int] = set()
            for item in requests:
                spatial_key = self.cache_store.spatial_key(
                    item.latitude, item.longitude, item.geometry
                )
                requests_by_location[spatial_key].append(item)
                if item.scenario != "historical":
                    all_years.add(item.year)
            # JBA requires a 2-letter country code per request (at time of writing), so it
            # is normally necessary to geocode - except when the storm-surge backfill is
            # enabled, in which case we use JBA's worldwide model ("WR") for every location
            # instead, skipping geocoding entirely.
            if self.backfill_storm_surge_slr:
                countries = ["WR"] * len(requests_by_location)
            else:
                lats, lons = (
                    [r[0].latitude for r in requests_by_location.values()],
                    [r[0].longitude for r in requests_by_location.values()],
                )
                countries = [
                    country_mapping.get(c, c)
                    for c in self.geocoder.get_countries(lats, lons)
                ]
            # note a single cache entry can provide information for multiple requests, because each entry contains
            # information about different hazards, or because points are close.
            # for interpolation, the list of pillar years for different requested years is calculated
            # ahead of time: e.g. 2036 needs 2030 and 2050 pillars.
            requested_years = sorted(list(all_years))
            weights = HazardDataProvider._weights(
                "ssp", self.pillar_years, requested_years, self.historical_year
            )
            weights_histo = HazardDataProvider._weights(
                "historical", self.pillar_years, requested_years, self.historical_year
            )
            pillar_years_lookup = {k.year: v for k, v in weights.items()}
            pillar_years_lookup[-1] = weights_histo[ScenarioYear("historical", -1)]
            cache_keys: set[JBACacheKey] = (
                set()
            )  # the set of cache keys to be requested (spatial key, year and scenario)
            req_weights_set: List[
                RequestWeights
            ] = []  # for each request the linear combination of cache keys required (interpolating)
            for reqs, country in zip(requests_by_location.values(), countries):
                for req in reqs:
                    req_weights: List[Tuple[JBACacheKey, float]] = []
                    for weight in pillar_years_lookup[
                        -1 if req.scenario == "historical" else req.year
                    ].weights:
                        cache_key = JBACacheKey(
                            jba_scenario=self.jba_scenario(
                                req.scenario, weight[0].year, country
                            ),
                            spatial_key=self.cache_store.spatial_key(
                                req.latitude, req.longitude, req.geometry
                            ),
                        )
                        cache_key_country[cache_key] = country
                        cache_keys.add(cache_key)
                        req_weights.append((cache_key, weight[1]))
                    req_weights_set.append(RequestWeights(req, req_weights))
            # requests are grouped by country
            for cache_key in cache_keys:
                country_code = cache_key_country[cache_key]
                request_groups[RequestKey(country_code=country_code)].append(cache_key)
            access_token = self.credentials.jba_access_key()
            api_requests: List[APIRequest] = []
            # contains the raw results for all cache keys, first populated by looking in cache
            # and then by making API calls if needed.
            cached_responses: Dict[JBACacheKey, Dict] = {}
            for request_key, group_cache_keys in request_groups.items():
                # process anything that can be sourced from the cache and identify extra API requests needed
                group_cached_responses, api_requests_batch = (
                    self._identify_api_requests(
                        request_key, group_cache_keys, requests_by_location
                    )
                )
                cached_responses.update(group_cached_responses)
                api_requests.extend(api_requests_batch)
            # if there are extra API requests, make these (in parallel) and process to get results
            n_requests = len([k for r in api_requests for k in r.spatial_keys])
            logger.info(f"{n_requests} API requests total")
            batches = [len(r.spatial_keys) for r in api_requests]
            logger.info(f"{len(api_requests)} batches of requests of size ({batches})")
            if n_requests > self.max_requests:
                raise ValueError(
                    f"would make {n_requests} requests to JBA API, more than {self.max_requests} maximum. "
                    "provider_max_requests may be set incorrectly."
                )
            cached_responses.update(
                self._process_api_requests(api_requests, access_token)
            )
            for req_weight in req_weights_set:
                resps = [
                    self._process_response(
                        req_weight.request, cached_responses.get(key, {"stats": None})
                    )
                    for key, _ in req_weight.weights
                ]
                if len(resps) == 1:
                    result[req_weight.request] = resps[0]
                elif len(resps) == 2:
                    weight0, weight1 = (
                        req_weight.weights[0][1],
                        req_weight.weights[1][1],
                    )
                    if isinstance(resps[0], HazardEventDataResponse) and isinstance(
                        resps[1], HazardEventDataResponse
                    ):
                        result[req_weight.request] = HazardEventDataResponse(
                            resps[0].return_periods,
                            resps[0].intensities * weight0
                            + resps[1].intensities * weight1,
                            units="m",
                            path="jba",
                        )
                    elif isinstance(
                        resps[0], HazardParameterDataResponse
                    ) and isinstance(resps[1], HazardParameterDataResponse):
                        result[req_weight.request] = HazardParameterDataResponse(
                            resps[0].parameters * weight0
                            + resps[1].parameters * weight1,
                            resps[0].param_defns,
                            units=resps[0].units,
                            path="jba",
                        )
                    else:
                        result[req_weight.request] = HazardDataFailedResponse(
                            ValueError("no data returned")
                        )
            failures = [
                r for r in result.values() if isinstance(r, HazardDataFailedResponse)
            ]
            if any(failures):
                logger.error(
                    f"{len(failures)} errors in JBA batch (logs limited to first 3)"
                )
                errors = (str(i.error) for i in failures)
                for _ in range(min(len(failures), 3)):
                    logger.error(next(errors))
            return result

    def jba_cache_id(self, key: JBACacheKey):
        # for dealing with buffer > 10 m, we have two options
        # 1) Change the spatial key resolution to match the buffer size
        # 2) Add a buffer part to the key
        return f"jba/{key.jba_scenario}/{key.spatial_key}"

    def jba_request_id(self, spatial_key: str):
        # requests to JBA API are made for all scenarios
        return f"jba/{spatial_key}"

    def jba_scenario(self, scenario: str, year: int, country: str = ""):
        if scenario == "historical":
            return "historical"
        if self.cmip == 5:
            if scenario == "rcp2p6" or scenario == "ssp126":
                prefix = "rcp26"
            elif scenario == "rcp8p5" or scenario == "ssp585":
                prefix = "rcp85"
            elif scenario == "rcp45" or scenario == "ssp245":
                prefix = "rcp45"
            else:
                raise ValueError(
                    f"scenario {scenario} not supported by JBA Risk Management API"
                )
        else:
            if scenario not in ["ssp126", "ssp245", "ssp585"]:
                raise ValueError(
                    f"scenario {scenario} not supported by JBA Risk Management API"
                )
            prefix = scenario
        if self.cmip == 6:
            try:
                range = self.year_ranges[year]
            except Exception:
                raise ValueError(
                    f"scenario {scenario} not supported by JBA Risk Management API"
                )
        else:
            gb_code = country in ["GB", "NI", "ROI"] and "rcp" in scenario
            if year == 2030:
                range = "2031-2035" if gb_code else "2016-2045"
            elif year == 2040:
                range = "2041-2045" if gb_code else "2026-2055"
            elif year == 2050:
                range = "2051-2055" if gb_code else "2036-2065"
            elif year == 2080:
                range = "2081-2085" if gb_code else "2066-2095"
            else:
                raise ValueError(
                    f"scenario {scenario} not supported by JBA Risk Management API"
                )
        return prefix + "_" + range

    # --- storm surge (STSU_U) backfill for future climates --------------------------------
    # JBA's main flood-depths API does not currently return STSU_U (storm surge) for future
    # scenarios, only baseline/historical. In the interim we backfill it from JBA's separate
    # GMSLR API (see storm_surge_slr below), which returns results for a single global mean
    # sea level rise bucket rather than a scenario/year - so for a given scenario/year we
    # linearly interpolate between the two GMSLR buckets (self._GMSLR_BUCKET_METRES) that
    # bracket its expected sea level rise (self._gmslr_mapping). This whole block - this
    # method, storm_surge_slr, _merge_storm_surge_slr, _interpolate_storm_surge_stats, the
    # backfill_storm_surge_slr flag, and their use in get_hazard_data - can be deleted once
    # JBA populate STSU_U for future scenarios directly in the main API.
    def _gmslr_interpolation_weights(self, slr_m: float) -> List[Tuple[str, float]]:
        """Linear interpolation weights across the GMSLR buckets (0.1m/0.2m/0.4m/0.8m) for
        a given global mean sea level rise in metres. Clamps to the nearest bucket if slr_m
        is outside the 0.1-0.8m range.
        """
        buckets = self._GMSLR_BUCKET_METRES
        if slr_m <= buckets[0][1]:
            return [(buckets[0][0], 1.0)]
        if slr_m >= buckets[-1][1]:
            return [(buckets[-1][0], 1.0)]
        for (bucket_lo, m_lo), (bucket_hi, m_hi) in zip(buckets, buckets[1:]):
            if m_lo <= slr_m <= m_hi:
                weight_hi = (slr_m - m_lo) / (m_hi - m_lo)
                return [(bucket_lo, 1 - weight_hi), (bucket_hi, weight_hi)]
        return [(buckets[-1][0], 1.0)]  # unreachable given the checks above

    async def _call_jba_flood_api(
        self,
        api_request: APIRequest,
        access_token: str,
        session: aiohttp.ClientSession,
        req_ids: List[str],
        params: Optional[Dict[str, str]],
        log_label: str,
    ):
        """Shared POST + response-status handling for JBA's flood-depths-style endpoints,
        used by both flood_depth and storm_surge_slr (which differ only in what
        api_request.country_code means, whether CSTHs/baseline params are sent, and how
        response items map back to cache keys - handled by each caller).

        Returns the parsed JSON response (a list of per-location items) on success (HTTP
        200), or a string describing the error - callers treat a string return as a
        retryable failure. Raises ValueError on authentication failure (401/403), since
        that's not something a per-location retry can fix.
        """
        if self.credentials.jba_api_disabled():
            logger.error("JBA requests made but API calls disabled")
            raise ValueError("JBA requests made but API calls disabled")
        country_code = api_request.country_code  # e.g. CN, FR, or a GMSLR bucket
        # https://api.jbarisk.com/docs/1.2/index.html
        url = "https://api.jbarisk.com/flooddepths/" + country_code
        request = {
            "country_code": country_code,
            "geometries": [
                {
                    "id": id,
                    "wkt_geometry": (
                        f"POINT({lon} {lat})" if geom is None else geom.wkt
                    ),
                    "buffer": (self.default_buffer if geom is None else 0),
                }
                for id, lat, lon, geom in zip(
                    req_ids,
                    api_request.latitudes,
                    api_request.longitudes,
                    api_request.geometries,
                )
            ],
        }
        logger.debug(f"{log_label} request URL: " + url)
        logger.debug(f"{log_label} request payload: " + json.dumps(request))
        headers = {"Authorization": f"Basic {access_token}"}
        proxies = self.credentials.proxies()
        response_dict = None
        status = None
        try:
            async with session.post(
                url=url,
                json=request,
                params=params,
                proxy=proxies["https"],
                headers=headers,  # , ssl=False can be used *in dev* if SSL verify issue
            ) as response:
                # capture status before attempting to parse the body: auth failures often
                # come back with a non-JSON body, and we still need the status in that case.
                status = response.status
                try:
                    response_dict = await response.json()
                    logger.debug(f"{log_label} response: " + json.dumps(response_dict))
                except Exception:
                    logger.exception(f"{log_label} response body was not valid JSON")
        except Exception:
            # network or proxy errors, or failure to establish the response, land here,
            # hence use of logger.exception to ensure exception info is included.
            logger.exception(f"{log_label} raised exception")
            return (
                f"{log_label} request failed"
                if response_dict is None
                else str(response_dict)
            )
        if status in (401, 403):
            # not something a per-location retry can fix: fail loudly rather than
            # silently retrying every location and reporting misleading "no data" results.
            logger.error(f"{log_label} authentication failed (status {status})")
            raise ValueError(
                f"{log_label} authentication failed (status {status}); check credentials"
            )
        if response_dict is None:
            logger.error(
                f"{log_label} response status {status} but body was not valid JSON"
            )
            return f"{log_label} request failed (status {status})"
        if status != 200:
            logger.error(f"{log_label} response status {status}")
            return str(response_dict)
        return response_dict

    async def flood_depth(
        self, api_request: APIRequest, access_token: str, session: aiohttp.ClientSession
    ):
        if len(api_request.spatial_keys) == 0:
            return {}
        # scenarios actually being requested from the API for this batch: used both to build the
        # CSTHs request parameter and to know what keys to expect back in the response, so that the
        # two stay in sync (e.g. under only_request_required, which narrows the scenario set).
        requested_scenarios = (
            self.jba_scenarios
            if not self.only_request_required
            else list(
                set(
                    k.jba_scenario
                    for v in api_request.location_cache_keys.values()
                    for k in v
                    if k.jba_scenario != "historical"  # requested via "baseline" param
                )
            )
        )
        req_id_to_keys: Dict[str, List[JBACacheKey]] = {
            self.jba_request_id(k): [
                JBACacheKey(s, k) for s in (["historical"] + requested_scenarios)
            ]
            for k in api_request.spatial_keys
        }
        req_ids = list(req_id_to_keys.keys())
        response = await self._call_jba_flood_api(
            api_request,
            access_token,
            session,
            req_ids,
            params={"CSTHs": ",".join(requested_scenarios), "baseline": "true"},
            log_label="JBA flood-depth",
        )
        if isinstance(response, str):
            return response
        try:
            # we expect results for all requested_scenarios and "stats"
            result = {}
            for item in response:
                for cache_key in req_id_to_keys[item["id"]]:
                    key = (
                        "stats"
                        if cache_key.jba_scenario == "historical"
                        else cache_key.jba_scenario
                    )
                    result[cache_key] = {"stats": item[key]}
            return result
        except Exception:
            ids = ",".join(req_ids)
            logger.error(
                f"Unexpected flood-depth response for {api_request.country_code} "
                f"(request IDs: {ids})"
            )
            # no logger.exception here - we assume useful info is in the response.
            return str(response)

    async def storm_surge_slr(
        self, api_request: APIRequest, access_token: str, session: aiohttp.ClientSession
    ):
        """Backfill for STSU_U (storm surge) under future climates: calls JBA's GMSLR API,
        which returns results for a single global mean sea level rise bucket (encoded via
        api_request.country_code, e.g. "GMSLR4", instead of a real country) rather than a
        scenario/year - so there is one cache key per location instead of one per scenario,
        and no CSTHs/baseline params. Part of the interim storm-surge backfill - see the
        comment above _gmslr_interpolation_weights.
        """
        if len(api_request.spatial_keys) == 0:
            return {}
        bucket = api_request.country_code  # e.g. "GMSLR4"
        req_id_to_key: Dict[str, JBACacheKey] = {
            self.jba_request_id(k): JBACacheKey(bucket, k)
            for k in api_request.spatial_keys
        }
        req_ids = list(req_id_to_key.keys())
        response = await self._call_jba_flood_api(
            api_request,
            access_token,
            session,
            req_ids,
            params=None,
            log_label="JBA GMSLR",
        )
        if isinstance(response, str):
            return response
        try:
            result = {}
            for item in response:
                cache_key = req_id_to_key[item["id"]]
                result[cache_key] = {"stats": item["stats"]}
            return result
        except Exception:
            ids = ",".join(req_ids)
            logger.error(f"Unexpected GMSLR response for {bucket} (request IDs: {ids})")
            return str(response)

    def _identify_api_requests(
        self,
        request_key: RequestKey,
        cache_keys: Iterable[JBACacheKey],
        requests_by_location: Dict[str, List[HazardDataRequest]],
    ):
        """Process any results that can be sourced from the cache and identify the
        requests to the API that are needed."""
        batches: List[APIRequest] = []
        cache_ids = [self.jba_cache_id(k) for k in cache_keys]
        # first checks cache
        cached_responses: Dict[JBACacheKey, Dict] = {}
        for cache_key, item in zip(cache_keys, self.cache_store.getitems(cache_ids)):
            if item is not None:
                value = json.loads(item)
                if value["stats"] is not None:
                    cached_responses[cache_key] = value
        # we need to create requests for anything not in cache
        # req_keys_all = [k for k in cache_keys if k not in cached_responses]
        location_cache_keys: Dict[str, List[JBACacheKey]] = defaultdict(list)
        missing_cache_keys = [k for k in cache_keys if k not in cached_responses]
        for k in missing_cache_keys:
            location_cache_keys[k.spatial_key].append(k)
        req_keys_all = list(location_cache_keys.keys())
        # but batch up for requesting
        batch_size = self.batch_size
        req_key_batches = [
            req_keys_all[i : min(i + batch_size, len(req_keys_all))]
            for i in range(0, len(req_keys_all), batch_size)
        ]
        for req_keys in req_key_batches:
            first_req = [requests_by_location[k][0] for k in req_keys]
            lats = [r.latitude for r in first_req]
            lons = [r.longitude for r in first_req]
            geoms = [r.geometry for r in first_req]
            batches.append(
                APIRequest(
                    location_cache_keys={k: location_cache_keys[k] for k in req_keys},
                    spatial_keys=req_keys,
                    latitudes=lats,
                    longitudes=lons,
                    geometries=geoms,
                    country_code=request_key.country_code,
                )
            )
        return cached_responses, batches

    def _process_api_requests(
        self,
        api_requests: Sequence[APIRequest],
        access_token: str,
        concurrent_requests: int = 8,
    ):
        """Make required requests to the API, updating cache and process responses."""
        check_total = 0
        # the code is async in order to run a large number of API requests in parallel
        # and may be called from within a thread-pool
        # we could create a new event loop, e.g. via (later closing loop):
        # loop = asyncio.new_event_loop()
        # loop.run_until_complete(gather_requests(api_requests))
        # but we prefer to use a single loop in away analagous to accessing Zarr data
        # which uses the AsyncFileSystem of fsspec.
        loop = get_loop()
        cached_responses = {}
        with aiohttp.TCPConnector(
            limit_per_host=concurrent_requests, loop=loop
        ) as conn:
            reruns: List[APIRequest] = []

            async def gather_requests(api_requests: Sequence[APIRequest]):
                semaphore = asyncio.Semaphore(concurrent_requests)
                async with aiohttp.ClientSession(
                    connector=conn, connector_owner=False
                ) as session:

                    async def request_single(request: APIRequest):
                        nonlocal check_total
                        async with semaphore:
                            # interim storm-surge backfill (see _gmslr_interpolation_weights):
                            # fetch the main flood-depth data and all GMSLR bucket data for this batch's
                            # locations concurrently, then merge before caching. The enriched
                            # result is cached as a normal flood-depth entry afterwards, so no
                            # separate GMSLR cache is needed.
                            slr_requests = (
                                [
                                    APIRequest(
                                        spatial_keys=request.spatial_keys,
                                        latitudes=request.latitudes,
                                        longitudes=request.longitudes,
                                        geometries=request.geometries,
                                        country_code=bucket,
                                        location_cache_keys={},
                                    )
                                    for bucket in self.gmslr_buckets
                                ]
                                if self.backfill_storm_surge_slr
                                else []
                            )
                            responses, *slr_batches = await asyncio.gather(
                                self.flood_depth(request, access_token, session),
                                *(
                                    self.storm_surge_slr(r, access_token, session)
                                    for r in slr_requests
                                ),
                            )
                            check_total += len(request.spatial_keys)
                            if isinstance(responses, str):
                                # a string indicates an error
                                reruns.append(request)
                            else:
                                if slr_batches:
                                    self._merge_storm_surge_slr(responses, slr_batches)
                                self.cache_store.setitems(
                                    {
                                        self.jba_cache_id(k): json.dumps(v)
                                        for k, v in responses.items()
                                    }
                                )
                                cached_responses.update(responses)
                            # if check_total // 500 != (check_total - len(request.spatial_keys)) // 500:
                            #    logger.info(
                            #        f"Total of {check_total} spatial location requests made"
                            #    )

                    await asyncio.gather(*(request_single(req) for req in api_requests))

            def run_checked(coro):
                # run() reports coroutine exceptions by returning the exception
                # instance rather than raising it; re-raise here so that e.g.
                # authentication failures in flood_depth fail loudly instead of
                # being silently dropped.
                result = run(coro, loop)
                if isinstance(result, Exception):
                    raise result
                return result

            run_checked(gather_requests(api_requests))
            # for failed batches we run again, but as single requests
            # this is needed because the JBA API will fail for all locations
            # if a single one is out of bounds (e.g. off-shore wind farm)
            single_api_requests = [
                APIRequest(
                    spatial_keys=[spatial_key],
                    latitudes=[lat],
                    longitudes=[lon],
                    geometries=[geometry],
                    country_code=rerun.country_code,
                    location_cache_keys={
                        spatial_key: rerun.location_cache_keys[spatial_key]
                    },
                )
                for rerun in reruns
                for spatial_key, lat, lon, geometry in zip(
                    rerun.spatial_keys,
                    rerun.latitudes,
                    rerun.longitudes,
                    rerun.geometries,
                )
            ]
            if len(single_api_requests) > 0:
                run_checked(gather_requests(single_api_requests))
            logger.info(f"Check: {check_total} requests made")
            logger.info(f"Check: {len(single_api_requests)} reruns")
        return cached_responses

    def _merge_storm_surge_slr(
        self, responses: Dict[JBACacheKey, Dict], slr_batches: List
    ):
        """Backfills STSU_U into responses (in place) from freshly-fetched slr_batches, for
        any cache key that doesn't already have it populated. STSU_U is linearly
        interpolated across the GMSLR buckets that bracket the cache key's expected sea
        level rise - see the comment above _gmslr_interpolation_weights. The enriched
        responses are cached as normal flood-depth entries afterwards, so no separate
        GMSLR cache is needed.
        """
        slr_stats_by_key: Dict[JBACacheKey, Dict] = {}
        for slr_batch in slr_batches:
            if isinstance(slr_batch, str):
                logger.error(f"GMSLR backfill request failed: {slr_batch}")
                continue
            for slr_key, value in slr_batch.items():
                stats = (value or {}).get("stats") or {}
                if "STSU_U" in stats:
                    slr_stats_by_key[slr_key] = stats["STSU_U"]
        for cache_key, resp in responses.items():
            weights = self._jba_scenario_to_gmslr_weights.get(cache_key.jba_scenario)
            if not weights:
                continue
            stats = resp.get("stats") or {}
            if stats.get("STSU_U"):
                continue  # JBA already returned it - nothing to backfill
            weighted_stsu = [
                (slr_stats_by_key[slr_key], weight)
                for bucket, weight in weights
                if (slr_key := JBACacheKey(bucket, cache_key.spatial_key))
                in slr_stats_by_key
            ]
            if len(weighted_stsu) != len(weights):
                continue  # a bracketing bucket failed to fetch - skip rather than guess
            stats["STSU_U"] = self._interpolate_storm_surge_stats(weighted_stsu)
            resp["stats"] = stats

    def _interpolate_storm_surge_stats(
        self, weighted_stats: List[Tuple[Dict, float]]
    ) -> Dict:
        """Linearly combines one or two STSU_U stats blocks (each e.g. {"rp_20": {"max20":
        0.5, ...}, ...}) according to the given weights. Part of the interim storm-surge
        backfill - see the comment above _gmslr_interpolation_weights.
        """
        if len(weighted_stats) == 1:
            return weighted_stats[0][0]
        rp_keys = set().union(*(stats.keys() for stats, _ in weighted_stats))
        combined: Dict[str, Dict[str, float]] = {}
        for rp_key in rp_keys:
            fields = set().union(
                *(stats.get(rp_key, {}).keys() for stats, _ in weighted_stats)
            )
            combined[rp_key] = {
                field: sum(
                    stats.get(rp_key, {}).get(field, 0) * weight
                    for stats, weight in weighted_stats
                )
                for field in fields
            }
        return combined

    def _process_response(self, request: HazardDataRequest, response: Dict):
        if request.hazard_type == RiverineInundation:
            tag = "FLRF_U"
            path = "jba_riverine"
        elif request.hazard_type == PluvialInundation:
            tag = "FLSW_U"
            path = "jba_pluvial"
        elif request.hazard_type == CoastalInundation:
            tag = "STSU_U"
            path = "jba_coastal"
        else:
            raise ValueError("unexpected hazard type")
        if response["stats"] is None:
            return HazardDataFailedResponse(ValueError("no data returned"))
        elif request.indicator_id == "flood_sop":
            sop = response["stats"].get(tag, {}).get("sop", 0)
            return HazardParameterDataResponse(
                [sop, sop], units="years", path=path
            )  # min and max: in this case just a single value
        elif request.indicator_id == "flood_depth":
            return_periods: List[float] = []
            intens: List[float] = []
            for key, value in response["stats"].get(tag, {}).items():
                assert isinstance(key, str)
                if key.startswith("rp_"):
                    return_periods.append(float(key[3:]))
                    intens.append(value["max" + key[3:]])
            return HazardEventDataResponse(
                np.array(return_periods),
                np.array(intens),
                units="m",
                path=path,
            )
        else:
            raise NotImplementedError()

    # scenarios = self.climate_change_scenarios("CN", access_token)
    # def climate_change_scenarios(self, country_code: str, access_token: str):
    #     url = f"https://api.jbarisk.com/flooddepths/ccscenarios/{country_code}"
    # ...
