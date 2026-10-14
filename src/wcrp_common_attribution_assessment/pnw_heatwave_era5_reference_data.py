from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr

# Configuration
ERA5_SINGLE_LEVELS_URL = (
    "https://data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr"
)
ERA5_PRESSURE_LEVELS_URL = "https://data.earthdatahub.destine.eu/era5/era5-pressure-levels-v0.zarr"
OUTPUT_DIR = Path("/network/group/aopp/predict/AWH018_LEACH_ASEASON/wcrp_ceaa/pnw_heatwave")

# Bounds use descending latitude and the ERA5 0-360 longitude convention.
PNW_LATITUDE_BOUNDS = (52, 45)
PNW_LONGITUDE_BOUNDS = (237, 241)
ANCILLARY_LATITUDE_BOUNDS = (70, 30)
ANCILLARY_LONGITUDE_BOUNDS = (180, 300)

CLIMATOLOGY_MONTHS = [6, 7, 8]
CLIMATOLOGY_DATE_RANGE = ("1991", "2020")
EVENT_DATE_RANGE = ("2021-06-26", "2021-06-30")
RETRIEVAL_DATE = date.today().isoformat()

print("Opening ERA5 dataset...", flush=True)
# To ensure a common dataset source is used, we use the DestinE Earth Data Hub ARCO zarr.
# Note that to be able to access this dataset, you will need to set up an Earthdata login
# and configure your environment. See the DestinE Earth Data Hub documentation for details.
era5 = xr.open_dataset(
    ERA5_SINGLE_LEVELS_URL,
    storage_options={"client_kwargs": {"trust_env": True}},
    chunks={},
    engine="zarr",
)

## REFERENCE SERIES ##

print("Computing domain-mean daily maximum 2 m temperatures for the PNW region...", flush=True)
TX = (
    era5.mx2t.sel(  # select maximum temperature variable
        latitude=slice(*PNW_LATITUDE_BOUNDS),
        longitude=slice(*PNW_LONGITUDE_BOUNDS),
    )  # select chosen PNW domain
    .resample(valid_time="1D")
    .max()  # resample to daily maximum at each gridpoint
    .weighted(np.cos(np.deg2rad(era5.latitude.sel(latitude=slice(*PNW_LATITUDE_BOUNDS)))))
    .mean(dim=["latitude", "longitude"])  # average over the domain
    .compute()  # trigger computation and load into memory
)

print("Computing annual maxima from the daily time series...", flush=True)
TXx = TX.resample(valid_time="1YE").max()

TXx.attrs.update(
    {
        "source_dataset": "ERA5 reanalysis, single levels",
        "source_variable": "mx2t (maximum temperature at 2 metres)",
        "source_url": ERA5_SINGLE_LEVELS_URL,
        "retrieval_date": RETRIEVAL_DATE,
        "processing": (
            "Selected the PNW domain (45-52 N, 119-123 W), resampled to daily maxima, "
            "computed a cosine-latitude-weighted spatial mean, and resampled to annual maxima."
        ),
        "spatial_domain": "Pacific Northwest: 45-52 N, 119-123 W",
        "spatial_weighting": "Cosine of latitude",
        "temporal_aggregation": "Annual maximum of daily maximum 2 m temperature",
        "processing_notes": (
            "Data were loaded from the DestinE Earth Data Hub ARCO Zarr store using xarray."
        ),
    }
)

TXx.sel(valid_time=slice("1940", "2025")).rename(valid_time="time").to_netcdf(
    OUTPUT_DIR / "ERA5_reference_mx2t.nc"
)

## ANCILLARY DATA ##

# SURFACE VARIABLES #

era5_t3_domain = era5.sel(  # select maximum temperature variable
    latitude=slice(*ANCILLARY_LATITUDE_BOUNDS),
    longitude=slice(*ANCILLARY_LONGITUDE_BOUNDS),
)  # select chosen PNW domain

era5_t3_domain_clim = (
    era5_t3_domain.sel(valid_time=era5_t3_domain.valid_time.dt.month.isin(CLIMATOLOGY_MONTHS))
    .sel(valid_time=slice(*CLIMATOLOGY_DATE_RANGE))
    .mean(dim="valid_time")
)  # compute climatology for June-August 1991-2020

era5_t3_domain_event = (
    era5_t3_domain.sel(valid_time=slice(*EVENT_DATE_RANGE))
    .resample(valid_time="1D")
    .mean(dim="valid_time")
)  # compute mean for the event period

era5_t3_domain_clim = era5_t3_domain_clim.get(["t2m", "msl", "tcc", "tcwv", "swvl1"]).compute()

era5_t3_domain_event = era5_t3_domain_event.get(["t2m", "msl", "tcc", "tcwv", "swvl1"]).compute()

era5_t3_domain_common_attrs = {
    "source_dataset": "ERA5 reanalysis, single levels",
    "source_variables": "t2m, msl, tcc, tcwv, swvl1",
    "source_url": ERA5_SINGLE_LEVELS_URL,
    "retrieval_date": RETRIEVAL_DATE,
    "spatial_domain": "30-70 N, 180-300 E (180 W-60 W)",
    "spatial_weighting": "None; variables retained on the native ERA5 grid",
    "processing_notes": (
        "Data were loaded from the DestinE Earth Data Hub ARCO Zarr store using xarray."
    ),
}

era5_t3_domain_clim.attrs.update(
    {
        **era5_t3_domain_common_attrs,
        "temporal_aggregation": "Mean over June-August for 1991-2020",
        "processing": (
            "Selected the 30-70 N, 180-300 E domain, retained June-August 1991-2020, "
            "computed the mean over valid_time, selected t2m, msl, tcc, tcwv, "
            "and swvl1, "
            "and loaded the result into memory."
        ),
    }
)

era5_t3_domain_event.attrs.update(
    {
        **era5_t3_domain_common_attrs,
        "temporal_aggregation": "Daily mean over 2021-06-26 to 2021-06-30",
        "processing": (
            "Selected the 30-70 N, 180-300 E domain and the 2021-06-26 to "
            "2021-06-30 event period, resampled to daily means, selected t2m, "
            "msl, tcc, tcwv, and swvl1, and loaded the result "
            "into memory."
        ),
    }
)

era5_t3_domain_clim.to_netcdf(OUTPUT_DIR / "ERA5_reference_T3_sfc_climate.nc")
era5_t3_domain_event.rename(valid_time="time").to_netcdf(OUTPUT_DIR / "ERA5_reference_T3_sfc.nc")

# PRESSURE-LEVEL VARIABLES #

print("Opening ERA5 pressure-level dataset...", flush=True)
era5_pl = xr.open_dataset(
    ERA5_PRESSURE_LEVELS_URL,
    storage_options={"client_kwargs": {"trust_env": True}},
    chunks={},
    engine="zarr",
)

print("Selecting 500 hPa geopotential over the domain...", flush=True)
era5_z500_domain = era5_pl.z.sel(
    isobaricInhPa=500,
    latitude=slice(*ANCILLARY_LATITUDE_BOUNDS),
    longitude=slice(*ANCILLARY_LONGITUDE_BOUNDS),
)

print("Computing 1991-2020 climate and 5-day event daily averages...", flush=True)
era5_z500_domain_clim = (
    era5_z500_domain.sel(valid_time=era5_z500_domain.valid_time.dt.month.isin(CLIMATOLOGY_MONTHS))
    .sel(valid_time=slice(*CLIMATOLOGY_DATE_RANGE))
    .mean(dim="valid_time")
)

era5_z500_domain_event = (
    era5_z500_domain.sel(valid_time=slice(*EVENT_DATE_RANGE))
    .resample(valid_time="1D")
    .mean(dim="valid_time")
)

era5_z500_domain_clim = era5_z500_domain_clim.compute()
era5_z500_domain_event = era5_z500_domain_event.compute()

era5_z500_domain_common_attrs = {
    "source_dataset": "ERA5 reanalysis, pressure levels",
    "source_variable": "z at 500 hPa (geopotential)",
    "source_url": ERA5_PRESSURE_LEVELS_URL,
    "pressure_level": "500 hPa",
    "retrieval_date": RETRIEVAL_DATE,
    "spatial_domain": "30-70 N, 180-300 E (180 W-60 W)",
    "spatial_weighting": "None; geopotential retained on the native ERA5 grid",
    "processing_notes": (
        "Data were loaded from the DestinE Earth Data Hub pressure-level ARCO Zarr "
        "store using xarray. The z variable is retained in native ERA5 geopotential "
        "units; no conversion to geopotential height "
        "in metres was applied."
    ),
}

era5_z500_domain_clim.attrs.update(
    {
        **era5_z500_domain_common_attrs,
        "temporal_aggregation": "Mean over June-August for 1991-2020",
        "processing": (
            "Selected z at 500 hPa over the 30-70 N, 180-300 E domain, "
            "retained June-August 1991-2020, "
            "and computed the mean over valid_time."
        ),
    }
)

era5_z500_domain_event.attrs.update(
    {
        **era5_z500_domain_common_attrs,
        "temporal_aggregation": "Daily mean over 2021-06-26 to 2021-06-30",
        "processing": (
            "Selected z at 500 hPa over the 30-70 N, 180-300 E domain and the "
            "2021-06-26 to 2021-06-30 "
            "event period, then resampled to daily means."
        ),
    }
)

era5_z500_domain_clim.to_netcdf(OUTPUT_DIR / "ERA5_reference_T3_pl_climate.nc")
era5_z500_domain_event.rename(valid_time="time").to_netcdf(OUTPUT_DIR / "ERA5_reference_T3_pl.nc")
