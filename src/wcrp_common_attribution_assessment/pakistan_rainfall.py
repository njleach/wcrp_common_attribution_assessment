import numpy as np
import regionmask
import xarray as xr

from wcrp_common_attribution_assessment import constants

MM_PER_M = 1000.0


if __name__ == "__main__":
    print("Opening ERA5 dataset...", flush=True)
    # To ensure a common dataset source is used, we use the DestinE Earth Data Hub ARCO zarr.
    # Note that to be able to access this dataset, you will need to set up an Earthdata login
    # and configure your environment. See the DestinE Earth Data Hub documentation for details.
    era5 = xr.open_dataset(
        "https://data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr",
        storage_options={"client_kwargs": {"trust_env": True}},
        chunks={},
        engine="zarr",
    )

    print(
        "Selecting ERA5 total precipitation over the Pakistan rainfall analysis region...",
        flush=True,
    )
    era5_tp = (
        era5.tp.sel(
            latitude=slice(60, -10),
            longitude=slice(50, 130),
            valid_time=slice("1990", "2022"),
        )
        .coarsen(valid_time=24)  # use coarsen for efficiency
        .sum()
    )

    print("Building Pakistan land mask...", flush=True)
    pk_regions = regionmask.defined_regions.natural_earth_v5_1_2.countries_10
    pk_key = pk_regions.map_keys("PK")
    pk_mask = pk_regions.mask(era5_tp.longitude, era5_tp.latitude, flag=None)
    pk_mask = pk_mask == pk_key

    print("Computing Pakistan area-mean daily total precipitation...", flush=True)
    # Convert hourly accumulations to daily totals before deriving the seasonal mean
    # and Rx5d series.
    era5_tp_pk = (
        era5_tp.where(pk_mask)
        .weighted(np.cos(np.deg2rad(era5_tp.latitude)))
        .mean(dim=["latitude", "longitude"])
        .compute()
    )

    print("Computing JJA accumulated precipitation and annual Rx5d precipitation...", flush=True)
    pr_jja = (
        era5_tp_pk.sel(valid_time=era5_tp_pk.valid_time.dt.month.isin([6, 7, 8]))
        .groupby("valid_time.year")
        .sum()
    )
    rx5d = (
        era5_tp_pk.rolling(valid_time=5, center=True, min_periods=5)
        .sum()
        .groupby("valid_time.year")
        .max()
    )

    print(
        "Calculating event definitions for the 2022 Pakistan rainfall JJA accumulated "
        "precipitation...",
        flush=True,
    )
    # 1. Absolute definition
    pr_jja_absolute = pr_jja.sel(year=2022).squeeze()

    # 2. Seasonal mean anomaly definition
    pr_jja_anom = pr_jja_absolute - pr_jja.sel(year=slice(1991, 2020)).mean()

    # 3. Seasonal mean standard deviation definition
    pr_jja_stdev = pr_jja_anom / pr_jja.sel(year=slice(1991, 2020)).std()

    jja_output_lines = [
        (
            "Absolute event definition "
            "(2022 June-August accumulation of the Pakistan area-mean daily total precipitation, "
            "in mm): "
            f"{pr_jja_absolute.item() * MM_PER_M:.2f} mm"
        ),
        (
            "Seasonal-accumulation anomaly event definition "
            "(2022 June-August accumulation minus the 1991-2020 mean June-August Pakistan "
            "area-mean precipitation accumulation, in mm): "
            f"{pr_jja_anom.item() * MM_PER_M:.2f} mm"
        ),
        (
            "Seasonal-accumulation standardized event definition "
            "(seasonal-accumulation anomaly divided by the 1991-2020 standard deviation of "
            "June-August Pakistan area-mean precipitation accumulation): "
            f"{pr_jja_stdev.item():.2f}"
        ),
    ]

    for line in jja_output_lines:
        print(line, flush=True)

    print(
        "Calculating event definitions for the 2022 Pakistan rainfall Rx5d precipitation...",
        flush=True,
    )
    # 1. Absolute definition
    rx5d_absolute = rx5d.sel(year=2022).squeeze()

    # 2. Annual maximum anomaly definition
    rx5d_anom = rx5d_absolute - rx5d.sel(year=slice(1991, 2020)).mean()

    # 3. Annual maximum standard deviation definition
    rx5d_stdev = rx5d_anom / rx5d.sel(year=slice(1991, 2020)).std()

    rx5d_output_lines = [
        (
            "Absolute event definition "
            "(2022 annual maximum of the Pakistan area-mean 5-day precipitation sum, in mm): "
            f"{rx5d_absolute.item() * MM_PER_M:.2f} mm"
        ),
        (
            "Annual-maximum anomaly event definition "
            "(2022 annual maximum minus the 1991-2020 mean annual maximum of the same "
            "Pakistan area-mean 5-day precipitation sum, in mm): "
            f"{rx5d_anom.item() * MM_PER_M:.2f} mm"
        ),
        (
            "Annual-maximum standardized event definition "
            "(annual-maximum anomaly divided by the 1991-2020 standard deviation of annual "
            "maxima for the same Pakistan area-mean 5-day precipitation sum): "
            f"{rx5d_stdev.item():.2f}"
        ),
    ]

    for line in rx5d_output_lines:
        print(line, flush=True)

    output_lines = jja_output_lines + [""] + rx5d_output_lines

    output_path = constants.OUTPUTS_DIR / "pakistan_rainfall.event_definitions"
    print(f"Writing event definitions to {output_path}...", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines) + "\n")
    print("Done.", flush=True)
