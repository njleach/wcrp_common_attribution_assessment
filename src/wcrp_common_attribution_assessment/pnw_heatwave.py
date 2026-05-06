import numpy as np
import xarray as xr

from wcrp_common_attribution_assessment import constants

if __name__ == "__main__":
    era5 = xr.open_dataset(
        "https://data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr",
        storage_options={"client_kwargs": {"trust_env": True}},
        chunks={},
        engine="zarr",
    )

    TX = (
        era5.mx2t.sel(  # select maximum temperature variable
            latitude=slice(52, 45),
            longitude=slice((-123 - 180) % 360 + 180, (-119 - 180) % 360 + 180),
        )  # select chosen PNW domain
        .resample(valid_time="1D")
        .max()  # resample to daily maximum at each gridpoint
        .mean(dim=["latitude", "longitude"])  # average over the domain
        .compute()  # trigger computation and load into memory
    )

    TXx = TX.resample(valid_time="1YE").max()

    # 1. Absolute definition
    absolute = TXx.sel(valid_time="2021").squeeze().values

    # 2. Annual maximum anomaly definition
    TXx_anom = (absolute - TXx.sel(valid_time=slice("1991", "2020")).mean()).squeeze().values

    # 3. Annual maximum standard deviation definition
    TXx_stdev = (TXx_anom / TXx.sel(valid_time=slice("1991", "2020")).std()).squeeze().values

    # 4. Climatological anomaly definition
    TX_dayofyear = TX.sel(valid_time="2021").idxmax().dt.dayofyear.squeeze().values
    era5_climatology_TX = TX.sel(
        valid_time=TX.valid_time.dt.dayofyear.isin(np.arange(TX_dayofyear - 15, TX_dayofyear + 15))
    ).sel(valid_time=slice("1991", "2020"))
    TX_anom = (absolute - era5_climatology_TX.mean()).squeeze().values

    # 5. Climatological standard deviation definition
    TX_stdev = (TX_anom / era5_climatology_TX.std()).squeeze().values

    print(f"Absolute event definition: {absolute - constants.ZERO_CELSIUS_IN_KELVIN:.2f}°C")
    print(f"Anomaly event definition: {TXx_anom:.2f}°C")
    print(f"Standard deviation event definition: {TXx_stdev:.2f}")
    print(f"Climatological anomaly event definition: {TX_anom:.2f}°C")
    print(f"Climatological standard deviation event definition: {TX_stdev:.2f}")
