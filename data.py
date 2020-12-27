import shutil
import urllib.request as request
from contextlib import closing
from pathlib import Path
from datetime import date
import xarray as xr
import numpy as np

from crop import crop_center, crop_boundaries
from normalize import standardize


def download_year(year: int):
    for variable in ["air.2m", "mslet", "shum.2m", "uwnd.10m", "vwnd.10m"]:
        filename = Path(f"data/{variable.split('.')[0]}.{year}.nc")
        if filename.exists():
            continue
        filename.parent.mkdir(parents=True, exist_ok=True)
        url = f"ftp://ftp.cdc.noaa.gov/Datasets/NARR/monolevel/{variable}.{year}.nc"
        with closing(request.urlopen(url)) as r:
            with open(filename, "wb") as f:
                shutil.copyfileobj(r, f)


def get_date(dt: date):
    download_year(dt.year)
    result = []
    for variable in ["air", "mslet", "shum", "uwnd", "vwnd"]:
        filename = Path(f"data/{variable}.{dt.year}.nc")
        file = xr.open_dataset(filename, cache=False, engine='netcdf4')
        var = file[variable]
        result.append(np.expand_dims(standardize(var.sel(time=np.datetime64(dt)), var.name).fillna(0).values, -1))
        file.close()
    x = np.concatenate(result, axis=-1)
    x = crop_center(crop_boundaries(x), (256, 256, 5))
    return x[np.newaxis, ...]
