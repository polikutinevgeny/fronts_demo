import matplotlib.colors
import numpy as np
import xarray as xr
from cartopy import crs as ccrs
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from crop import crop_center, crop_2d


def plot(data, fronts, in_size, date):
    proj = ccrs.LambertConformal(
        central_latitude=50,
        central_longitude=-107,
        false_easting=5632642.22547,
        false_northing=4612545.65137,
        standard_parallels=(50, 50),
        cutoff=-30
    )
    f = plt.figure(figsize=(8, 8))
    f.suptitle("Атмосферные фронты на {}".format(date.strftime("%Y-%m-%d %H:%M")), fontsize=16)
    ax = plt.subplot(1, 1, 1, projection=proj)
    with xr.open_dataset("/home/polikutin/FrontsDataset/narr_full/data.nc") as example:
        lat = crop_center(crop_2d(example.lat.values), in_size)
        lon = crop_center(crop_2d(example.lon.values), in_size)
        lon = (lon + 220) % 360 - 180  # Shift due to problems with crossing dateline in cartopy
    shift = ccrs.PlateCarree(central_longitude=-40)
    ax.set_xmargin(0.1)
    ax.set_ymargin(0.1)
    ax.set_extent((2.0e+6, 1.039e+07, 6.0e+5, 8959788), crs=proj)
    plt.contour(lon, lat, data[..., 1], levels=20, transform=shift, colors='black', linewidths=0.5)
    cmap = matplotlib.colors.ListedColormap([(0, 0, 0, 0), 'red', 'blue', 'green', 'purple'])
    plt.pcolormesh(lon, lat, fronts, cmap=cmap, zorder=10, transform=shift)
    hot = mpatches.Patch(facecolor='red', label='Тёплый фронт', alpha=1)
    cold = mpatches.Patch(facecolor='blue', label='Холодный фронт', alpha=1)
    stat = mpatches.Patch(facecolor='green', label='Стационарный фронт', alpha=1)
    occl = mpatches.Patch(facecolor='purple', label='Фронт окклюзии', alpha=1)
    ax.legend(handles=[hot, cold, stat, occl], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2,
              prop={'size': 12})
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    return f
