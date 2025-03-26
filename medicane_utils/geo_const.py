from mpl_toolkits.basemap import Basemap

latcorners = [30, 48]
loncorners = [-7, 46]
lat_min, lat_max = latcorners
lon_min, lon_max = loncorners
basemap_obj = Basemap(
    projection='geos',
    rsphere=(6378137.0, 6356752.3142),
    resolution='i',
    area_thresh=10000.,
    lon_0=9.5,
    satellite_height=3.5785831E7,
    llcrnrlon=lon_min,
    llcrnrlat=lat_min,
    urcrnrlon=lon_max,
    urcrnrlat=lat_max)

x_center, y_center = basemap_obj(9.5, 0)