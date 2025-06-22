from time import time
import datetime
import os
import numpy as np
import pandas as pd
import imageio.v2 as imageio
#import sys
import argparse
import warnings
warnings.filterwarnings("ignore") #, category=UserWarning)

#from read_plt_seviri import latcorners, loncorners, get_basemap
from geo_const import default_basem_obj, lat_min, lat_max, lon_min, lon_max
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

import xarray as xr
import ocf_blosc2  # non cancellare

#import dask
# Stampa sorgenti di config lette (in ordine di precedenza)
#print("Config sources:", dask.config.config_source)

# Verifica i valori specifici
#print("worker.memory.target =", dask.config.get("distributed.worker.memory.target"))
#print("worker.memory.spill  =", dask.config.get("distributed.worker.memory.spill"))
#print("worker.memory.pause  =", dask.config.get("distributed.worker.memory.pause"))


#dask.config.set(scheduler="threads", num_workers=16)  # usa i thread, tipicamente saturando i core
from dask.distributed import Client, LocalCluster


import gcsfs
#import fsspec

import traceback





########## funzioni per la riscalatura a TB, considerando solo i canali per AIRmassRGB
mins=np.array([2.844452,199.10002,199.57048,198.95093])
maxs=np.array([317.86752,313.2767,249.91806,286.96323])
variable_order=["IR_097","IR_108","WV_062","WV_073"]
def inverse_rescale_bulk(da):
    """
    Riporta da [0..1] ai valori originali [mins..maxs] su TUTTI i time-step.
    
    da: DataArray con dimensioni (time, y_geostationary, x_geostationary, variable)
        o (y_geostationary, x_geostationary, variable) se il time fosse già collassato.
    Ritorna un DataArray float32 con i valori fisici (Temperature di Brillanza o riflettanza).
    """
    # Assicurati che `variable` sia nell'ordine definito in variable_order
    # (Se variable_order = ["WV_062","WV_073","IR_097","IR_108"], e hai i min/max globali).
    da = da.reindex({"variable": variable_order}, copy=False)

    # Imposta l'ordine delle dimensioni se presenti
    # Se time c'è, verrà incluso. Se non c'è, ignorato.
    wanted_dims = ["time", "y_geostationary", "x_geostationary", "variable"]
    existing_dims = [d for d in wanted_dims if d in da.dims]
    da = da.transpose(*existing_dims)
    
    #se mins e maxs sono array con la stessa lunghezza di variable_order
    range_ = (maxs - mins)  # shape (4,) se i canali sono 4
    da = da * range_
    da = da + mins

    da = da.astype(np.float32)
    return da



def spatial_cut_geos(da, lat_min, lat_max, lon_min, lon_max, x_increasing=True, y_increasing=True, flip_north_up=False):

    """
    Esegue un taglio spaziale su 'da' (che ha dims x_geostationary, y_geostationary)
    usando Basemap per calcolare x_min, x_max, y_min, y_max in coordinate geostazionarie.
    Attenzione: devi avere la stessa proiezione e bounding dell'array.
    In particolare, calcoliamo un offset = m(9.5, 0), che corrisponde al sub-satellite point,
    e lo scaliamo a x0, y0, x1, y1.

    Parametri:
      da: DataArray (o Dataset) con coordinate x_geostationary, y_geostationary
      lat_min, lat_max, lon_min, lon_max: float, definiscono i corner in lat/lon
    Restituisce il sottoinsieme spaziale corrispondente.
    """
    #m = Basemap(
    #    projection='geos',
    #    rsphere=(6378137.0, 6356752.3142),
    ##    resolution='i',
    #    area_thresh=10000.,
    #    lon_0=9.5,
    #    satellite_height=3.5785831E7
    #)
    
    # Trovo offset del sub-satellite point (9.5E, 0)
    x_center, y_center = default_basem_obj(9.5, 0)
    
    # Converte lat/lon in coordinate x,y (geostazionarie) con la stessa proiezione
    x0, y0 = default_basem_obj(lon_min, lat_min)
    x1, y1 = default_basem_obj(lon_max, lat_max)
    
    # Sottraggo l'offset
    x0 -= x_center
    y0 -= y_center
    x1 -= x_center
    y1 -= y_center

    
    #print("Range X:", data_array.x_geostationary.min().values, data_array.x_geostationary.max().values)
    #print("Range Y:", data_array.y_geostationary.min().values, data_array.y_geostationary.max().values)
    #print(f"Corner finali: x0={x0} - x1={x1} \t y0={y0} - y1={y1}")
    
    #print(data_array.y_geostationary.values[:10])  # i primi 10
    #print(data_array.y_geostationary.values[-10:]) # gli ultimi 10
    
    # TROVATO L'ERRORE: QUì LE COORDINATE SONO DECRESCENTI, DOBIAMO FARE slice(x1, x0) DOPO
    #print("x coords[:10]:", data_array.x_geostationary.values[:10])
    #print("x coords[-10:]:", data_array.x_geostationary.values[-10:])


    # Niente swap automatico, usiamo l'info su x_increasing / y_increasing
    # Se x_increasing=False, dobbiamo passare slice(x1, x0)
    if x_increasing:
        x_start, x_end = (min(x0, x1), max(x0, x1))
    else:
        x_start, x_end = (max(x0, x1), min(x0, x1))

    if y_increasing:
        y_start, y_end = (min(y0, y1), max(y0, y1))
    else:
        y_start, y_end = (max(y0, y1), min(y0, y1))


    # Taglio su x_geostationary, y_geostationary
    da_cut = da.sel(
        x_geostationary=slice(x1, x0),  # QUì slice(x0, x1) risultava in un area vuota
        y_geostationary=slice(y0, y1)
    )

    nx = da_cut.sizes.get('x_geostationary', 0)
    ny = da_cut.sizes.get('y_geostationary', 0)
    if nx == 0 or ny == 0:
        print("Nessun pixel nell'area selezionata!\n")
        
    if flip_north_up:
        da_cut = da_cut.isel(y_geostationary=slice(None, None, -1))
        da_cut = da_cut.isel(x_geostationary=slice(None, None, -1))
    
    return da_cut


def download_date_range(data_array, date_range):
    now = time()
    immagini = []
    for d in date_range:
        img_dataora = data_array.sel(time=d, method='nearest')
        immagini.append(img_dataora)
    dopo = time()
    print(f"Tempo impiegato per download del chunk: {round(dopo - now, 2)} s.")
    return immagini


#medicane_start_date = "2020-09-05 00:00"
#medicane_end_date = "2020-09-25 23:59"
# Creazione della serie temporale con intervallo di 5 minuti
#medicane_range = pd.date_range(start=medicane_start_date, end=medicane_end_date, freq="5min")
#anno_2020_range = pd.date_range(start="2020-01-01 00:00", end="2020-12-31 23:59", freq="5min") 


    
    
# Funzione per combinare i canali e creare un DataArray con R, G, B
# R = WV_062 - WV_073
# B = IR_097 - IR_108
# G = WV_062
# Restituisce un DataArray con dimensione channel = [R, G, B].
def create_rgb_array(data_array_tb):
    # Seleziona i canali necessari
    wv62 = data_array_tb.sel(variable="WV_062")
    wv73 = data_array_tb.sel(variable="WV_073")
    ir097 = data_array_tb.sel(variable="IR_097")
    ir108 = data_array_tb.sel(variable="IR_108")

    # Costruisci i 3 canali
    R = wv62 - wv73
    G = ir097 - ir108
    B = wv62
    
    #print("--- Debug canali ---")
    #print("R:", R.min().values, R.max().values)
    #print("G:", G.min().values, G.max().values)
    #print("B:", B.min().values, B.max().values)

    # Concatena lungo una nuova dimensione "channel"
    rgb = xr.concat([R, G, B], dim="channel")
    # Assegna i nomi dei canali
    rgb = rgb.assign_coords(channel=["R", "G", "B"])

    # Metti in ordine (time, y_geostationary, x_geostationary, channel) se esiste time
    wanted_dims = ["time", "y_geostationary", "x_geostationary", "channel"]
    existing_dims = [d for d in wanted_dims if d in rgb.dims]
    rgb = rgb.transpose(*existing_dims)

    return rgb


def to_8bit_airmass(arr_float32):
    """
    Converte un array (y, x, 3) float32 in 8 bit secondo la "ricetta" Air Mass EUMETSAT.
    Canali:
      R in [-26..0]
      G in [-43..6]
      B in [243..208] (spesso invertito)
    """
    # arr_float32 ha shape (y, x, 3): [R, G, B] = [IR9.7-IR10.8, WV6.2-WV7.3, WV6.2]

    R_chan = arr_float32[:,:,0]  # range ~ [-26..0]
    G_chan = arr_float32[:,:,1]  # range ~ [-43..6]
    B_chan = arr_float32[:,:,2]  # range ~ [243..208]
    
    # Imposta i range
    #dipende dalla dinamica che si vuole enfatizzare.
    Rmin, Rmax = -25, 0   
    Gmin, Gmax = -40, 5    
    Bmin, Bmax = 243, 208

    # Funzione per scalare con range custom.
    # Esempio: se range=[-26,0], un valore -26 -> 0 e 0 -> 1
    def scale_custom(x, lower, upper, invert=False):
        # Se invert=True, invertiamo la scala.
        lo, hi = min(lower, upper), max(lower, upper)
        # Clip
        x_clipped = np.clip(x, lo, hi)
        # Normalizzo
        scaled = (x_clipped - lo) / (hi - lo)
        if invert:
            # invertiamo  (0 -> 1, 1 -> 0)
            scaled = 1.0 - scaled

        return np.clip(scaled, 0, 1)

    # Scalature sui range indicati da EUMETSAT    
    R_scaled = scale_custom(R_chan, Rmin, Rmax, invert=False)
    G_scaled = scale_custom(G_chan, Gmin, Gmax, invert=False)
    B_scaled = scale_custom(B_chan, Bmin, Bmax, invert=True)

    rgb_normalized = np.stack([R_scaled, G_scaled, B_scaled], axis=-1)
    arr_8bit = (rgb_normalized * 255).astype(np.uint8)
    return arr_8bit




# Esempio di funzione che elabora un intervallo di date (start_date, end_date).
# Scarica dati, fa il cut, denormalizza, crea RGB e salva.
def process_date_range(data_array, start_date, end_date):
	# creo la cartella se non c'è 
    if not os.path.exists('from_gcloud/'):os.makedirs('from_gcloud/')
	
    x_increasing = data_array.x_geostationary[0] < data_array.x_geostationary[-1]
    y_increasing = data_array.y_geostationary[0] < data_array.y_geostationary[-1]
    #print(f"x_increasing? {x_increasing}")
    #print(f"y_increasing? {y_increasing}")

    day_list = pd.date_range(start=start_date, end=end_date, freq="1D")
    data_array = data_array.sortby("time")
                             
    
    ds_sub = data_array.sel(time=slice(start_date, end_date),
        variable=["IR_097","IR_108","WV_062","WV_073"])  # solo i canali necessari
    #print(ds_sub.shape)
                             
    ds_sub_cut = spatial_cut_geos(ds_sub, lat_min, lat_max, lon_min, lon_max,
                            x_increasing=x_increasing,
                            y_increasing=y_increasing,
                            flip_north_up=True)
    #print(ds_sub_cut.shape)
                             
    ds_tb = inverse_rescale_bulk(ds_sub_cut)
    #print(ds_tb.shape)
                             
    ds_rgb = create_rgb_array(ds_tb)
    #print(ds_rgb.shape)
    
    for day in day_list:
        day_str = day.strftime("%Y-%m-%d")
        print(f"Process day: {day_str}")
        # Seleziono i time-step di questo giorno
        try:
            ds_day = ds_rgb.sel(time=day_str)
            if "time" not in ds_day.dims or ds_day.sizes["time"] == 0:
                print("Nessun time-step in questo giorno, skip")
                continue

            # Carico in RAM la data di 1 giorno
            t0 = time()
            ds_day_loaded = ds_day.compute()  # dask userà chunk in parallelo
            print(f"Compute day {day_str} in {round(time()-t0,2)}s, shape={ds_day_loaded.sizes}")

             # Ora salvo i singoli frame
            n_time = ds_day_loaded.sizes["time"]
            t0 = time() 
            for i in range(n_time):
                try:
                    da_one = ds_day_loaded.isel(time=i)
                    arr_f32 = da_one.values  # shape (y, x, 3)
                    #print(f"arr_f32 shape: {arr_f32.shape}")
                    arr_8bit = to_8bit_airmass(arr_f32)
                    #print(f"arr_8bit shape: {arr_8bit.shape}")

                    time_val = ds_day_loaded.time.isel(time=i).values
                    time_str = pd.to_datetime(time_val).strftime('%Y%m%d_%H%M')
                    out_png = f"from_gcloud/airmass_rgb_{time_str}.png"
                    
                    imageio.imwrite(out_png, arr_8bit)                    
                    
                    # quì cambio se voglio mettere le linee di costa, 
                    ###############################################à#################
                    ### MA CAMBIA LA DIMENSIONE DEL FILE! diventava 992 X 323 pixel
                    ### dpi = 96
                    ### plt.figure(figsize=(1290/dpi, 420/dpi), dpi=dpi)
                    ### m.imshow(arr_8bit, origin='upper')
                    ### m.drawcoastlines()                    
                    ### plt.savefig(out_png, bbox_inches='tight', pad_inches=0) 
                    #################################################################
                except Exception as e:
                    print(f"Salto la {i} slice")
                    #print(e.stack_trace())
                    print(traceback.format_exc())

        except:
            print(f"Salto il giorno {day_str}")
            n_time = 0
            t0 = 0

        dopo = time()
        print(f"Finiti {n_time} frame per {day_str} in {round(dopo-t0,2)}s. ")
        
    print("Tutto completato!")
    

    
if __name__ == "__main__":
    
    cluster = LocalCluster(n_workers=16, threads_per_worker=1)
    client = Client(cluster)
    print(client)
    fs = gcsfs.GCSFileSystem(token='anon',     #'google_default', 
    	max_pool_connections=20)
    
    
    ### argomenti date
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    args = parser.parse_args()

    start_date = pd.to_datetime(args.start)
    end_date = pd.to_datetime(args.end)
    year_start = start_date.year
    year_end = end_date.year
    
    if year_start != year_end:
        print("Attenzione: l'intervallo copre più anni, usare un solo anno")
    else:
        year_for_zarr = year_start
    
    zarrNONHRVfile_path = f"gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v4/{year_for_zarr}_nonhrv.zarr" 
    mapper = fs.get_mapper(zarrNONHRVfile_path)
    #dataset = load_zarr_file(zarrNONHRVfile_path)
    dataset = xr.open_zarr(mapper,  consolidated=True)
    data_array = dataset['data']
    print(f"Scarico il file {zarrNONHRVfile_path}")


    
    process_date_range(data_array, start_date, end_date)
    #parallel_processing()

    
    
    
#Compute day 2020-12-31 in 287.9s
#Finiti 288 frame per 2020-12-31 in 64.39s. 
