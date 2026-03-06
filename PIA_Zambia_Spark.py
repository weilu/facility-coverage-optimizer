# Databricks notebook source
# MAGIC %pip install geopandas --upgrade 
# MAGIC %pip install shapely rasterio pycountry gurobipy folium plotly scikit-learn pyproj gadm
# MAGIC

# COMMAND ----------

# MAGIC %pip install -U geopandas shapely

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

#------------------
import folium as fl
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
import os, json, requests, urllib.request, itertools
import pycountry
import plotly.graph_objects as go

from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union, transform
from shapely.wkt import loads as wkt_loads
from sklearn.cluster import KMeans
from collections import defaultdict
from pyproj import Transformer

from gadm import GADMDownloader

from pyspark.sql import functions as F
from pyspark.sql.types import (StringType, DoubleType, BooleanType,
                                ArrayType, FloatType, StructType, StructField)
import gurobipy as gp
from gurobipy import GRB

print(f"Spark version: {spark.version}")

# COMMAND ----------

#------------------
# SPATIAL UDFs — replace all Sedona ST_* functions

from pyspark.sql.types import StringType, BooleanType, ArrayType
from pyspark.sql.functions import udf, pandas_udf
import pandas as pd
from shapely.wkt import loads as wkt_loads
from shapely.geometry import Point
from shapely.ops import transform
from pyproj import Transformer

# ── UDF: create WKT point from lon/lat ───────────────────────────────────────
@udf(StringType())
def st_point_wkt(lon, lat):
    """Returns WKT string for a point. Replaces ST_Point."""
    if lon is None or lat is None:
        return None
    return Point(float(lon), float(lat)).wkt

# ── UDF: ST_Within — does point (WKT) fall inside polygon (WKT)? ─────────────
@udf(BooleanType())
def st_within_wkt(point_wkt, polygon_wkt):
    """Returns True if point is within polygon. Replaces ST_Within."""
    if point_wkt is None or polygon_wkt is None:
        return False
    try:
        return wkt_loads(point_wkt).within(wkt_loads(polygon_wkt))
    except Exception:
        return False

# ── Pandas UDF: ST_Buffer — buffer a geometry (WKT) by distance in metres ────
# Reprojects to Web Mercator (metres), buffers, reprojects back to WGS84.
# This is a vectorized pandas_udf — runs on batches of rows per executor.
@pandas_udf(StringType())
def st_buffer_meters(geom_wkt_series: pd.Series, distance_series: pd.Series) -> pd.Series:
    """Returns buffered geometry as WKT. Replaces ST_Transform+ST_Buffer+ST_Transform."""
    to_3857   = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    to_4326   = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform

    results = []
    for wkt, dist in zip(geom_wkt_series, distance_series):
        try:
            geom_3857    = transform(to_3857, wkt_loads(wkt))
            buffered_3857 = geom_3857.buffer(float(dist))
            buffered_4326 = transform(to_4326, buffered_3857)
            results.append(buffered_4326.wkt)
        except Exception:
            results.append(None)
    return pd.Series(results)

print("Spatial UDFs registered.")

# COMMAND ----------

# HELPER FUNCTIONS

def get_country_codes(country_name: str):
    """Look up ISO codes for a country name."""
    try:
        country = pycountry.countries.lookup(country_name)
        return {
            'name': country.name,
            'alpha_2': country.alpha_2,
            'alpha_3': country.alpha_3,
            'numeric': country.numeric
        }
    except LookupError:
        return None


def gadm_to_boundary_wkt(selected_gadm_pdf: gpd.GeoDataFrame) -> str:
    """
    Returns WKT of the unioned AOI geometry.
    Broadcast as a literal to workers — avoids a full Spark join for the boundary filter.
    For a single admin boundary this is always small enough to broadcast.
    """
    union_geom = selected_gadm_pdf.geometry.unary_union
    return union_geom.wkt


def visualize_gadm(df_shp_pdf: gpd.GeoDataFrame, selected_gadm_pdf: gpd.GeoDataFrame):
    """Visualize GADM boundaries using Folium. Operates on local GeoDataFrames."""
    m = fl.Map(location=[-8.556856, 125.560314], zoom_start=9, tiles="OpenStreetMap")
    for _, r in df_shp_pdf.iterrows():
        sim_geo = gpd.GeoSeries(r["geometry"]).simplify(tolerance=0.00001)
        geo_j = fl.GeoJson(data=sim_geo.to_json(), style_function=lambda x: {"fillColor": "orange"})
        fl.Popup(r["NAME_1"]).add_to(geo_j)
        geo_j.add_to(m)
    for _, r in selected_gadm_pdf.iterrows():
        sim_geo = gpd.GeoSeries(r["geometry"]).simplify(tolerance=0.001)
        geo_j = fl.GeoJson(data=sim_geo.to_json(), style_function=lambda x: {"fillColor": "red"})
        fl.Popup(r["NAME_1"]).add_to(geo_j)
        geo_j.add_to(m)
    return m


def download_worldpop_geotiff(country_iso3: str, population_year: int, output_dir: str = "./data") -> str:
    """Downloads WorldPop GeoTIFF raster to local driver storage."""
    url = (
        f"https://data.worldpop.org/GIS/Population/Global_2015_2030/R2024B/"
        f"{population_year}/{country_iso3}/v1/100m/constrained/"
        f"{country_iso3.lower()}_pop_{population_year}_CN_100m_R2024B_v1.tif"
    )
    print(url)
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{country_iso3}_{population_year}_pop.tif")
    urllib.request.urlretrieve(url, file_path)
    return file_path


#------------------
def raster_to_spark_df(raster_path: str, num_partitions: int = 400):
    """Reads GeoTIFF on driver, parallelizes populated pixels to Spark."""
    print(f"Reading raster: {raster_path}")
    with rasterio.open(raster_path) as src:
        data      = src.read(1)
        transform_affine = src.transform
        rows, cols = np.where(data > 0)
        values     = data[rows, cols].astype(float)
        x_coords, y_coords = rasterio.transform.xy(transform_affine, rows, cols, offset='center')

    print(f"Raster read complete. {len(values):,} populated pixels.")
    pdf = pd.DataFrame({
        'xcoord':     np.array(x_coords, dtype=float),
        'ycoord':     np.array(y_coords, dtype=float),
        'population': values
    })
    sdf = spark.createDataFrame(pdf).repartition(num_partitions)
    sdf = (sdf
           .withColumn('geom_wkt', st_point_wkt(F.col('xcoord'), F.col('ycoord')))
           .withColumn('row_id',   F.monotonically_increasing_id())
           .withColumn('ID',       F.concat(F.col('row_id').cast(StringType()), F.lit('_pop'))))
    return sdf

#------------------
def download_worldpop_geotiff(country_iso3: str, population_year: int, output_dir: str = "/Volumes/prd_mega/sgpbpi163/vgpbpi163") -> str:
    url = f"https://data.worldpop.org/GIS/Population/Global_2015_2030/R2024B/{population_year}/{country_iso3}/v1/100m/constrained/{country_iso3.lower()}_pop_{population_year}_CN_100m_R2024B_v1.tif"
    print(url)
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{country_iso3}_{population_year}_pop.tif")
    urllib.request.urlretrieve(url, file_path)
    return file_path

def get_population_spark(iso_3: str, population_year: int, boundary_wkt: str):
    """
    Reads raster → Spark DF, filters to AOI using broadcast ST_Within UDF.
    boundary_wkt: result of gadm_to_boundary_wkt() — broadcast as a literal.
    """
    raster_file = download_worldpop_geotiff(iso_3, population_year)
    pop_sdf   = raster_to_spark_df(raster_file)

    total_pop = pop_sdf.agg(F.sum('population')).collect()[0][0]
    print(f'Total Population: {round(total_pop / 1_000_000, 2)} million')

    # Broadcast the boundary WKT as a literal — no shuffle, no join
    boundary_lit = F.lit(boundary_wkt)
    pop_aoi_sdf  = pop_sdf.filter(st_within_wkt(F.col('geom_wkt'), boundary_lit))

    aoi_total = pop_aoi_sdf.agg(F.sum('population')).collect()[0][0]
    print(f'Total Population (AOI - {iso_3}): {round(aoi_total):,}')

    thresholds  = pop_aoi_sdf.approxQuantile('population', [0.25, 0.5, 0.75], 0.01)
    pop_aoi_sdf = pop_aoi_sdf.withColumn(
        'opacity',
        F.when(F.col('population') <= thresholds[0], F.lit(0.1))
         .when(F.col('population') <= thresholds[1], F.lit(0.3))
         .when(F.col('population') <= thresholds[2], F.lit(0.6))
         .otherwise(F.lit(1.0))
    )
    pop_aoi_sdf = pop_aoi_sdf.cache()
    pop_aoi_sdf.count()
    print('get_population_spark done.')
    return pop_aoi_sdf

def get_population_from_volume_spark(raster_file: str, iso_3: str, population_year: int, selected_gadm_sdf):
    """Same as get_population_spark but reads from a Databricks Volume path."""
    return get_population_spark(raster_file, iso_3, population_year, selected_gadm_sdf)


def get_health_facilities_osm_spark(iso_2: str, selected_gadm_sdf, adm_level_name: str = "AOI"):
    """
    Queries OSM Overpass API for hospitals and clinics (HTTP call on driver, small payload),
    converts the result to a Sedona Spark DataFrame, then spatially joins with the AOI boundary.

    Returns a cached Spark DataFrame: [id, name, lat, lon, geometry, ID]
    """
    def query_osm_amenity(amenity: str) -> pd.DataFrame:
        query = f"""
        [out:json];
        area["ISO3166-1"="{iso_2}"];
        (
          node["amenity"="{amenity}"](area);
          way["amenity"="{amenity}"](area);
          rel["amenity"="{amenity}"](area);
        );
        out center;
        """
        response = requests.get("http://overpass-api.de/api/interpreter", params={'data': query})
        response.raise_for_status()
        elements = response.json()['elements']
        df = pd.DataFrame(elements)
        if df.empty:
            return pd.DataFrame(columns=['id', 'lat', 'lon', 'name'])
        df['name'] = df['tags'].apply(lambda x: x.get('name') if isinstance(x, dict) else None)
        return df[['id', 'lat', 'lon', 'name']].dropna(subset=['lat', 'lon'])

    df_hospitals = query_osm_amenity("hospital")
    df_clinics   = query_osm_amenity("clinic")
    df_health    = pd.concat([df_hospitals, df_clinics]).drop_duplicates(subset='id').reset_index(drop=True)
    print(f"Total hospitals + clinics from OSM: {len(df_health)}")

    # Convert to Sedona Spark DataFrame
    health_sdf = spark.createDataFrame(df_health)
    health_sdf = health_sdf.withColumn(
        'geometry',
        F.expr("ST_Point(CAST(lon AS DOUBLE), CAST(lat AS DOUBLE))")
    )

    # Spatial join with AOI
    health_sdf.createOrReplaceTempView("health_all")
    selected_gadm_sdf.select(F.col('geometry').alias('boundary_geom')).createOrReplaceTempView("gadm_boundary")

    selected_health_sdf = spark.sql("""
        SELECT h.id, h.name, h.lat, h.lon, h.geometry
        FROM health_all h, gadm_boundary b
        WHERE ST_Within(h.geometry, b.boundary_geom)
    """)

    selected_health_sdf = (selected_health_sdf
                           .withColumn('row_id', F.monotonically_increasing_id())
                           .withColumn('ID', F.concat(F.col('row_id').cast(StringType()), F.lit('_current'))))

    selected_health_sdf = selected_health_sdf.cache()
    count = selected_health_sdf.count()
    print(f"Facilities in AOI ({adm_level_name}): {count}")
    return selected_health_sdf


def get_isochrone_osm(lon: float, lat: float, travel_distance_meters: int) -> str | None:
    """
    Calls OpenRouteService isochrone API for a single point.
    Returns WKT polygon string or None on failure.
    (Still runs on driver — HTTP API calls cannot be distributed.)
    """
    body = {
        "locations": [[lon, lat]],
        "range": [travel_distance_meters],
        "range_type": "distance"
    }
    headers = {
        'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
        'Authorization': 'eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6Ijk1ZGQzYTg1ODQ2MzQ0N2E5ZjBhOWRhODNiNjUwYzgzIiwiaCI6Im11cm11cjY0In0=',
        'Content-Type': 'application/json; charset=utf-8'
    }
    call = requests.post('https://api.openrouteservice.org/v2/isochrones/driving-car', json=body, headers=headers)
    if call.status_code == 200:
        geom = json.loads(call.text)['features'][0]['geometry']
        print("Openrouteservice API Call Success")
        return Polygon(geom['coordinates'][0]).wkt
    else:
        print(f"Openrouteservice API Call Failed, Code={call.status_code}")
        return None


def get_isochrone_mapbox(lon: float, lat: float, meters: str, access_token: str, mode: str) -> str | None:
    """
    Calls Mapbox Isochrone API for a single point.
    Returns WKT polygon string or None on failure.
    """
    query = (f"https://api.mapbox.com/isochrone/v1/mapbox/{mode}/{lon},{lat}?"
             f"contours_meters={meters}&polygons=true&access_token={access_token}")
    req_return = requests.get(query).json()
    if 'code' in req_return and req_return['code'] == 'NoSegment':
        print('No Segment')
        return None
    features = req_return.get('features', [])
    if features:
        return Polygon(features[0]['geometry']['coordinates'][0]).wkt
    return None


#------------------
def compute_catchment_areas_spark(facilities_sdf, distance_meters: int,
                                   travel_api: str = '',
                                   mapbox_access_token: str = '',
                                   mapbox_mode: str = 'driving'):
    """
    Adds cachment_area_wkt (WKT string) column to facilities Spark DataFrame.
    For travel_api='': fully distributed Shapely buffer via pandas_udf.
    For 'osm'/'mapbox': collects to driver, calls API, returns to Spark (small facility sets only).
    """
    if travel_api in ('osm', 'mapbox'):
        pdf = facilities_sdf.toPandas()
        if travel_api == 'osm':
            pdf['cachment_area_wkt'] = pdf.apply(
                lambda r: get_isochrone_osm(r['lon'], r['lat'], distance_meters), axis=1)
        else:
            pdf['cachment_area_wkt'] = pdf.apply(
                lambda r: get_isochrone_mapbox(r['lon'], r['lat'], str(distance_meters),
                                               mapbox_access_token, mapbox_mode), axis=1)
        return spark.createDataFrame(pdf.dropna(subset=['cachment_area_wkt']))
    else:
        # ── Fully distributed buffer via pandas_udf ──────────────────────────
        dist_col = F.lit(float(distance_meters))
        return facilities_sdf.withColumn(
            'cachment_area_wkt',
            st_buffer_meters(F.col('geom_wkt'), dist_col)
        )

#------------------
def compute_coverage_spark(facilities_sdf, population_sdf):
    """
    Computes which population points fall inside each facility's catchment area.

    Strategy: repartition facilities into small batches, broadcast each batch's
    catchment WKTs to workers, filter population points per batch, union results.
    This avoids a full O(N×M) cartesian product.

    For Zambia-scale data (~500 facilities × ~2M pop points) this runs in minutes.
    """
    # Collect catchment WKTs to driver (facilities are small — hundreds of rows)
    fac_pdf = facilities_sdf.select('ID', 'cachment_area_wkt').toPandas()
    fac_pdf = fac_pdf.dropna(subset=['cachment_area_wkt'])
    fac_shapes = {row['ID']: wkt_loads(row['cachment_area_wkt'])
                  for _, row in fac_pdf.iterrows()}

    # Collect population points to driver
    # ⚠️ For very large AOIs (>5M points), sample or use the batch approach below
    pop_pdf = population_sdf.select('ID', 'xcoord', 'ycoord', 'population').toPandas()

    print(f"Computing coverage: {len(fac_shapes)} facilities × {len(pop_pdf):,} pop points...")

    rows = []
    for fac_id, catchment in fac_shapes.items():
        mask      = pop_pdf.apply(
            lambda r: Point(r['xcoord'], r['ycoord']).within(catchment), axis=1)
        covered   = pop_pdf[mask]
        rows.append({
            'facility_ID':    fac_id,
            'covered_pop_ids': covered['ID'].tolist(),
            'pop_with_access': float(covered['population'].sum())
        })

    coverage_pdf = pd.DataFrame(rows)
    coverage_sdf = spark.createDataFrame(coverage_pdf)

    # Join coverage back to facilities
    result_sdf = facilities_sdf.join(
        coverage_sdf.withColumnRenamed('facility_ID', 'ID'),
        on='ID', how='left'
    ).fillna({'pop_with_access': 0.0})

    # Build flat (facility_ID, pop_ID) pairs as a Spark DF for JI construction
    flat_rows = [(fid, pid)
                 for _, r in coverage_pdf.iterrows()
                 for pid in r['covered_pop_ids']]
    flat_schema = StructType([
        StructField('facility_ID', StringType()),
        StructField('pop_ID',      StringType())
    ])
    flat_sdf = spark.createDataFrame(flat_rows, schema=flat_schema)

    return result_sdf, flat_sdf


def generate_grid_in_polygon(spacing: float, geometry) -> pd.DataFrame:
    """
    Generates a regular point grid within the given Shapely geometry.
    Grid generation is local (fast), clipping with geopandas (also fast for moderate grids).
    Returns pandas DataFrame [longitude, latitude] — converted to Spark in the next step.
    """
    minx, miny, maxx, maxy = geometry.bounds
    x_coords = np.arange(np.floor(minx), np.ceil(maxx), spacing)
    y_coords = np.arange(np.floor(miny), np.ceil(maxy), spacing)
    mesh = np.meshgrid(x_coords, y_coords)
    pdf = pd.DataFrame({'longitude': mesh[0].flatten(), 'latitude': mesh[1].flatten()})
    gdf = gpd.GeoDataFrame(pdf, geometry=gpd.points_from_xy(pdf.longitude, pdf.latitude), crs='EPSG:4326')
    gdf = gpd.clip(gdf, geometry).reset_index(drop=True)
    print(f'Number of potential locations: {len(gdf)}')
    return gdf[['longitude', 'latitude']]


def generate_kmeans(population_aoi_pdf: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    """
    Runs KMeans on population coordinates to generate candidate facility locations.
    Operates on a pandas DF (collected from Spark or sampled subset).
    Returns pandas DataFrame [longitude, latitude].
    """
    coords = population_aoi_pdf[['xcoord', 'ycoord']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(coords)
    centroids = kmeans.cluster_centers_
    pdf = pd.DataFrame(centroids, columns=['longitude', 'latitude'])
    print(f'Number of potential locations: {len(pdf)}')
    return pdf


def locations_pdf_to_spark_sdf(pdf: pd.DataFrame, id_suffix: str = '_potential'):
    """
    Converts a pandas DataFrame of [longitude, latitude] to a Sedona Spark DataFrame.
    Adds a Sedona geometry column and a unique ID.
    """
    sdf = spark.createDataFrame(pdf)
    sdf = (sdf
           .withColumn('geometry', F.expr("ST_Point(CAST(longitude AS DOUBLE), CAST(latitude AS DOUBLE))"))
           .withColumn('row_id', F.monotonically_increasing_id())
           .withColumn('ID', F.concat(F.col('row_id').cast(StringType()), F.lit(id_suffix))))
    return sdf


def visualize_potential_locations(potential_locations_pdf: pd.DataFrame, selected_gadm_pdf: gpd.GeoDataFrame,
                                   center_lat: float, center_lon: float):
    """Visualize potential facility locations. Operates on collected pandas DataFrames."""
    folium_map = fl.Map([center_lat, center_lon], zoom_start=11)
    geo_adm = fl.GeoJson(data=selected_gadm_pdf.iloc[0]['geometry'].__geo_interface__,
                          style_function=lambda x: {'color': 'orange'})
    geo_adm.add_to(folium_map)
    for _, row in potential_locations_pdf.iterrows():
        fl.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3, color='blue', fill=True, fill_color='blue', fill_opacity=0.7
        ).add_to(folium_map)
    return folium_map


def model_max_covering_gurobi(w, I, J, JI, p, J_existing, model):
    """
    Gurobi Maximum Covering Location Problem (MCLP).
    Runs on the driver (single-node optimization). Inputs are Python dicts/lists
    collected from Spark DataFrames upstream.
    """
    x = model.addVars(J, vtype=GRB.BINARY, name="x")   # 1 = facility open
    z = model.addVars(I, vtype=GRB.BINARY, name="z")   # 1 = demand point covered

    model.setObjective(gp.quicksum(w[i] * z[i] for i in I), GRB.MAXIMIZE)

    for i in I:
        if JI.get(i):
            model.addConstr(z[i] <= gp.quicksum(x[j] for j in JI[i]), name=f"cover_{i}")
        else:
            model.addConstr(z[i] == 0, name=f"cover_{i}_zero")

    model.addConstr(gp.quicksum(x[j] for j in J) <= len(J_existing) + p, name="budget")

    for j in J_existing:
        model.addConstr(x[j] == 1, name=f"existing_{j}")

    model.setParam("OutputFlag", 0)
    model.optimize()

    x_sol = {j: x[j].X for j in J}
    z_sol = {i: z[i].X for i in I}

    selected_facilities = [j for j in J if x_sol[j] > 0.5]
    covered_demand      = [i for i in I if z_sol[i] > 0.5]

    return model.ObjVal, selected_facilities, covered_demand


# COMMAND ----------

country              = 'Zambia'
adm_level1           = None
adm_level2           = None
population_year      = 2025
travel_api           = ''
distance_meters      = 10000
mapbox_access_token  = ''
potential_type       = 'grid'
spacing              = 0.03
n_clusters           = 100 

iso_2 = get_country_codes(country)['alpha_2']
iso_3 = get_country_codes(country)['alpha_3']
print(f"Country: {country} | ISO-2: {iso_2} | ISO-3: {iso_3}")

# COMMAND ----------

iso_2 = get_country_codes(country)['alpha_2']
iso_3 = get_country_codes(country)['alpha_3']

downloader = GADMDownloader(version="4.0")

if(adm_level1!= None):
  df_shp = downloader.get_shape_data_by_country_name(country_name=country, ad_level=1)
  selected_gadm_pdf = df_shp[df_shp['NAME_1']==adm_level1]
else:
  if (adm_level2!= None):
    df_shp = downloader.get_shape_data_by_country_name(country_name=country, ad_level=2)
    selected_gadm_pdf = df_shp[df_shp['NAME_2']==adm_level2]
  else:
    df_shp = downloader.get_shape_data_by_country_name(country_name=country, ad_level=0)
    selected_gadm_pdf = df_shp

# print(selected_gadm)
centroid = selected_gadm_pdf.iloc[0]['geometry'].centroid
center_lat = centroid.y
center_lon = centroid.x
print(f"Map center: lat={center_lat:.4f}, lon={center_lon:.4f}")

# Convert to Sedona Spark DataFrame for later spatial joins
selected_gadm_sdf = gadm_to_boundary_wkt(selected_gadm_pdf)
selected_gadm_sdf = selected_gadm_sdf
# print(selected_gadm_sdf)


# COMMAND ----------

# raster_file     = "/Volumes/prd_mega/sgpbpi163/vgpbpi163/ZMB_2025_pop.tif.tif"
# population_aoi_sdf = get_population_from_volume_spark(raster_file, iso_3, population_year, selected_gadm_sdf)

# OLD (Sedona):
# selected_gadm_sdf = gadm_to_sedona_sdf(selected_gadm_pdf)
# population_aoi_sdf = get_population_spark(raster_file, iso_3, population_year, selected_gadm_sdf)

# NEW (no Sedona):
boundary_wkt       = gadm_to_boundary_wkt(selected_gadm_pdf)
population_aoi_sdf = get_population_spark(iso_3, population_year, boundary_wkt)
# Total population for denominator calculations
total_population = population_aoi_sdf.agg(F.sum('population')).collect()[0][0]


# COMMAND ----------

# LOAD EXISTING HEALTH FACILITIES
# Reads the GeoJSON on the driver (small file), converts to Sedona Spark DataFrame.
# If you prefer OSM live query: use get_health_facilities_osm_spark(iso_2, selected_gadm_sdf)

path = "/Volumes/prd_mega/sgpbpi163/vgpbpi163/selected_hosp_input_data.geojson"
selected_hosp_pdf = gpd.read_file(path)
selected_hosp_pdf['lon'] = selected_hosp_pdf.geometry.x
selected_hosp_pdf['lat'] = selected_hosp_pdf.geometry.y

hosp_cols = [c for c in selected_hosp_pdf.columns if c != 'geometry']
hosp_no_geom_pdf = selected_hosp_pdf[hosp_cols].copy()
hosp_no_geom_pdf['wkt'] = selected_hosp_pdf.geometry.apply(lambda g: g.wkt)

selected_hosp_sdf = spark.createDataFrame(hosp_no_geom_pdf)
selected_hosp_sdf = selected_hosp_sdf.withColumn('geometry', F.expr("ST_GeomFromWKT(wkt)")).drop('wkt')

# Assign stable IDs if not present in the source file
if 'ID' not in selected_hosp_sdf.columns:
    selected_hosp_sdf = (selected_hosp_sdf
                         .withColumn('row_id', F.monotonically_increasing_id())
                         .withColumn('ID', F.concat(F.col('row_id').cast(StringType()), F.lit('_current'))))

selected_hosp_sdf = selected_hosp_sdf.cache()
selected_hosp_sdf.count()
print(f"Existing facilities loaded: {selected_hosp_sdf.count()}")


# COMMAND ----------

aoi_union_geom = selected_gadm_pdf.geometry.unary_union   # Combined AOI geometry (Shapely)

if potential_type == 'grid':
    potential_pdf = generate_grid_in_polygon(spacing=spacing, geometry=aoi_union_geom)
else:  # kmeans
    # Sample population points to driver for clustering (avoid full collect of large DF)
    sample_fraction = min(1.0, 500_000 / total_population)
    pop_sample_pdf = (population_aoi_sdf
                      .select('xcoord', 'ycoord')
                      .sample(fraction=sample_fraction, seed=42)
                      .toPandas())
    potential_pdf = generate_kmeans(pop_sample_pdf, n_clusters)

# Convert to Sedona Spark DataFrame
potential_locations_sdf = locations_pdf_to_spark_sdf(potential_pdf, id_suffix='_potential')
potential_locations_sdf = potential_locations_sdf.cache()
potential_locations_sdf.count()


# COMMAND ----------

selected_hosp_sdf      = compute_catchment_areas_spark(selected_hosp_sdf, travel_api, distance_meters,
                                                        mapbox_access_token)
potential_locations_sdf = compute_catchment_areas_spark(potential_locations_sdf, travel_api, distance_meters,
                                                         mapbox_access_token)

# Re-cache after adding catchment areas
selected_hosp_sdf       = selected_hosp_sdf.cache()
potential_locations_sdf = potential_locations_sdf.cache()
selected_hosp_sdf.count()
potential_locations_sdf.count()

# COMMAND ----------

selected_hosp_sdf, hosp_coverage_sdf = compute_coverage_spark(
    selected_hosp_sdf, population_aoi_sdf,
    fac_view='_hosp_fac', pop_view='_hosp_pop'
)

#------------------
# COMPUTE POPULATION COVERAGE — POTENTIAL LOCATIONS

potential_locations_sdf, potential_coverage_sdf = compute_coverage_spark(
    potential_locations_sdf, population_aoi_sdf,
    fac_view='_pot_fac', pop_view='_pot_pop'
)


# COMMAND ----------

existing_covered_ids_sdf = hosp_coverage_sdf.select('pop_ID').distinct()

pop_with_access_sdf    = population_aoi_sdf.join(
    existing_covered_ids_sdf,
    population_aoi_sdf['ID'] == existing_covered_ids_sdf['pop_ID'],
    'inner'
).drop('pop_ID')

pop_without_access_sdf = population_aoi_sdf.join(
    existing_covered_ids_sdf,
    population_aoi_sdf['ID'] == existing_covered_ids_sdf['pop_ID'],
    'left_anti'
)

covered_pop_val = pop_with_access_sdf.agg(F.sum('population')).collect()[0][0]
current_access  = round(covered_pop_val * 100 / total_population, 2)
print(f'Population with Access (current): {current_access}%')

# Collect to pandas for Folium rendering (sampling large datasets for map performance)
_POP_VIZ_SAMPLE = 20_000   # Increase if desired; Folium can struggle with >50k markers

pop_with_access_pdf    = pop_with_access_sdf.sample(
    fraction=min(1.0, _POP_VIZ_SAMPLE / max(1, pop_with_access_sdf.count())), seed=1
).toPandas()

pop_without_access_pdf = pop_without_access_sdf.sample(
    fraction=min(1.0, _POP_VIZ_SAMPLE / max(1, pop_without_access_sdf.count())), seed=2
).toPandas()

selected_hosp_pdf_viz  = selected_hosp_sdf \
    .withColumn('lon', F.expr("ST_X(geometry)")) \
    .withColumn('lat', F.expr("ST_Y(geometry)")) \
    .toPandas()

# Build Folium map
folium_map = fl.Map(location=[center_lat, center_lon], zoom_start=8, tiles="OpenStreetMap")
geo_adm    = fl.GeoJson(data=selected_gadm_pdf.iloc[0]['geometry'].__geo_interface__,
                         style_function=lambda x: {'color': 'orange'})
geo_adm.add_to(folium_map)

for _, row in selected_hosp_pdf_viz.iterrows():
    fl.Marker([row['lat'], row['lon']], popup=row.get('name', ''),
              icon=fl.Icon(color='blue')).add_to(folium_map)

for _, row in pop_without_access_pdf.iterrows():
    fl.CircleMarker(location=[row['ycoord'], row['xcoord']], radius=5,
                    color=None, fill=True, fill_color='red',
                    fill_opacity=row['opacity']).add_to(folium_map)

for _, row in pop_with_access_pdf.iterrows():
    fl.CircleMarker(location=[row['ycoord'], row['xcoord']], radius=5,
                    color=None, fill=True, fill_color='green',
                    fill_opacity=row['opacity']).add_to(folium_map)

folium_map

# COMMAND ----------

all_covered_ids_sdf = (hosp_coverage_sdf.select('pop_ID')
                        .union(potential_coverage_sdf.select('pop_ID'))
                        .distinct())

max_covered_pop     = (population_aoi_sdf
                       .join(all_covered_ids_sdf,
                             population_aoi_sdf['ID'] == all_covered_ids_sdf['pop_ID'], 'inner')
                       .agg(F.sum('population'))
                       .collect()[0][0])

max_access_possible = round(max_covered_pop * 100 / total_population, 2)
print(f'Maximum access attainable with potential location list: {max_access_possible}%')


# COMMAND ----------


print("Collecting Gurobi inputs from Spark to driver...")

# w: population weight per demand point
w_rows = population_aoi_sdf.select('ID', 'population').collect()
w      = {row['ID']: float(row['population']) for row in w_rows}
I      = sorted(w.keys())

# Facility ID sets
hosp_id_rows      = selected_hosp_sdf.select('ID').collect()
potential_id_rows = potential_locations_sdf.select('ID').collect()
J_existing        = sorted(row['ID'] for row in hosp_id_rows)
J_potential       = sorted(row['ID'] for row in potential_id_rows)
J                 = sorted(set(J_existing) | set(J_potential))

# JI: pop_ID → list of facility IDs that cover it
# Built from the union of hospital and potential coverage spatial joins
all_coverage_sdf = (hosp_coverage_sdf
                    .select('facility_ID', 'pop_ID')
                    .union(potential_coverage_sdf.select('facility_ID', 'pop_ID')))

JI_rows = (all_coverage_sdf
           .groupBy('pop_ID')
           .agg(F.collect_list('facility_ID').alias('fac_ids'))
           .collect())

JI = {row['pop_ID']: sorted(row['fac_ids']) for row in JI_rows}

print(f"Gurobi inputs ready | Demand points (I): {len(I):,} | Facilities (J): {len(J):,}")


# COMMAND ----------

#------------------
# GUROBI MCLP OPTIMIZATION LOOP
# Iterates over increasing numbers of new facilities (p),
# solving the Maximum Covering Location Problem at each step.

params = {
    "WLSACCESSID": "REDACTED",
    "WLSSECRET": "REDACTED",
    "LICENSEID": 0       # <-- Fill in your Gurobi WLS license ID
}

env   = gp.Env(params=params)
model = gp.Model(env=env)

pareto_gurobi = []
previous_obj  = -1

for p in range(1, len(J_potential) + 1):
    obj, selected_facilities, covered_demand = model_max_covering_gurobi(
        w, I, J, JI, p, J_existing, model
    )
    pareto_gurobi.append({
        'p': p,
        'objective': obj,
        'selected_facilities': selected_facilities,
        'covered_demand': covered_demand
    })

    if round(obj) == round(previous_obj):
        print("No further improvement. Stopping.")
        break

    previous_obj = obj
    print(f"Gurobi | p={p} | Covered pop={obj:.0f} | Total facilities: {len(selected_facilities)}")

#------------------
# PLOTLY PARETO FRONTIER

x_values = [len(J_existing) + item['p'] for item in pareto_gurobi]
y_values = [round(100 * item['objective'] / total_population, 2) for item in pareto_gurobi]

x_values.insert(0, len(J_existing))
y_values.insert(0, current_access)

fig = go.Figure(data=go.Scatter(x=x_values, y=y_values, mode='lines+markers',
                                 name='Gurobi Pareto Frontier'))
fig.update_layout(
    xaxis_title="Number of facilities (existing + new)",
    yaxis_title="Percentage of population with access",
    plot_bgcolor='white',
    yaxis=dict(range=[0, 100]),
    xaxis=dict(range=[0, len(J_potential) + len(J_existing)]),
    width=1200
)
fig.add_vline(x=len(J_existing), line_width=3, line_dash="dash", line_color="green")
fig.add_annotation(
    x=len(J_existing), y=current_access,
    text="Number of<br>existing facilities",
    showarrow=True, arrowhead=1, ax=90, ay=-30
)
fig.add_hline(y=max_access_possible, line_width=3, line_dash="dash", line_color="green")
fig.add_annotation(
    x=len(J_existing) + 20, y=max_access_possible,
    text="Maximum access possible with potential location list",
    showarrow=True, arrowhead=1, ax=0, ay=-30
)
fig.show()

# COMMAND ----------


target_p = 7

entry           = next(item for item in pareto_gurobi if item['p'] == target_p)
opened_ids      = set(entry['selected_facilities'])
covered_ids_set = set(entry['covered_demand'])

# Identify new vs existing among opened facilities
new_facility_ids = opened_ids - set(J_existing)

# Fetch new facility geometries from Spark as pandas (small set)
new_fac_sdf = potential_locations_sdf.filter(F.col('ID').isin(new_facility_ids))
new_fac_pdf = (new_fac_sdf
               .withColumn('lon', F.expr("ST_X(geometry)"))
               .withColumn('lat', F.expr("ST_Y(geometry)"))
               .select('ID', 'lon', 'lat')
               .toPandas())

# Population split for p=target_p result
covered_ids_sdf   = spark.createDataFrame(pd.DataFrame({'pop_ID': list(covered_ids_set)}))
pop_covered_sdf   = population_aoi_sdf.join(covered_ids_sdf,
                                             population_aoi_sdf['ID'] == covered_ids_sdf['pop_ID'], 'inner').drop('pop_ID')
pop_uncovered_sdf = population_aoi_sdf.join(covered_ids_sdf,
                                             population_aoi_sdf['ID'] == covered_ids_sdf['pop_ID'], 'left_anti')

pop_covered_pdf   = pop_covered_sdf.sample(fraction=min(1.0, _POP_VIZ_SAMPLE / max(1, pop_covered_sdf.count())),   seed=3).toPandas()
pop_uncovered_pdf = pop_uncovered_sdf.sample(fraction=min(1.0, _POP_VIZ_SAMPLE / max(1, pop_uncovered_sdf.count())), seed=4).toPandas()

# Build Folium map
folium_map = fl.Map(location=[center_lat, center_lon], zoom_start=8, tiles="OpenStreetMap")
geo_adm = fl.GeoJson(data=selected_gadm_pdf.iloc[0]['geometry'].__geo_interface__,
                      style_function=lambda x: {'color': 'orange'})
geo_adm.add_to(folium_map)

# Existing facilities (blue)
for _, row in selected_hosp_pdf_viz.iterrows():
    fl.Marker([row['lat'], row['lon']], popup=row.get('name', ''),
              icon=fl.Icon(color='blue')).add_to(folium_map)

# New Gurobi-selected facilities (dark purple)
for _, row in new_fac_pdf.iterrows():
    fl.Marker([row['lat'], row['lon']], icon=fl.Icon(color='darkpurple')).add_to(folium_map)

# Uncovered population (red)
for _, row in pop_uncovered_pdf.iterrows():
    fl.CircleMarker(location=[row['ycoord'], row['xcoord']], radius=5,
                    color=None, fill=True, fill_color='red',
                    fill_opacity=row['opacity']).add_to(folium_map)

# Covered population (green)
for _, row in pop_covered_pdf.iterrows():
    fl.CircleMarker(location=[row['ycoord'], row['xcoord']], radius=5,
                    color=None, fill=True, fill_color='green',
                    fill_opacity=row['opacity']).add_to(folium_map)

folium_map