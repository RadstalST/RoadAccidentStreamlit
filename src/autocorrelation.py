# @dask.delayed
import streamlit as st
import pandas as pd
import geopandas as gpd
from dask.threaded import get
import plotly.express as px
from pysal.lib import weights
import plotly.graph_objects as go


@st.cache_data
def readCrashsiteGDF()->gpd.GeoDataFrame:
    crashsite_df = pd.read_csv("./data/Road_Crashes_for_five_Years_Victoria.csv",parse_dates=["ACCIDENT_DATE"],index_col="ACCIDENT_NO")
    crashsite_gdf = gpd.GeoDataFrame(
        crashsite_df, geometry=gpd.points_from_xy(crashsite_df.LONGITUDE, crashsite_df.LATITUDE),crs="EPSG:4326"
    )

    crashsite_gdf = crashsite_gdf.to_crs(3857)
    return crashsite_gdf

# @dask.delayed
@st.cache_data
def readHexGridGDF(path="./data/grid1KM.json"):
    grid = gpd.read_file(
        path,crs=None
    )
    grid["grid_id"] = grid.index
    return grid


# @dask.delayed
# @st.cache_data
def getDB(crashsite_gdf,hexgrid_gdf):
    joined = gpd.GeoDataFrame(
        hexgrid_gdf.sjoin(crashsite_gdf.to_crs(hexgrid_gdf.crs),how="left")
    )
    return joined
# @dask.delayed 
# @st.cache_data
def aggregation(joined_gdf,k=6,):
    
    agged = gpd.GeoDataFrame(
    joined_gdf.groupby("grid_id").agg({
        "FATALITY":"mean",
        "geometry":"first",
        "TOTAL_PERSONS": "mean",
        "INJ_OR_FATAL": "mean",
        "FATALITY": "mean",
        "SERIOUSINJURY": "mean",
        "index_right": "count",
    })
    )
    agged.rename(columns={"index_right":"count"},inplace=True)

    data_buffer = agged.copy().dropna().query("count>0")
    w = weights.KNN.from_dataframe(data_buffer, k=k) #k is the number of neighbour
    w.transform = "R" #normalise the weights matrix

    data_buffer["count_lag"] = weights.spatial_lag.lag_spatial(
        w, data_buffer["count"]
    )
    columns = ["FATALITY","TOTAL_PERSONS","INJ_OR_FATAL","FATALITY","SERIOUSINJURY"]
    for column in columns:
        data_buffer[column+"_lag"] = weights.spatial_lag.lag_spatial(
            w, data_buffer[column]
        )

    qx = data_buffer["count"].mean()
    qy = data_buffer["count_lag"].mean()
    def get_quantile(x):

        if(x["count"]>qx and x["count_lag"]>qy):
            return "HotSpot"
        elif(x["count"]<=qx and x["count_lag"]<=qy):
            return "ColdSpot"
        elif(x["count"]>qx and x["count_lag"]<=qy):
            return "EmergingHotSpot"
        elif(x["count"]<=qx and x["count_lag"]>qy):
            return "DecliningHotSpot"
    data_buffer["count_id_quantile"] = data_buffer.apply(get_quantile,axis=1)
    return data_buffer

# @st.cache_data
def make_plot(groupedHexGrid_gdf,column_name="count",title=None,zoom=8):
    # Assuming you have your GeoDataFrame named 'groupedHexGrid_gdf'
    # Convert GeoDataFrame to GeoJSON for Plotly
    geojson = groupedHexGrid_gdf.__geo_interface__
    # Create a Choroplethmapbox trace

    title = title if title else f"{column_name.replace('_',' ')} Rank Pct"

    choropleth_trace = go.Choroplethmapbox(
        geojson=geojson,
        locations=groupedHexGrid_gdf.index,  # Spatial coordinates
        z=(groupedHexGrid_gdf[column_name].rank(pct=True)),        
        colorscale="RdBu_r",
        colorbar={"title": title},
        marker_opacity=0.3,
        marker_line_width=2,
        marker_line_color="white",
        hovertemplate="<b>Crashsite "+column_name+"</b>: %{z}<br>" +"<b>Quantile</b>: %{z}<br>" +"<extra></extra>",
        # animation_frame=groupedHexGrid_gdf.
        
    )
    layout_choropleth = go.Layout(
        mapbox_style="carto-positron",
        mapbox_zoom=zoom,  # Adjust as needed
        mapbox_center={"lat": -37.8, "lon": 144.95},
        margin={"r": 0, "t": 0, "l": 0, "b": 0},

    )
    # Create the figure for the choropleth map
    fig_choropleth = go.Figure(data=[choropleth_trace], layout=layout_choropleth)
    return fig_choropleth


@st.cache_data
def getResults():
    dsk = {'crashsite_gdf': (readCrashsiteGDF,),
        'hexgrid_gdf': (readHexGridGDF,),
        'joined_gdf': (getDB, 'crashsite_gdf', 'hexgrid_gdf'),
        'groupedHexGrid_gdf': (aggregation, 'joined_gdf'),
        "plot": (make_plot, "groupedHexGrid_gdf","INJ_OR_FATAL"),
        "lagplot": (make_plot, "groupedHexGrid_gdf", "INJ_OR_FATAL_lag")
        }
    
    joined_gdf,groupedHexGrid_gdf =  get(dsk, ['joined_gdf',"groupedHexGrid_gdf"])
    return joined_gdf,groupedHexGrid_gdf
