import streamlit as st 
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import plotly.express as px
import plotly.graph_objects as go




import matplotlib.pyplot as plt
import seaborn
from pysal.viz import splot
from splot.esda import plot_moran
import contextily

import geopandas
import pandas
from pysal.explore import esda
from pysal.lib import weights
from numpy.random import seed
@st.cache_data
def get_data(num_years_offset=3):
    crashsite_df = pd.read_csv("data/Road_Crashes_for_five_Years_Victoria.csv",parse_dates=["ACCIDENT_DATE"])
    last_data = crashsite_df['ACCIDENT_DATE'].max()
    last_data_year = last_data - pd.DateOffset(years=num_years_offset)
    last_year_selection = (crashsite_df['ACCIDENT_DATE'] > last_data_year) & (crashsite_df['ACCIDENT_DATE'] <= last_data)
    lat_long_df = crashsite_df[last_year_selection][['LATITUDE', 'LONGITUDE']]
    return lat_long_df,crashsite_df

@st.cache_data
def get_cluster(params,lat_long_df):
    clustering = DBSCAN(
    **params,
    algorithm='ball_tree',
    metric='haversine' # haversine is the distance between two points on a sphere 
    ).fit(np.radians(lat_long_df)) #need to be radians for haversine
    return clustering

st.title('Welcome to the Road Accident Project!')
st.caption("This is a project to help you understand the road accident in Victoria by various analysis and visualisation.")
lat_long_df,crashsite_df = get_data()

with st.expander("Show raw data"):
    st.dataframe(crashsite_df)
with st.form("cluster_form"):

    st.write("Please select the parameters for DBSCAN")
    distance_in_meters = st.number_input("Distance in meters", min_value=0, max_value=1000, value=10, step=1)
    number_of_minimum_occourence = st.number_input("Number of minimum occourence", min_value=0, max_value=10, value=5, step=1)
    
    DBSCAN_params = {
        "eps": (distance_in_meters/1000)/6371,
        "min_samples": number_of_minimum_occourence
    }
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write(DBSCAN_params)




clustering = get_cluster(DBSCAN_params,lat_long_df)

_lat_long_df = lat_long_df.copy()

_lat_long_df["cluster"]=clustering.labels_
non_outliner_selector = _lat_long_df["cluster"]!=-1
_lat_long_df = _lat_long_df.loc[non_outliner_selector]
cluster_index = _lat_long_df.index
clusterd_crashsite_df =  crashsite_df.loc[cluster_index].join(_lat_long_df["cluster"])
cluster_info_df =  clusterd_crashsite_df.groupby("cluster").agg(
    {
        "TOTAL_PERSONS": "mean",
        "INJ_OR_FATAL": "mean",
        "FATALITY": "mean",
        "SERIOUSINJURY": "mean",
        "OTHERINJURY": "mean",
        "NONINJURED": "mean",
        "LATITUDE": "mean",
        "LONGITUDE": "mean",
        "ACCIDENT_NO": "count"
        
    }
)


map_center = cluster_info_df[['LATITUDE', 'LONGITUDE']].mean().values.tolist()


fig = px.scatter_mapbox(cluster_info_df, 
    lat="LATITUDE", lon="LONGITUDE", 
    # hover_name="INJ_OR_FATAL", 
    hover_data=["FATALITY", "SERIOUSINJURY","OTHERINJURY"],
    size=(cluster_info_df["ACCIDENT_NO"]),
    opacity=0.8,
    size_max=50,
    color="INJ_OR_FATAL", 
    color_continuous_scale="agsunset",
    zoom=3, height=600)

# add another layer of scatter plot
fig.add_scattermapbox(
    lat=crashsite_df['LATITUDE'],
    lon=crashsite_df['LONGITUDE'],
    mode='markers',
    name='data point',
    marker=go.scattermapbox.Marker(
        # x shape
        size=5,
        color='black',
        opacity=0.1
    ),
    hoverinfo='none'
)

fig.add_scattermapbox(
    lat=clusterd_crashsite_df['LATITUDE'],
    lon=clusterd_crashsite_df['LONGITUDE'],
    mode='markers',
    name='cluster point',
    marker=go.scattermapbox.Marker(
        size=10,
        color=clusterd_crashsite_df['cluster'],
        colorscale='tempo',
        opacity=1
    ),
    hoverinfo='none'
)

## horizontal colorbar
fig.update_layout(
    coloraxis_colorbar=dict(
        title="Average Number of People Injured or Killed",
        orientation="h",
        yanchor="bottom", y=1,
        # xanchor="right", x=0,
        ticks="outside", ticksuffix=" people",
        dtick=1
    )
)
# fig.update_layout(**{'orientation':'h'})
# fig.update_coloraxes(colorbar={'orientation':'h', 'thickness':20, 'y': -1.0})



fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.update_layout(mapbox_zoom=8)




st.header("Cluster Information Visualisation")
st.caption("""
The following is the information of each cluster of the road accident collected in Victoria. 
The size of the circle represents the number of accidents in the cluster. 
The color of the circle represents the average number of people injured or killed in the cluster. 
The darker the color, the more people are injured or killed in the cluster.
The green circle represents each data point that are similar to each cluster.
""")
tab1, tab2 = st.tabs(["With Map", "Without Map"])

with tab1:
    st.plotly_chart(fig, use_container_width=True)
with tab2:
    fig.update_layout(mapbox_style="white-bg")
    st.plotly_chart(fig, use_container_width=True)
