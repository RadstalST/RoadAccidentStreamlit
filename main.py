import streamlit as st 
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import plotly.express as px
import plotly.graph_objects as go
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

st.title('Streamlit!')
lat_long_df,crashsite_df = get_data()
st.dataframe(crashsite_df)
with st.form("cluster_form"):

   
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
    opacity=0.7,
    size_max=50,
    color="INJ_OR_FATAL", 
    color_continuous_scale=px.colors.sequential.Viridis,
    zoom=3, height=600)

# add another layer of scatter plot
fig.add_scattermapbox(
    lat=crashsite_df['LATITUDE'],
    lon=crashsite_df['LONGITUDE'],
    mode='markers',
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
    marker=go.scattermapbox.Marker(
        size=10,
        color=clusterd_crashsite_df['cluster'],
        colorscale='tempo',
        opacity=1
    ),
    hoverinfo='none'
)


fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.update_layout(mapbox_zoom=8)

st.plotly_chart(fig, use_container_width=True)

fig.update_layout(mapbox_style="white-bg")
st.plotly_chart(fig, use_container_width=True)
