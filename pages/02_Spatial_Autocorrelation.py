import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from pysal.viz import splot
from splot.esda import plot_moran
import contextily
import os 
import geopandas as gpd
import pandas as pd
from pysal.explore import esda
from pysal.lib import weights
from numpy.random import seed
from dask.threaded import get
import plotly.express as px
import dask
from dask.graph_manipulation import bind
import plotly.graph_objects as go
import geopandas as gpd
from plotly.subplots import make_subplots
from src import autocorrelation
os.environ['USE_PYGEOS'] = '0'



headerContainer = st.container()
SpatialLagPlotContainer = st.container()

with headerContainer:
    st.title("Spatial Autocorrelation")
    with st.expander("What is Spatial Autocorrelation"):
        st.markdown("Spatial Autocorrelation, also known as spatial autocorrelation analysis, is a statistical technique used to examine and quantify the degree to which the values of a variable are correlated or related to their spatial locations. In other words, it assesses whether nearby observations are more similar to each other than those that are farther apart. This concept is often applied in fields like geography, ecology, urban planning, and epidemiology, where understanding spatial patterns and dependencies is crucial.")


joined_gdf,groupedHexGrid_gdf = autocorrelation.getResults()


with SpatialLagPlotContainer:
    zoom = st.slider("Zoom",min_value=6,max_value=20,value=8)

    spatialCountPlotFig = autocorrelation.make_plot(groupedHexGrid_gdf,"count",zoom=zoom)
    spatialCountPlotFig_lag= autocorrelation.make_plot(groupedHexGrid_gdf,"count_lag",zoom=zoom)
    with st.expander("Number of accidents mapped with hexagonal grid"):
        st.subheader("Number of accidents mapped with hexagonal grid")
        st.plotly_chart(spatialCountPlotFig)

    st.subheader("Autocorrelation of number of accidents mapped with hexagonal grid")
    st.plotly_chart(spatialCountPlotFig_lag)


    spatialCountPlotFig = autocorrelation.make_plot(groupedHexGrid_gdf,"INJ_OR_FATAL",zoom=zoom)
    spatialCountPlotFig_lag= autocorrelation.make_plot(groupedHexGrid_gdf,"INJ_OR_FATAL_lag",zoom=zoom)
    with st.expander("Number of injuries and fatality mapped with hexagonal grid"):
        st.subheader("Number of injuries and fatality mapped with hexagonal grid")
        st.plotly_chart(spatialCountPlotFig)

    st.subheader("Autocorrelation of number of injuries and fatality mapped with hexagonal grid")
    st.plotly_chart(spatialCountPlotFig_lag)

