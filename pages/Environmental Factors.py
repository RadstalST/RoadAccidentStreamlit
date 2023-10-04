import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_data
def loadData():
    roadAccidents_df = pd.read_csv("out/crashsite_df.csv")
    roadAccidents_df['ACCIDENT_DATE'] = pd.to_datetime(roadAccidents_df['ACCIDENT_DATE'])
    roadAccidents_df['ACCIDENT_YEAR'] = roadAccidents_df['ACCIDENT_DATE'].dt.year
    roadAccidents_df['ACCIDENT_YEARMONTH'] = roadAccidents_df['ACCIDENT_DATE'].dt.strftime('%Y-%m')
    roadAccidents_df['ACCIDENT_MONTH'] = roadAccidents_df['ACCIDENT_DATE'].dt.strftime('%m')
    return roadAccidents_df


st.title("Environmental Factors and Clustering") 
st.subheader("Environmental Factors")
st.markdown("""

The following is the analysis of environmental factors that contribute to road accidents and their severity.

Steps would be
1. find out environmental factor such as road surface condition, weather condition, light condition, and traffic control.
2. find out the relationship between environmental factor and accident severity.
3. reduce the number of environmental factor combination to find out the most significant factor.
4. environmental factor combination with accident severity.
""".strip ())


crashsite_df = loadData()


environmentalColumns = [ 'ALCOHOLTIME',
        'LIGHT_CONDITION',
       'ROAD_GEOMETRY', 'SPEED_ZONE',"RMA"
       ]

noneEnvironmentalColumns = crashsite_df.columns.difference(environmentalColumns)


_df  = crashsite_df[environmentalColumns+["SEVERITY"]].copy()
_df = _df.dropna()
_df["count"] = 1

cols = environmentalColumns+["SEVERITY"]
fig = px.parallel_categories(_df, dimensions=cols)
# fig = px.sunburst(_df, path=cols,values="count")
st.plotly_chart(fig)
