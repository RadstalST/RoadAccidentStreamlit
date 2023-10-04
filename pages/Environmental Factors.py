import streamlit as st

@st.cache_data
def loadData():
    roadAccidents_df = pd.read_csv("data/Road_Crashes_for_five_Years_Victoria.csv")
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
"""
