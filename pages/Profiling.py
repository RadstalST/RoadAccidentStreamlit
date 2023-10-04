import pandas as pd
import pandas_profiling
import streamlit as st

from streamlit_pandas_profiling import st_profile_report

# @st.cache_data
def profilingReport():

    df = pd.read_csv("data/Road_Crashes_for_five_Years_Victoria.csv")
    pr = df.profile_report()
    return st_profile_report(pr)


profilingReport()