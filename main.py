import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

@st.cache_data
def loadData():
    roadAccidents_df = pd.read_csv("data/Road_Crashes_for_five_Years_Victoria.csv")
    roadAccidents_df['ACCIDENT_DATE'] = pd.to_datetime(roadAccidents_df['ACCIDENT_DATE'])
    roadAccidents_df['ACCIDENT_YEAR'] = roadAccidents_df['ACCIDENT_DATE'].dt.year
    roadAccidents_df['ACCIDENT_YEARMONTH'] = roadAccidents_df['ACCIDENT_DATE'].dt.strftime('%Y-%m')
    roadAccidents_df['ACCIDENT_MONTH'] = roadAccidents_df['ACCIDENT_DATE'].dt.strftime('%m')
    return roadAccidents_df



def header():
    st.title("Road Accidents Project")

def introduction():
    st.header("Introduction")
    st.write("This is a project to help you understand the road accident in Victoria by various analysis and visualisation.")
def projectScope():

    st.header("Project Scope")
    st.markdown("""
The Project Scope is aligned with the Victoria Government Road Safety Strategy, which aims to reduce the number of road deaths and serious injuries. This project will focus on the following areas:

1. Identifying the most frequent locations of road accidents.
2. Identifying the environmental factors that contribute to road accidents and their severity.
3. Identifying the spatial trends of road accidents in Victoria.

By analyzing the above factors, we can provide the following insights to the government:

1. The most frequent locations of road accidents, so that the government can take actions to improve road conditions in those areas.
2. The environmental factors that contribute to road accidents and their severity, so that the government can take actions to improve road conditions and reduce the likelihood of accidents.
3. The spatial trends of road accidents in Victoria, so that the government can take actions to improve road conditions in areas with high accident rates.

This project aims to provide valuable insights to the government, which can help them make informed decisions to improve road safety in Victoria.
""".strip())

def visualisation():
    st.header("Visualisation")
    st.subheader("Road Accidents of Victoria")
    roadAccidents_df = loadData()

    with st.expander("Show raw data"):
        st.dataframe(roadAccidents_df)

    st.subheader("Road Accidents Analysis")
    st.write("The following is the analysis of road accidents in Victoria in time series")

    roadAccidents_df_by_year = roadAccidents_df.groupby(['ACCIDENT_YEARMONTH']).agg({
        'TOTAL_PERSONS': 'sum',
        "FATALITY" : "sum",
        'SERIOUSINJURY': 'sum',
        'OTHERINJURY': 'sum',
        'NONINJURED': 'sum'
        }).reset_index()
    

    # number of casualties by year using plotly using stacked bar chart
    fig = px.bar(
        roadAccidents_df_by_year, 
        x="ACCIDENT_YEARMONTH", 
        y=["FATALITY", "SERIOUSINJURY", "OTHERINJURY", "NONINJURED"], 
        title="Number of casualties by year month", 
        barmode='stack')
    
    # add line chart for total number of casualties
    fig.add_scatter(x=roadAccidents_df_by_year['ACCIDENT_YEARMONTH'], y=roadAccidents_df_by_year['TOTAL_PERSONS'], name="TOTAL_PERSONS", mode='lines')

    fig.update_layout(
        xaxis_title="Year Month",
        yaxis_title="Number of casualties",
        legend_title="Casualties Type"
    )
    st.plotly_chart(fig)

    roadAccidents_df_by_month = roadAccidents_df.groupby(['ACCIDENT_MONTH']).agg({
        'TOTAL_PERSONS': 'mean',
        "FATALITY" : "mean",
        'SERIOUSINJURY': 'mean',
        'OTHERINJURY': 'mean',
        'NONINJURED': 'mean',
        }).reset_index()
    

    fig = px.bar(
        roadAccidents_df_by_month, 
        x="ACCIDENT_MONTH", 
        y=["FATALITY", "SERIOUSINJURY", "OTHERINJURY", "NONINJURED"], 
        title="Average Number of Casualties by month", 
        barmode='stack')
    fig.add_scatter(x=roadAccidents_df_by_month['ACCIDENT_MONTH'], y=roadAccidents_df_by_month['TOTAL_PERSONS'], name="TOTAL_PERSONS", mode='lines')
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Average Number of casualties",
        legend_title="Casualties Type"
    )
    st.plotly_chart(fig)

    #type of accidents
    st.subheader("Type of Accidents")

    roadAccidents_df_by_accident_type = roadAccidents_df.groupby(['ACCIDENT_TYPE']).agg({
        'TOTAL_PERSONS': 'sum',
        "FATALITY" : "sum",
        'SERIOUSINJURY': 'sum',
        'OTHERINJURY': 'sum',
        'NONINJURED': 'sum',
        }).reset_index()
    
    fig = px.bar(
        roadAccidents_df_by_accident_type.sort_values(by=['TOTAL_PERSONS'], ascending=False),
        x="ACCIDENT_TYPE", 
        y=["FATALITY", "SERIOUSINJURY", "OTHERINJURY", "NONINJURED"], 
        title="Average Number of Casualties by Accident Type", 
        barmode='stack')
    # fig.add_scatter(x=roadAccidents_df_by_accident_type['ACCIDENT_TYPE'], y=roadAccidents_df_by_accident_type['TOTAL_PERSONS'], name="TOTAL_PERSONS", mode='lines')
    fig.update_layout(
        xaxis_title="Accident Type",
        yaxis_title="Average Number of casualties in log scale",
        legend_title="Casualties Type",
    )
    # log y
    fig.update_yaxes(type="log")
    # add number to the bar

    fig.update_traces(
        texttemplate='%{y:.2s}',
        textposition='outside'
    )


    st.plotly_chart(fig)

    # rank of RMA type by number of casualties in each category 
    st.subheader("Rank of RMA Type by Number of Casualties in Each Category")

    roadAccidents_df_by_rma_type = roadAccidents_df.groupby(['RMA_ALL']).agg({
        'TOTAL_PERSONS': 'sum',
        "FATALITY" : "sum",
        'SERIOUSINJURY': 'sum',
        'OTHERINJURY': 'sum',
        'NONINJURED': 'sum',
        }).reset_index()
    
    # horizontal bar chart
    fig = px.bar(
        roadAccidents_df_by_rma_type.sort_values(by=['TOTAL_PERSONS'], ascending=True),
        y="RMA_ALL", 
        x=["FATALITY", "SERIOUSINJURY", "OTHERINJURY", "NONINJURED"], 
        title="Average Number of Casualties by RMA Type", 
        barmode='stack',
        orientation='h',
        height=800,
        )
        
    
    fig.update_layout(
        yaxis_title="RMA Type",
        xaxis_title="Number of casualties",
        legend_title="Casualties Type",
    )

    fig.update_traces(
        texttemplate='%{x:.2s}',
        textposition='outside'
    )

    fig.update_xaxes(type="log")
    st.plotly_chart(fig)
    

    # fig = px.bar(
    #     roadAccidents_df,
    #     x="LIGHT_CONDITION",
    #     y=["FATALITY", "SERIOUSINJURY", "OTHERINJURY", "NONINJURED"],
    #     title="Average Number of Casualties by Light Condition",
    #     barmode='stack',
    #     )
    
    # st.plotly_chart(fig)





def main():
    header()
    introduction()
    projectScope()
    st.divider()
    visualisation()


if __name__ == "__main__":
    main()