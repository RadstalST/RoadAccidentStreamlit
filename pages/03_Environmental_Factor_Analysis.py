import streamlit as st
import pandas as pd
import plotly.express as px
# import linear regression related libraries
from sklearn.linear_model import LinearRegression
@st.cache_resource
def loadData():
    roadAccidents_df = pd.read_parquet("out/crashsite_df_envcluster.parquet")
    roadAccidents_df['ACCIDENT_DATE'] = pd.to_datetime(roadAccidents_df['ACCIDENT_DATE'])
    roadAccidents_df['ACCIDENT_YEAR'] = roadAccidents_df['ACCIDENT_DATE'].dt.year
    roadAccidents_df['ACCIDENT_YEARMONTH'] = roadAccidents_df['ACCIDENT_DATE'].dt.strftime('%Y-%m')
    roadAccidents_df['ACCIDENT_MONTH'] = roadAccidents_df['ACCIDENT_DATE'].dt.strftime('%m')
    roadAccidents_df["labels"] = roadAccidents_df["predicted_environmental_cluster"].apply(lambda x: "Cluster #" + str(x))
    roadAccidents_df["labels"][roadAccidents_df["labels"]=="Cluster #-1"] = "Noise"
    return roadAccidents_df
@st.cache_resource
def loadTsneData():
    tsne_df = pd.read_parquet("out/tsne_df_10K.parquet")
    tsne_df["labels"] = tsne_df["labels"].apply(lambda x: "Cluster #" + str(x))
    tsne_df["labels"][tsne_df["labels"]=="Cluster #-1"] = "Noise"
    return tsne_df
@st.cache_resource
def getGroupedSeverityData():
    grouped = crashsite_df.groupby("labels").agg(
    {
        "TOTAL_PERSONS":"mean",
        "INJ_OR_FATAL":"mean",
        "ACCIDENT_DATE":"count",
        "POLICE_ATTEND":lambda x: (x=="Yes").sum()/len(x),
        "SEVERITY": lambda x: (x=="Fatal accident").sum()/len(x),
        }

    ).sort_values(by="SEVERITY",ascending=False)

    grouped["count"] = grouped["ACCIDENT_DATE"].astype(int)
    grouped.drop("Noise",inplace=True)
    return grouped
    
crashsite_df = loadData()
tsne_df = loadTsneData()


headerContainer = st.container()
envEDAContainer = st.container()
tsneContainer = st.container()
clusterPredictionContainer = st.container()
environmentalFactorCorrelationContainer = st.container()
with headerContainer:
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


with envEDAContainer:
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
    st.plotly_chart(fig, use_container_width=True)

    
    with st.expander("Polar Chart of Each Environmental Factor"):

        for i,col in enumerate(environmentalColumns):
            _series = crashsite_df[col].value_counts()
            fig = px.line_polar(r=_series.values,theta=_series.index, line_close=True, title=f"{col} Distribution")
            st.plotly_chart(fig, use_container_width=True)

with tsneContainer:
    st.subheader("Clustering and Dimensionality Reduction")
    st.markdown("""
The following is the analysis of clustering and dimensionality reduction of environmental factors.
1. Dimensionality reduction using t-SNE by sampling 10000 data points into 3 components.
2. Clustering using DBSCAN to group the data points into similar groups.
3. 3D scatter plot of t-SNE with cluster labels.
                """.strip())
    with st.expander("What is DBSCAN?"):
        st.markdown("""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a clustering algorithm that groups data points into clusters based on their density.
                    which is different from other clustering algorithms such as K-Means, which groups data points into clusters based on their distance from the centroid.
                    """)
        
    with st.expander("What is T-SNE?"):
        st.markdown("""
t-SNE (t-Distributed Stochastic Neighbor Embedding) is a dimensionality reduction algorithm that reduces the dimensionality of data points while preserving their local structure.
                    in this case of reducting categorical data, T-SNE is able to handle the `categorical data` and reduce the dimensionality of the data points into 3 components.
                    """)
    fig = px.scatter_3d(
        tsne_df,
        x='x', y='y', z='z',
        color='labels',
        opacity=0.2)
    fig.update_layout(
        title="DBSCAN of Reduced Dimentionality of Environmental Features",
        scene = dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z'
        ),
        height=800
    )
    # legend name
    fig.for_each_trace(lambda t: t.update(name=t.name.replace("labels=", "Cluster ")))
    st.plotly_chart(fig)

    with st.expander("What does this 3D scatter plot mean?",expanded=True):
        st.markdown("""
1. Each data point is represented by a circle.
2. Each data point is colored by their cluster label.
3. the Cluster label represents similar data group
                    
>This method of analysis reduce the number of total combination of environmental factors from 8190 to `161 common combination`.
      
ALCOHOLTIME         2
LIGHT_CONDITION     7
ROAD_GEOMETRY       9
SPEED_ZONE         13
RMA                 5
number of combinations 8190
number of clusters 161
""")



with clusterPredictionContainer:


    st.subheader("Predicted Environmental Cluster Analysis")
    st.write("In the graph below, the x-axis represents the predicted environmental cluster, and the y-axis represents the severity of the accident.")
    st.write("The color of the bar represents the number of accidents in each cluster.")
    st.write("The graph shows that the clustering algorithm is able to group the accidents into similar groups and shows that the severity of the accidents in each cluster is different and can be ranked.")
    
    grouped = getGroupedSeverityData()
    fig = px.bar(
        grouped,
        x=grouped.index,
        y="SEVERITY",
        labels={"x":"labels","y":"SEVERITY"},
        color="count"
        )   
    fig.update_layout(
        title="Predicted Environmental Cluster Analysis")

    st.plotly_chart(fig, use_container_width=True)

@st.cache_data(persist=True)
def trainRegression(_df, metrices, norm,metrice="INJ_OR_FATAL"):
    metrices_df = _df[metrices].div(_df[norm], axis=0) # normalize metrices
    X = pd.get_dummies(crashsite_df[[*environ_col,"labels"]],columns=environ_col).groupby("labels").mean() # one hot encode environmental factors
    # X = pd.get_dummies(crashsite_df[[*environ_col]],columns=environ_col) # one hot encode environmental factors
    y = metrices_df[metrice].groupby(crashsite_df["predicted_environmental_cluster"]).mean() # get the mean of the metrices
    reg = LinearRegression()
    reg.fit(X, y)
    return reg

with environmentalFactorCorrelationContainer:
    st.subheader("Environmental Factor Correlation")

    environ_col = ['ALCOHOLTIME', 'LIGHT_CONDITION', 'ROAD_GEOMETRY', 'SPEED_ZONE', 'RMA']
    metrices = ['INJ_OR_FATAL', 'FATALITY', 'SERIOUSINJURY', 'OTHERINJURY','NONINJURED']
    norm = 'TOTAL_PERSONS'
    models = {}

    with st.status("Training Linear Regression Model"):
        st.write("training linear regression model")
        for metrice in metrices:
            st.write(f"training linear regression model for {metrice}")
            models[metrice] = { "model":trainRegression(crashsite_df, metrices, norm, metrice) }

    with st.expander("Trained Model"):
        st.write(models)


    st.write("The following is the correlation between environmental factors and accident severity.")
    st.write("The correlation is calculated by training a linear regression model for each environmental factor and accident severity.")
    st.write("The correlation is calculated by the coefficient of the linear regression model.")


    X = pd.get_dummies(crashsite_df[environ_col])
    X.head()

    tabs = st.tabs(metrices)
    for tab,matrice in zip(tabs,metrices):
        with tab:
            coef_df = pd.DataFrame(models[matrice]["model"].coef_, X.columns, columns=['Coefficient'])
            coef_df.sort_values("Coefficient",inplace=True,ascending=False) #sort by impact

            
            with st.expander("Overall Impact for each Enviromental Factors"):
                fig = px.bar(
                    coef_df,
                    x=coef_df.index,
                    y=coef_df["Coefficient"],
                    title=f"coefficent to {matrice}",
                    color=coef_df["Coefficient"],
                    color_continuous_scale="RdBu_r",
                    color_continuous_midpoint=0
                    )
                
                
                # taller fig
                fig.update_layout(
                    height=800
                )
                st.plotly_chart(fig)

            coef_df.index = pd.MultiIndex.from_tuples([ (" ".join(i.split("_")[:-1]).capitalize(),i.split("_")[-1]) for i in coef_df.index ])
            coef_level0Index = coef_df.index.get_level_values(0).unique()

            
            models[matrice]["coef"] = coef_df
            with st.expander("raw coef"):
                st.table(coef_df)

            with st.expander("Impact of Separate Environmental Factor", expanded=True):
                for idx  in coef_level0Index:
                    coef_df.sort_values("Coefficient",inplace=True,ascending=False)
                    fig = px.bar(
                        coef_df.loc[(idx,)],
                        title=f"coefficent of {idx} to {matrice}",
                        color=coef_df.loc[(idx,),"Coefficient"],
                        color_continuous_scale="RdBu_r",
                        color_continuous_midpoint=0
                        )
                        
                    st.plotly_chart(fig)
            
    st.caption("From the analysis we can quickly see the most impactful for each metrices.")
            #plotly barchart for each level 0 index

# 

        



# create a dictionary of models



