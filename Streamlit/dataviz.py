import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Page "Dataviz"
def app(df) : 
    st.write("### DataViz")

    df1 = df[[ 'Co2', 'MassRunningOrder','EngineCapacity', 'EnginePower']]
    fig, ax = plt.subplots()
    sns.heatmap(df1.corr(), ax=ax)
    st.write(fig)

    # fig = plt.figure()
    # sns.countplot(x = 'trucd', hue = 'truc', data = df)
    # st.pyplot(fig)
 
    # fig = sns.displot(x = 'truc', data = df)
    # plt.title("Distribution de qquchose")
    # st.pyplot(fig)

    # fig = sns.catplot(x='col1', y='col2', data=df, kind='point')
    # st.pyplot(fig)

    # fig = sns.lmplot(x='col5', y='col4', hue="col3", data=df)
    # st.pyplot(fig)

    # fig, ax = plt.subplots()
    # sns.heatmap(df1.corr(), ax=ax)
    # st.write(fig)