import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px


st.title('Week 13 | Lab - Streamlit and Backblaze')


df = pd.read_csv(r'C:\Users\prasa\Desktop\simplestreamlit\simplestreamlit\train.csv',)

print(df['defects'].value_counts())

st.title('Class distribution of the data')

#histogram
fig, ax = plt.subplots()
df['defects'].value_counts(normalize = True).plot(kind = 'bar', color = ['steelblue', 'orange'],ax=ax)
plt.ylabel('Percentage')
plt.xlabel('Class')
plt.title('Defects Distribution')

st.pyplot(fig)


# Dropping the index and target columns
corr_mat = df.drop(columns = [ 'defects'], axis = 1).corr()

data_mask = np.triu(np.ones_like(corr_mat, dtype = bool))
cmap = sns.diverging_palette(100, 7, s = 75, l = 40, n = 20, center = 'light', as_cmap = True)
fig2, ax2 = plt.subplots(figsize = (18, 13))
sns.heatmap(corr_mat, annot = True, cmap = cmap, fmt = '.2f', center = 0,
            annot_kws = {'size': 18}, mask = data_mask).set_title('Correlations Among Input Features');

st.pyplot(fig2)



st.title('Feature Distributions after Log Transformation')

data = df
# Create a function for log transformation and plotting
def log_transform_and_plot(column):
    fig, ax = plt.subplots()
    ax.hist(np.log1p(data[column]), bins=100, color='magenta')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    ax.set_title(f'{column} Distribution after Log Transformation')
    return fig

# Display log-transformed feature distributions in Streamlit
for col in data.columns:
    st.pyplot(log_transform_and_plot(col))





