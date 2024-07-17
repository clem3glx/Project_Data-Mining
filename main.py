# main.py

# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import csv

# Part I: Initial Data Exploration

def detect_delimiter(file):
    sample = file.read(1024).decode('utf-8')
    file.seek(0)
    possible_delimiters = [',', ';', '/', '|']

    delimiter_counts = {delimiter: sample.count(delimiter) for delimiter in possible_delimiters}
    detected_delimiter = max(delimiter_counts, key=delimiter_counts.get)
    return detected_delimiter

def load_data():
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        delimiter = detect_delimiter(file)
        delimiter = st.text_input("Delimiter detected:", value=delimiter)

        header = st.text_input("Header row (default is 0)", value="0")
        data = pd.read_csv(file, header= int(header), delimiter=delimiter)
        st.dataframe(data.head())
        st.dataframe(data.tail())
        return data
    return None

def data_description(data):
    st.write("## Data Overview")

    # Split into two columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Number of rows", data.shape[0])
        st.metric("Number of columns", data.shape[1])

    with col2:
        st.write("### Column Names")
        st.markdown("<ul style='list-style-position: inside;'>", unsafe_allow_html=True)
        for col in data.columns:
            st.markdown(f"<li style='font-family: monospace;'>{col}</li>", unsafe_allow_html=True)
        st.markdown("</ul>", unsafe_allow_html=True)

    st.write("### Missing Values per Column")
    missing_values = data.isnull().sum()
    missing_values_df = pd.DataFrame(missing_values, columns=['Missing Values'])
    st.table(missing_values_df)

# Part II: Data Pre-processing and Cleaning
def handle_missing_values(data):
    method = st.selectbox("Choose method to handle missing values", ["Delete rows", "Delete columns", "Mean", "Median", "Mode", "KNN Imputation", "Simple Imputation"])
    if method == "Delete rows":
        data = data.dropna()
    elif method == "Delete columns":
        data = data.dropna(axis=1)
    elif method in ["Mean", "Median", "Mode"]:
        strategy = method.lower()
        imputer = SimpleImputer(strategy=strategy)
        data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    elif method == "KNN Imputation":
        imputer = KNNImputer()
        data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    elif method == "Simple Imputation":
        value = st.text_input("Value to replace missing values")
        data = data.fillna(value)
    return data

def normalize_data(data):
    method = st.selectbox("Choose normalization method", ["Min-Max", "Z-score"])
    
    # Drop any rows or columns that are completely empty
    data = data.dropna(how='all')
    data = data.dropna(axis=1, how='all')
    
    # Keep only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        st.error("No numeric columns available for normalization.")
        return data
    
    if method == "Min-Max":
        scaler = MinMaxScaler()
    elif method == "Z-score":
        scaler = StandardScaler()
        
    try:
        normalized_data = pd.DataFrame(scaler.fit_transform(numeric_data), columns=numeric_data.columns)
        data[numeric_data.columns] = normalized_data
    except ValueError as e:
        st.error(f"Error in normalization: {e}")
        return data

    st.write("Data after normalization:")
    st.dataframe(data.head())
    st.dataframe(data.tail())
    
    return data

# Part III: Visualization of the Cleaned Data
def plot_histogram(data):
    column = st.selectbox("Choose column for histogram", data.columns)
    fig, ax = plt.subplots()
    sns.histplot(data[column], kde=True, ax=ax)
    st.pyplot(fig)

def plot_boxplot(data):
    column = st.selectbox("Choose column for box plot", data.columns)
    fig, ax = plt.subplots()
    sns.boxplot(x=data[column], ax=ax)
    st.pyplot(fig)

# Part IV: Clustering or Prediction
def clustering(data):
    algorithm = st.selectbox("Choose clustering algorithm", ["K-means", "DBSCAN"])
    if algorithm == "K-means":
        n_clusters = st.slider("Number of clusters", 2, 10, 3)
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(data)
    elif algorithm == "DBSCAN":
        eps = st.slider("Epsilon", 0.1, 10.0, 0.5)
        min_samples = st.slider("Minimum samples", 1, 10, 5)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data)
    data['Cluster'] = labels
    return data, labels

def visualize_clusters(data):
    fig = px.scatter(data, x=data.columns[0], y=data.columns[1], color='Cluster')
    st.plotly_chart(fig)

def cluster_statistics(data, labels):
    st.write("Number of data points in each cluster: ", np.bincount(labels))
    if 'Cluster' in data.columns:
        centers = data.groupby('Cluster').mean()
        st.write("Centers of each cluster: ")
        st.dataframe(centers)

# Main function to run the app
def main():
    st.title("Data Mining Project")
    
    data = load_data()
    if data is not None:
        data_description(data)
        
        if st.checkbox("Pre-process and Clean Data"):
            st.header("Data Pre-processing and Cleaning")
            data = handle_missing_values(data)
            data = normalize_data(data)
        
        if st.checkbox("Visualize Cleaned Data"):
            st.header("Visualization of the Cleaned Data")
            plot_histogram(data)
            plot_boxplot(data)
        
        if st.checkbox("Perform Clustering or Prediction"):
            # drop column with other than float type
            data = data.select_dtypes(include=[np.number])
            st.header("Clustering or Prediction")
            data, labels = clustering(data)
        
        if st.checkbox("Evaluate Learning"):
            st.header("Learning Evaluation")
            visualize_clusters(data)
            cluster_statistics(data, labels)


if __name__ == "__main__":
    main()
