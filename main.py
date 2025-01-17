import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score
import seaborn as sns
import plotly.express as px
import csv
import io

# Part I: Initial Data Exploration

def detect_delimiter(file):
    sample = file.read(1024).decode('utf-8')
    file.seek(0)
    possible_delimiters = [',', ';', '/', '|']

    delimiter_counts = {delimiter: sample.count(delimiter) for delimiter in possible_delimiters}
    detected_delimiter = max(delimiter_counts, key=delimiter_counts.get)
    return detected_delimiter

def load_data():
    file = st.file_uploader("Upload CSV", type=["csv", "data"])
    if file is not None:
        delimiter = detect_delimiter(file)
        delimiter = st.text_input("Delimiter detected:", value=delimiter)

        header = st.text_input("Header row (default is 0)", value="0")
        
        try:
            data = pd.read_csv(file, header=int(header), delimiter=delimiter, on_bad_lines='warn')
            st.dataframe(data.head())
            st.dataframe(data.tail())
            return data
        except pd.errors.ParserError as e:
            st.error(f"Error parsing file: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
    return None

def data_description(data):
    st.markdown("<h3 style='padding-left: 95px;'>Data Overview</h3>", unsafe_allow_html=True)

    # Split into two columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Number of rows", data.shape[0])
        st.metric("Number of columns", data.shape[1])

    with col2:
        st.write("##### Column Names")
        st.markdown("<ul style='list-style-position: inside;'>", unsafe_allow_html=True)
        for col in data.columns:
            st.markdown(f"<li style='font-family: monospace;'>{col}</li>", unsafe_allow_html=True)
        st.markdown("</ul>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='padding-left: 95px;'>Missing Values per Column</h3>", unsafe_allow_html=True)
    missing_values = data.isnull().sum()
    missing_percentage = (missing_values / len(data)) * 100
    missing_values_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentage
    })
    st.table(missing_values_df)
    
    st.markdown("<h3 style='padding-left: 95px;'>Descriptive Statistics</h3>", unsafe_allow_html=True)
    st.dataframe(data.describe())

# Part II: Data Pre-processing and Cleaning
def handle_missing_values(data):
    st.markdown("<h3 style='padding-left: 95px;'>Drop Columns</h3>", unsafe_allow_html=True)
    column_to_drop = st.multiselect("Select column(s) to drop", data.columns)
    if column_to_drop:
        data = data.drop(columns=column_to_drop)

    st.markdown("<h3 style='padding-left: 95px;'>Handle Missing Values</h3>", unsafe_allow_html=True)
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
    method = st.selectbox("Choose normalization method", ["No normalization for the moment", "Min-Max", "Z-score"])
    
    # Drop any rows or columns that are completely empty
    data = data.dropna(how='all')
    data = data.dropna(axis=1, how='all')
    
    # Keep only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        st.error("No numeric columns available for normalization.")
        return data
    
    if method == "No normalization for the moment":
        return data
    elif method == "Min-Max":
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

def additional_visualizations(data):
    # Distribution des Composants de Béton
    # st.header('Distribution des Composants de Béton')
    components = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr.', 'Fine Aggr.']
    # for component in components:
    #         st.subheader(f'Distribution de {component}')
    #         fig, ax = plt.subplots()
    #         sns.histplot(data[component], bins=30, kde=True, ax=ax)
    #         st.pyplot(fig)

    # Relations entre les composants et les sorties
    # st.header('Relations entre les Composants et les Sorties')
    outputs = ['SLUMP(cm)', 'FLOW(cm)', 'Compressive Strength (28-day)(Mpa)']
    # for output in outputs:
    #     for component in components:
    #         st.subheader(f'Relation entre {component} et {output}')
    #         fig, ax = plt.subplots()
    #         sns.scatterplot(x=data[component].to_numpy(), y=data[output].to_numpy(), ax=ax)
    #         st.pyplot(fig)

    # Correlation Matrix
    st.header('Correlation matrix')
    fig, ax = plt.subplots()
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # # Analysis of Compressive Force at 28 Days
    # st.header('Analysis of Compressive Force at 28 Days')
    # fig, ax = plt.subplots()
    # sns.boxplot(data=data[['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr.', 'Fine Aggr.', 'Compressive Strength (28-day)(Mpa)']])
    # st.pyplot(fig)

    input_vars = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr.', 'Fine Aggr.']
    output_vars = ['SLUMP(cm)', 'FLOW(cm)', 'Compressive Strength (28-day)(Mpa)']
    st.subheader('Matrices de Corrélation')

    # Corrélation entre variables d'entrée et de sortie
    corr_in_out = data[input_vars + output_vars].corr().loc[input_vars, output_vars]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_in_out, annot=True, cmap='coolwarm', ax=ax)
    plt.title('Corrélations entre Variables d\'Entrée et de Sortie')
    st.pyplot(fig)

    output_vars = ['SLUMP(cm)', 'FLOW(cm)', 'Compressive Strength (28-day)(Mpa)']
    st.subheader('Régression')
    x = st.selectbox("Choose X-axis variable for regression", data.columns)
    y = st.selectbox("Choose Y-axis variable for regression", output_vars)
    fig, ax = plt.subplots()
    sns.regplot(x=data[x], y=data[y], ax=ax)
    st.pyplot(fig)

def slump_flow_comp(data):
    # Comparison between SLUMP and FLOW
    st.header('Comparison between SLUMP and FLOW')
    fig, ax = plt.subplots()
    sns.scatterplot(x=data['SLUMP(cm)'].to_numpy(), y=data['FLOW(cm)'].to_numpy(), ax=ax)
    st.pyplot(fig)


# Part IV: Clustering or Prediction
def clustering(data):
    data_selected = data["Compressive Strength (28-day)(Mpa)"].to_numpy().reshape(-1, 1)
    algorithm = st.selectbox("Choose clustering algorithm", ["K-means", "DBSCAN"])
    labels = None

    if algorithm == "K-means":
        st.markdown("#### K-means Clustering")
        max_clusters = st.slider("Maximum number of clusters", 2, 20, 10)
        sse = []
        for k in range(2, max_clusters+1):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(data_selected)
            sse.append(kmeans.inertia_)
        fig, ax = plt.subplots()
        ax.plot(range(2, max_clusters+1), sse, marker='o')
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("SSE")
        ax.set_title("Elbow Method For Optimal k")
        st.pyplot(fig)

        n_clusters = st.slider("Select number of clusters", 2, max_clusters, 3)
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(data_selected)
        st.write(f"Inertia: {kmeans.inertia_}")

    elif algorithm == "DBSCAN":
        st.markdown("#### DBSCAN Clustering")
        eps = st.slider("Epsilon", 0.1, 10.0, 0.5)
        min_samples = st.slider("Minimum samples", 1, 10, 5)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data_selected)
        labels = np.where(labels == -1, 0, labels)
        core_sample_indices = len(dbscan.core_sample_indices_)
        st.write(f"Core sample indices: {core_sample_indices}")
        if len(set(labels)) == 1:
            st.error("DBSCAN could not form more than one cluster, please adjust the parameters.")
            return data, labels, None

    if labels is not None and len(set(labels)) > 1:
        silhouette_avg = silhouette_score(data_selected, labels)
        st.write(f"Silhouette Score: {silhouette_avg}")
        data['Cluster'] = labels
        return data, labels, silhouette_avg
    else:
        return None, None, None


def visualize_clusters(data):
    if len(data.columns) >= 2:
        fig = px.scatter(data, x=data.columns[0], y=data.columns[1], color='Cluster')
        st.plotly_chart(fig)
    # if len(data.columns) >= 3:
    #     fig = px.scatter_3d(data, x=data.columns[0], y=data.columns[1], z=data.columns[2], color='Cluster')
    #     st.plotly_chart(fig)


def cluster_statistics(data, labels):
    st.write("Number of data points in each cluster: ", np.bincount(labels))
    if 'Cluster' in data.columns:
        centers = data.groupby('Cluster').mean()
        st.write("Centers of each cluster: ")
        st.dataframe(centers)

# prediction
def prediction(data):
    st.header("Prediction using Linear Regression")
    input_vars = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr.', 'Fine Aggr.']
    output_vars = ['SLUMP(cm)', 'FLOW(cm)', 'Compressive Strength (28-day)(Mpa)']
    selected_output = st.selectbox("Select the output variable to predict", output_vars)
    X = data[input_vars]
    y = data[selected_output]
    model = LinearRegression()
    model.fit(X, y)
    st.subheader("Model Evaluation")
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R² Score: {r2}")
    st.subheader("Make a Prediction")
    input_data = {}
    for var in input_vars:
        input_data[var] = st.number_input(f"Enter value for {var}", value=float(data[var].mean()))
    input_df = pd.DataFrame([input_data])
    predicted_value = model.predict(input_df)[0]
    st.write(f"Predicted {selected_output}: {predicted_value}")
    
    # Visualization of prediction
    st.subheader("Prediction Visualization")
    fig, ax = plt.subplots()
    ax.scatter(y, y_pred, edgecolors=(0, 0, 0))
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    st.pyplot(fig)

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
            input_vars = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr.', 'Fine Aggr.']
            output_vars = ['SLUMP(cm)', 'FLOW(cm)', 'Compressive Strength (28-day)(Mpa)']
            st.header("Visualization of the Cleaned Data")
            plot_histogram(data)
            plot_boxplot(data)
            additional_visualizations(data)
            slump_flow_comp(data)
                    
        if st.checkbox("Perform Clustering or Prediction"):
            # drop column with other than float type
            data = data.select_dtypes(include=[np.number])
            st.header("Clustering or Prediction")
            data, labels, silhouette_avg = clustering(data)
        
        if st.checkbox("Evaluate Learning"):
            st.header("Learning Evaluation")
            visualize_clusters(data)
            cluster_statistics(data, labels)
        
        if st.checkbox("Make Predictions"):
            prediction(data)


if __name__ == "__main__":
    main()