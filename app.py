import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap, MarkerCluster
import plotly.express as px
from pyforest import * 
import io
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn import metrics
import plotly.graph_objects as go
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

st.title("Streamlit Application for Classification Analysis")
st.write("by Carrie Lyle Calaca")

uploaded_file = st.file_uploader("", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Integrity Checks:")
    with st.expander("Missing Values"):
        null_values = data.isnull().sum()
        columns_with_nulls = null_values[null_values > 0]
        st.write("Columns with null values:")
        st.dataframe(columns_with_nulls)
    with st.expander("Data Shape"):
        st.write(data.shape)
    with st.expander("View Data"):
        st.dataframe(data.head())

    st.write("Explanatory Data Analysis (EDA):")
    with st.expander("Histograms"):
        for column in data.columns:
            st.subheader(f'Histogram of {column}')
            fig = px.histogram(data, x=column, title=f'Histogram of {column}')
            st.plotly_chart(fig)
    with st.expander("Pair Plots"):
        fig = px.scatter_matrix(data, dimensions=data.columns, title='Pair Plots')
        st.plotly_chart(fig)
    with st.expander("Correlation Heatmap"):
        correlation_matrix = data.corr()
        fig = px.imshow(correlation_matrix, title="Correlation Matrix", color_continuous_scale='viridis', aspect="auto", width=1000, height=700)
        st.plotly_chart(fig)
    
    st.write("Model Selection and Training:")
    model_name = st.selectbox("Select the Model", ("Logistic Regression", "KNN"))

    def create_params(model_name):
        param = {}
        if model_name == 'KNN':
            param['n_neighbors'] = st.slider('K', 1, 20)
            param['weights'] = st.selectbox('Weights', ['uniform', 'distance'])
            param['metric'] = st.selectbox('Metric', ['minkowski', 'euclidean', 'manhattan'])
        else:
            param['solver'] = st.selectbox('Solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
        return param

    param = create_params(model_name)

    target_var = st.selectbox("Select the target variable",data.columns)

    X = data.drop(target_var, axis=1)
    y = data[target_var]

    random_seed = st.slider("Select the random state : ",1,20)
    param['random_seed'] = random_seed
    
    X_train, X_test, y_train, y_test = None, None, None, None

    if st.checkbox("Split Data"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_seed)
        st.write("Data has been split into training and test sets.")
        scale = st.selectbox("Scaling Algorithm", ("Standard Scaler", "MinMax Scaler"))
        model = None
        if st.button("Train Model"):
            if model_name == 'Logistic Regression':
                model = LogisticRegression(solver=param['solver'])
            else:
                model = KNeighborsClassifier(
                    n_neighbors=param['n_neighbors'],
                    weights=param['weights'],
                    metric=param['metric']
                )

            if model is not None:
                model.fit(X_train, y_train)
                st.write(f"{model_name} model trained successfully!")
                with open('model.pkl', 'wb') as file:
                    pickle.dump(model, file)

        if st.button("Predict"):
            try:
                with open('model.pkl', 'rb') as file:
                    model = pickle.load(file)
                if model is not None and X_test is not None and y_test is not None:
                    predictions = model.predict(X_test)
                    pred_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                    
                    st.write("Predictions:")
                    predictions_df = pd.DataFrame(predictions, columns=['Predictions'])
                    st.dataframe(predictions_df)
                    accuracy = metrics.accuracy_score(y_test, predictions)
                    st.write(f"Accuracy: {accuracy:.4f}")

                with st.expander("Show Confusion Matrix"):
                    cm = metrics.confusion_matrix(y_test, predictions)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, cmap='viridis', annot=True, fmt='d', vmax=1)
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    st.pyplot(plt)
                    plt.close()
                    
                with st.expander("Show ROC Curve"):
                    if pred_prob is not None:
                        y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
                        n_classes = y_test_binarized.shape[1]

                        fig = go.Figure()

                        fpr = dict()
                        tpr = dict()
                        roc_auc = dict()
                        for i in range(n_classes):
                            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], pred_prob[:, i])
                            roc_auc[i] = auc(fpr[i], tpr[i])
                            fig.add_trace(go.Scatter(x=fpr[i], y=tpr[i], mode='lines', name=f'ROC curve of class {i} (AUC = {roc_auc[i]:.4f})'))

                        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), pred_prob.ravel())
                        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                        fig.add_trace(go.Scatter(x=fpr["micro"], y=tpr["micro"], mode='lines', line=dict(dash='dash'), name=f'Micro-average ROC curve (AUC = {roc_auc["micro"]:.4f})'))

                        all_fpr = np.linspace(0, 1, 100)
                        mean_tpr = np.zeros_like(all_fpr)
                        for i in range(n_classes):
                            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                        mean_tpr /= n_classes
                        fpr["macro"] = all_fpr
                        tpr["macro"] = mean_tpr
                        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
                        fig.add_trace(go.Scatter(x=fpr["macro"], y=tpr["macro"], mode='lines', line=dict(dash='dot'), name=f'Macro-average ROC curve (AUC = {roc_auc["macro"]:.4f})'))

                        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='black', dash='dash'), name='Random'))

                        fig.update_layout(
                            title='ROC Curve',
                            xaxis_title='False Positive Rate',
                            yaxis_title='True Positive Rate',
                            legend_title='Classes',
                            showlegend=True
                        )

                        st.plotly_chart(fig)
                    else:
                        st.write("No prediction probabilities available for ROC Curve.")
                @st.cache_data
                def convert_df(df):
                    return df.to_csv().encode("utf-8")

                csv = convert_df(predictions_df)
                st.download_button(
                    label="Download predictions as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv",
                )
            except FileNotFoundError:
                st.write("No model found. Please train or load a model first.")
        else:
            st.write("Please train the model first.")
    else:
        st.write("Please split data.")
else:
    st.write("Please upload a CSV file to proceed.")
