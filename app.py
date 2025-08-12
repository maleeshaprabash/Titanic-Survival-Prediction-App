import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
from sklearn.model_selection import train_test_split

# Set page configuration for consistent styling
st.set_page_config(page_title="Titanic Survival Prediction", layout="wide")

# Title and Description
st.title("Titanic Survival Prediction")
st.markdown("""
This application uses a pre-trained machine learning model to predict whether a passenger survived the Titanic disaster.
Explore the dataset, visualize insights, input passenger details for predictions, and view model performance.
""")

# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Data Exploration", "Visualizations", "Model Prediction", "Model Performance"])

# Load dataset with caching
@st.cache_data
def load_data():
    try:
        titanic = pd.read_csv('C:\\Users\\ASUS\\Desktop\\New Project\\data\\Titanic-Dataset.csv')
        return titanic
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None
titanic = load_data()

# Load pre-trained model with caching
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
model = load_model()

# Preprocess input data for prediction
def preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked):
    """Preprocess user input for prediction."""
    sex_val = 1 if sex == "female" else 0
    embarked_val = {'Southampton': 0, 'Cherbourg': 1, 'Queenstown': 2}[embarked]
    return np.array([[pclass, sex_val, age, sibsp, parch, fare, embarked_val]])

# Data Exploration Section
if section == "Data Exploration":
    st.header("Data Exploration")
    st.subheader("Dataset Overview")
    if titanic is not None:
        df = titanic.copy()
        # Convert dtypes for consistent display
        df = df.convert_dtypes()
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype(str)
        for col in df.select_dtypes(include='Int64').columns:
            df[col] = df[col].astype('float')  # or .astype('int').fillna(0)
        for col in df.select_dtypes(include='Float64').columns:
            df[col] = df[col].astype('float')
        
        st.write(f"Shape: {df.shape}")
        st.write("Columns:", df.columns.tolist())
        st.write("Data Types:")
        st.write(df.dtypes)
        st.markdown("**Note**: Use the filter below to explore specific columns.")
        
        st.subheader("Sample Data")
        df_head = df.head().copy()
        df_head = df_head.convert_dtypes()
        for col in df_head.select_dtypes(include='object').columns:
            df_head[col] = df_head[col].astype(str)
        for col in df_head.select_dtypes(include='Int64').columns:
            df_head[col] = df_head[col].astype('float')
        for col in df_head.select_dtypes(include='Float64').columns:
            df_head[col] = df_head[col].astype('float')
        st.dataframe(df_head)
        
        st.subheader("Interactive Data Filtering")
        columns = st.multiselect("Select columns to display", df.columns, default=['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age'])
        if columns:
            df_filtered = df[columns].copy()
            df_filtered = df_filtered.convert_dtypes()
            for col in df_filtered.select_dtypes(include='object').columns:
                df_filtered[col] = df_filtered[col].astype(str)
            for col in df_filtered.select_dtypes(include='Int64').columns:
                df_filtered[col] = df_filtered[col].astype('float')
            for col in df_filtered.select_dtypes(include='Float64').columns:
                df_filtered[col] = df_filtered[col].astype('float')
            st.dataframe(df_filtered)

# Visualizations Section
elif section == "Visualizations":
    st.header("Visualizations")
    if titanic is not None:
        # Survival by Passenger Class
        st.subheader("Survival by Passenger Class")
        fig1 = px.histogram(titanic, x='Pclass', color='Survived', barmode='group',
                           title="Survival Count by Passenger Class")
        st.plotly_chart(fig1)
        
        # Age Distribution
        st.subheader("Age Distribution by Survival")
        fig2 = px.histogram(titanic, x='Age', color='Survived', nbins=30,
                           title="Age Distribution")
        st.plotly_chart(fig2)
        
        # Fare vs Age Scatter
        st.subheader("Fare vs Age by Survival")
        fig3 = px.scatter(titanic, x='Age', y='Fare', color='Survived',
                         title="Fare vs Age")
        st.plotly_chart(fig3)
        st.markdown("**Note**: Interact with the plots by hovering or selecting data points.")

# Model Prediction Section
elif section == "Model Prediction":
    st.header("Model Prediction")
    st.subheader("Enter Passenger Details")
    st.markdown("**Help**: Input the passenger's details below to predict survival. All fields are required.")
    
    if model is not None:
        col1, col2 = st.columns(2)
        with col1:
            pclass = st.selectbox("Passenger Class (1-3)", [1, 2, 3])
            sex = st.selectbox("Sex", ["male", "female"])
            age = st.slider("Age", 0, 100, 30)
        with col2:
            sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=8, value=0)
            parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=6, value=0)
            fare = st.number_input("Fare", min_value=0.0, max_value=500.0, value=30.0)
            embarked = st.selectbox("Embarked", ["Southampton", "Cherbourg", "Queenstown"])
        
        if st.button("Predict"):
            try:
                input_data = preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked)
                with st.spinner("Calculating prediction..."):
                    prediction = model.predict(input_data)[0]
                    prob = model.predict_proba(input_data)[0][1]
                st.success(f"Prediction: {'Survived' if prediction == 1 else 'Did not survive'}")
                st.write(f"Survival Probability: {prob:.2%}")
            except Exception as e:
                st.error(f"Error in prediction: {e}")
    else:
        st.error("Model not loaded. Please ensure 'model.pkl' exists.")

# Model Performance Section
elif section == "Model Performance":
    st.header("Model Performance")
    if titanic is not None and model is not None:
        # Preprocess data for performance evaluation
        titanic_processed = titanic.copy()
        titanic_processed['Age'] = titanic_processed['Age'].fillna(titanic_processed['Age'].median())
        titanic_processed['Embarked'] = titanic_processed['Embarked'].fillna(titanic_processed['Embarked'].mode()[0])
        titanic_processed['Fare'] = titanic_processed['Fare'].fillna(titanic_processed['Fare'].median())
        titanic_processed['Sex'] = titanic_processed['Sex'].map({'male': 0, 'female': 1})
        titanic_processed['Embarked'] = titanic_processed['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        X = titanic_processed[features]
        y = titanic_processed['Survived']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Calculate metrics
        y_pred = model.predict(X_test)
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred)
        }
        st.subheader("Model Metrics (Random Forest)")
        metrics_df = pd.DataFrame([metrics]).convert_dtypes()
        for col in metrics_df.select_dtypes(include='object').columns:
            metrics_df[col] = metrics_df[col].astype(str)
        for col in metrics_df.select_dtypes(include='Int64').columns:
            metrics_df[col] = metrics_df[col].astype('float')
        st.write(metrics_df)
        st.markdown("**Note**: Metrics are based on a 20% test set split.")
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix - Random Forest')
        st.pyplot(fig)
    else:
        st.error("Dataset or model not loaded.")