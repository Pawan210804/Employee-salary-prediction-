import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib # To save/load the model and preprocessor

# Set page configuration for better layout
st.set_page_config(layout="wide", page_title="Employee Salary Predictor")

# --- Helper Functions (Cached for performance) ---

@st.cache_data
def load_and_preprocess_data(uploaded_file):
    """
    Loads the adult.csv data, performs cleaning, and converts income to numeric.
    Returns the processed DataFrame.
    """
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

    # Data Cleaning and Preprocessing steps
    # Note: Using .copy() to avoid SettingWithCopyWarning later if modifications are made
    data = data.copy()
    data.replace('?', 'Others', inplace=True)
    data['native-country'].replace('?', 'United-States', inplace=True)

    # Remove categories with very few counts
    data = data[data['workclass'] != 'Without-pay']
    data = data[data['workclass'] != 'Never-worked']
    data = data[data['education'] != '1st-4th']
    data = data[data['education'] != '5th-6th']
    data = data[data['education'] != 'Preschool']

    # Outlier handling for numerical columns (age, capital-gain, educational-num)
    def cap_outliers_iqr(df_col):
        Q1 = df_col.quantile(0.25)
        Q3 = df_col.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Use .loc for setting values to avoid SettingWithCopyWarning
        df_col_capped = np.where(df_col < lower_bound, lower_bound, df_col)
        df_col_capped = np.where(df_col_capped > upper_bound, upper_bound, df_col_capped)
        return pd.Series(df_col_capped, index=df_col.index) # Return as Series to maintain index

    numerical_cols_for_outliers = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
    for col in numerical_cols_for_outliers:
        if col in data.columns:
            data[col] = cap_outliers_iqr(data[col])

    # Drop 'education' column as 'educational-num' is kept
    if 'education' in data.columns:
        data.drop(columns=['education'], inplace=True)

    # Convert 'income' to numerical for regression
    data['income_numeric'] = data['income'].apply(lambda x: 25000 if x.strip() == '<=50K' else 75000)

    return data

@st.cache_resource
# The _preprocessor argument is now correctly named with an underscore to avoid hashing.
# We also pass X_train (the raw dataframe) to the model.fit, as the preprocessor is inside the pipeline.
def train_model(X_train_raw, y_train, _preprocessor_transformer):
    """
    Trains the RandomForestRegressor model.
    Returns the trained model.
    """
    # Create the pipeline using the preprocessor transformer
    model = Pipeline(steps=[('preprocessor', _preprocessor_transformer),
                            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
    # Fit the pipeline on the raw training data
    model.fit(X_train_raw, y_train)
    return model

# --- Streamlit UI ---

st.title("💰 Employee Salary Prediction App")
st.markdown("""
    This application predicts employee salaries based on various demographic and work-related features
    using a Machine Learning model.
    **Upload your `adult.csv` file to get started!**
""")

# --- File Uploader ---
st.header("1. Upload Data")
uploaded_file = st.file_uploader("Upload adult.csv", type="csv")

if uploaded_file is not None:
    with st.spinner("Loading and preprocessing data..."):
        df = load_and_preprocess_data(uploaded_file)

    if df is not None:
        st.success("Data loaded and preprocessed successfully!")
        st.subheader("Data Preview")
        st.dataframe(df.head())

        st.subheader("Data Statistics")
        st.write(df.describe())

        # Define features (X) and target (y)
        X = df.drop(['income', 'income_numeric'], axis=1)
        y = df['income_numeric']

        # Identify categorical and numerical features for the preprocessor
        categorical_features = X.select_dtypes(include='object').columns.tolist()
        numerical_features = X.select_dtypes(include=np.number).columns.tolist()

        # Create a preprocessor (this object itself is unhashable, so it needs the underscore in the cached function)
        preprocessor_transformer = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                ('num', 'passthrough', numerical_features)
            ])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.info(f"Training set size: {len(X_train)} samples, Testing set size: {len(X_test)} samples")

        # --- Model Training Section ---
        st.header("2. Train Model")
        if st.button("Train Prediction Model"):
            with st.spinner("Training the RandomForestRegressor model... This might take a moment."):
                # Pass the raw X_train and the preprocessor_transformer (with underscore for caching)
                model = train_model(X_train, y_train, preprocessor_transformer)
            st.session_state['model'] = model
            st.session_state['X_columns'] = X.columns.tolist() # Store column order for prediction
            st.session_state['categorical_features'] = categorical_features
            st.session_state['numerical_features'] = numerical_features
            st.success("Model trained successfully!")

            # Evaluate the model
            st.subheader("Model Evaluation")
            # The model pipeline handles preprocessing for X_test as well
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"**Mean Absolute Error (MAE):** ${mae:,.2f}")
            st.write(f"**R-squared (R2) Score:** {r2:.4f}")
            st.info("MAE represents the average absolute difference between predicted and actual salaries. R2 score closer to 1 indicates a better fit.")
        else:
            if 'model' not in st.session_state:
                st.warning("Click 'Train Prediction Model' to train the model before making predictions.")

        # --- Prediction Sections ---
        if 'model' in st.session_state:
            model = st.session_state['model']
            X_columns = st.session_state['X_columns']
            # We don't need categorical_features and numerical_features explicitly here
            # because the model pipeline handles the preprocessing internally.

            # --- Single Prediction ---
            st.header("3. Make a Single Prediction")
            st.markdown("Enter the details of an employee to predict their salary.")

            # Create input fields for all features
            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.slider("Age", int(df['age'].min()), int(df['age'].max()), 30) # Use min/max from data
                workclass = st.selectbox("Workclass", df['workclass'].unique())
                fnlwgt = st.number_input("Fnlwgt (Final Weight)", min_value=int(df['fnlwgt'].min()), max_value=int(df['fnlwgt'].max()), value=200000)
                educational_num = st.slider("Educational Number", int(df['educational-num'].min()), int(df['educational-num'].max()), 9) # Corresponds to education level

            with col2:
                marital_status = st.selectbox("Marital Status", df['marital-status'].unique())
                occupation = st.selectbox("Occupation", df['occupation'].unique())
                relationship = st.selectbox("Relationship", df['relationship'].unique())
                race = st.selectbox("Race", df['race'].unique())

            with col3:
                gender = st.radio("Gender", df['gender'].unique())
                capital_gain = st.number_input("Capital Gain", min_value=int(df['capital-gain'].min()), max_value=int(df['capital-gain'].max()), value=0)
                capital_loss = st.number_input("Capital Loss", min_value=int(df['capital-loss'].min()), max_value=int(df['capital-loss'].max()), value=0)
                hours_per_week = st.slider("Hours per Week", int(df['hours-per-week'].min()), int(df['hours-per-week'].max()), 40)
                native_country = st.selectbox("Native Country", df['native-country'].unique())

            if st.button("Predict Salary"):
                new_data = pd.DataFrame([{
                    'age': age,
                    'workclass': workclass,
                    'fnlwgt': fnlwgt,
                    'educational-num': educational_num,
                    'marital-status': marital_status,
                    'occupation': occupation,
                    'relationship': relationship,
                    'race': race,
                    'gender': gender,
                    'capital-gain': capital_gain,
                    'capital-loss': capital_loss,
                    'hours-per-week': hours_per_week,
                    'native-country': native_country
                }])

                # Ensure the new_data DataFrame has columns in the same order as X_train
                # This is crucial for the preprocessor to work correctly
                new_data = new_data[X_columns]

                try:
                    predicted_salary = model.predict(new_data)[0]
                    st.success(f"**Predicted Salary:** ${predicted_salary:,.2f}")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    st.warning("Please ensure all input fields are correctly filled and the model is trained.")

            # --- Batch Prediction ---
            st.header("4. Batch Prediction")
            st.markdown("Upload a CSV file containing multiple employee records for batch salary prediction.")
            st.markdown("---")
            st.markdown("#### **Example CSV Format for Batch Prediction:**")
            st.code("""
age,workclass,fnlwgt,educational-num,marital-status,occupation,relationship,race,gender,capital-gain,capital-loss,hours-per-week,native-country
35,Private,200000,13,Married-civ-spouse,Exec-managerial,Husband,White,Male,5000,0,40,United-States
28,Local-gov,150000,10,Never-married,Other-service,Not-in-family,Black,Female,0,0,30,United-States
50,Self-emp-inc,180000,14,Married-civ-spouse,Prof-specialty,Wife,Asian-Pac-Islander,Female,10000,0,50,India
            """)
            st.markdown("---")

            batch_file = st.file_uploader("Upload CSV for Batch Prediction", type="csv", key="batch_uploader")

            if batch_file is not None:
                with st.spinner("Processing batch predictions..."):
                    try:
                        batch_data = pd.read_csv(batch_file)
                        st.write("Uploaded batch data preview:")
                        st.dataframe(batch_data.head())

                        # Ensure batch_data has the same columns and order as the training data
                        # A more robust solution would check and fill missing columns.
                        # For now, let's assume the batch file has all necessary columns in correct order.
                        # If not, this might cause errors in the preprocessor.
                        batch_data_for_pred = batch_data[X_columns] # Ensure column order

                        batch_preds = model.predict(batch_data_for_pred)
                        batch_data['Predicted_Salary'] = batch_preds
                        st.success("Batch predictions complete!")
                        st.write("Predictions:")
                        st.dataframe(batch_data.head())

                        csv_output = batch_data.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Predictions CSV",
                            data=csv_output,
                            file_name='predicted_salaries.csv',
                            mime='text/csv'
                        )
                    except Exception as e:
                        st.error(f"Error during batch prediction: {e}")
                        st.warning("Please ensure your batch CSV file is correctly formatted and contains all required columns.")
    else:
        st.error("Could not process the uploaded file. Please check its content and format.")
else:
    st.info("Please upload the `adult.csv` file to begin.")

