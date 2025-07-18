# **💰 Employee Salary Prediction App**

This repository contains a Streamlit-based web application for predicting employee salaries using a Machine Learning model. The application is built using Python and leverages the adult.csv dataset for training and prediction.

## **🌟 Features**

* **Data Loading & Preprocessing:** Handles adult.csv data, including cleaning, missing value imputation, outlier capping, and categorical feature encoding.  
* **Model Training:** Trains a **RandomForestRegressor** model using a scikit-learn pipeline for robust prediction.  
* **Model Evaluation:** Provides key evaluation metrics (Mean Absolute Error and R-squared) to assess model performance.  
* **Single Prediction Interface:** Allows users to input individual employee details via an intuitive UI to get an instant salary prediction.  
* **Batch Prediction Capability:** Supports uploading a CSV file with multiple employee records for bulk salary predictions, with an option to download the results.  
* **Interactive Web UI:** Built with Streamlit for a user-friendly and responsive experience.

## **🚀 Technologies Used**

* **Python 3.x**  
* **Streamlit:** For creating the interactive web application.  
* **Pandas:** For data manipulation and analysis.  
* **NumPy:** For numerical operations.  
* **Scikit-learn:** For machine learning model (RandomForestRegressor), preprocessing (OneHotEncoder, ColumnTransformer), and pipeline management.  
* **Joblib:** For potential model persistence (though Streamlit's caching handles it in this app).

## **🛠️ Setup and Installation**

To run this application locally, follow these steps:

1. **Clone the repository:**  
   git clone https://github.com/\[Your\_Github\_Username\]/\[Your\_Repo\_Name\].git  
   cd \[Your\_Repo\_Name\]

2. **Create a virtual environment (recommended):**  
   python \-m venv venv  
   \# On Windows  
   .\\venv\\Scripts\\activate  
   \# On macOS/Linux  
   source venv/bin/activate

3. **Install the required libraries:**  
   pip install streamlit pandas numpy scikit-learn joblib

4. **Obtain the dataset:**  
   * Download the adult.csv dataset. You can typically find this dataset on UCI Machine Learning Repository or Kaggle.  
   * Place the adult.csv file in the same directory as your app.py script.

## **🏃 How to Run the Application**

Once you have completed the setup:

1. **Navigate to the project directory:**  
   cd \[Your\_Repo\_Name\]

2. **Run the Streamlit application:**  
   streamlit run app.py

   This command will open the application in your default web browser.

## **📊 Usage**

### **1\. Upload Data**

* On the application's homepage, use the "Upload adult.csv" file uploader to load your dataset.  
* The app will display a preview and statistics of the loaded and preprocessed data.

### **2\. Train Model**

* Click the "Train Prediction Model" button.  
* The application will train the RandomForestRegressor model and display its evaluation metrics (MAE and R2 Score).

### **3\. Make a Single Prediction**

* After the model is trained, navigate to the "Make a Single Prediction" section.  
* Fill in the details for an employee using the provided input widgets (sliders, select boxes, radio buttons).  
* Click "Predict Salary" to see the estimated salary.

### **4\. Batch Prediction**

* In the "Batch Prediction" section, you can upload a CSV file containing multiple employee records.  
* **Example CSV Format:**  
  age,workclass,fnlwgt,educational-num,marital-status,occupation,relationship,race,gender,capital-gain,capital-loss,hours-per-week,native-country  
  35,Private,200000,13,Married-civ-spouse,Exec-managerial,Husband,White,Male,5000,0,40,United-States  
  28,Local-gov,150000,10,Never-married,Other-service,Not-in-family,Black,Female,0,0,30,United-States  
  50,Self-emp-inc,180000,14,Married-civ-spouse,Prof-specialty,Wife,Asian-Pac-Islander,Female,10000,0,50,India

* The app will process the file, add a Predicted\_Salary column, and allow you to download the results as a new CSV.

## **🧠 Model Details**

The core of the prediction system is a **RandomForestRegressor** model. It's integrated into a scikit-learn Pipeline that handles:

* **Categorical Feature Encoding:** Uses OneHotEncoder for features like workclass, marital-status, occupation, etc.  
* **Numerical Feature Passthrough:** Numerical features like age, fnlwgt, capital-gain, capital-loss, and hours-per-week are passed directly.  
* **Outlier Handling:** Numerical features are capped using the Interquartile Range (IQR) method to mitigate the impact of extreme values.  
* **Target Variable:** The income column (\<=50K, \>50K) is converted into numerical values ($25,000 and $75,000 respectively) for regression.

## **📸 Screenshots**

*(Replace these with actual screenshots of your running application)*

| Screenshot 1: Home Page & Upload | Screenshot 2: Data Preview & Statistics |
| :---- | :---- |
|  |  |

| Screenshot 3: Model Training & Evaluation | Screenshot 4: Single Prediction |
| :---- | :---- |
|  |  |

| Screenshot 5: Batch Prediction |
| :---- |
|  |

## **🤝 Contributing**

Contributions are welcome\! If you have suggestions for improvements or new features, please feel free to:

1. Fork the repository.  
2. Create a new branch (git checkout \-b feature/YourFeature).  
3. Make your changes.  
4. Commit your changes (git commit \-m 'Add some feature').  
5. Push to the branch (git push origin feature/YourFeature).  
6. Open a Pull Request.

## **📄 License**

This project is open-sourced under the MIT License. See the LICENSE file for more details.

## **📧 Contact**

For any questions or inquiries, please contact:

* **\[Your Name\]** \- \[Your Email Address\]  
* **\[Your College Name\]** \- \[Your Department\]