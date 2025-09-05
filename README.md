# 📦 Disease Diagnostic App (DocPal)
```

```

## 🤖 Disease Prediction App
This project provides an interactive web application for predicting a patient's disease(s) based on their symptoms and other demographic information. The application is built with Streamlit and uses a pre-trained machine learning model to generate predictions with confidence intervals.


## 🌟 Features
`Interactive Interface`: A user-friendly web interface built with Streamlit for easy user interaction.

`Comprehensive Input`: Collects various patient details, including basic information (age, gender, weight, health center) and a list of specific symptoms.

`Predictive Modeling`: Utilizes a machine learning model to predict a patient's disease based on the provided inputs.

`Multi-Label Prediction`: Supports multi-label classification, allowing for the prediction of multiple diseases simultaneously (e.g., Malaria and Dengue).

`Confidence Intervals`: Provides confidence intervals for each prediction, offering a measure of uncertainty and robustness.

`Clear Results`: Displays the prediction outcomes in an easy-to-understand format, including predicted probabilities for each disease.

`Robust Error Handling`: The app includes robust error handling to manage invalid user input and ensure a smooth user experience.


## ⚙️ Technologies

`Python`: The core programming language.

`Streamlit`: For building the interactive web application.

`Scikit-learn`: For the machine learning model and related functionalities.

`Pandas & NumPy`: For data manipulation and numerical operations.

`Joblib`: For efficient serialization and deserialization of the machine learning model and preprocessors.

`SciPy`: For statistical calculations used in bootstrapping.


## 📂 Project Structure

├── app.py                      # The main Streamlit application script.

├── best_model.pkl              # The trained machine learning model.

├── label_encoder.pkl           # The LabelEncoder used for target variable.

├── X_train.pkl                 # Training features used for bootstrapping.

├── y_train.pkl                 # Training labels used for bootstrapping.

└── README.md                   # This documentation file.



🚀 Installation

To set up and run this project, libraries used in the project were stored in requirements.txt file.

The following python libraries were used:

- streamlit

- numpy

- pandas

- scikit-learn

- joblib

- scipy

- Image


## 📖 Usage

To run the Streamlit web application, click `docpal.streamlit.app`

The app will open in your default web browser.


## How to use the app:

`Sidebar Input`: Use the sidebar to input basic patient information, such as age, gender, and weight.

`Main Form`: In the main content area, use the dropdown menus to select "True" or "False" for each symptom.

`Get Prediction`: Click the **"Get Prediction"** button to see the model's output.


## 🤝 Contribution

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.


📄 License

This project is licensed under the MIT License. See the LICENSE file for details.


