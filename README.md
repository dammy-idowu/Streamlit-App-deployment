# 📦 Disease Diagnostic App (DocPal)
```

docpal.streamlit.app

```

## 🤖 Disease Prediction Project
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

`SHAP`: SHAP (SHapley Additive exPlanations) values explain the output of any machine learning model.


## 📂 Project Structure

├── app.py -------------------                      # the main Streamlit application script.

├── best_model.pkl -------------------              # the trained machine learning model.

├── label_encoder.pkl -------------------          # the LabelEncoder used for target variable.

├── X_train.pkl -------------------                 # training features used for bootstrapping.

├── y_train.pkl -------------------                 # training labels used for bootstrapping.

└── README.md -------------------                  # this documentation file.




## 🚀 Installation and Setup

To set up and run this project, libraries used in the project were stored in requirements.txt file.

The following python libraries were used:

- streamlit

- numpy

- pandas

- scikit-learn

- joblib

- scipy

- Image

- SHAP

- request



## 📖 Usage

To run the Streamlit web application, click `docpal.streamlit.app`

The app will open in your default web browser.



## How to use the app:

`Sidebar Input`: Use the sidebar to input basic patient information, such as age, gender, and weight.

`Main Form`: In the main content area, use the dropdown menus to select "True" or "False" for each symptom.

`Get Prediction`: Click the **"Get Prediction"** button to see the model's output.



## 🤝 Contribution

Contributions to this project are welcome. Please submit a pull request with a detailed description of your intended changes. For collaboration, feel free to contact me at damilolapeter.idowu@gmail.com



## 📄 License

This project is licensed under the MIT License.



[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dammyidowu-docpal.streamlit.app/)




#### ⚠️ Disclaimer: 
*This medical app is for informational purposes only and should not be considered as medical advice; users are advised to consult a qualified healthcare professional before making any medical decisions*.
