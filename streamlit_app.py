import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Load the saved objects ---
try:
    best_model = joblib.load('best_model.pkl')
    le = joblib.load('label_encoder.pkl')
    # If you used a pipeline, load that instead of the raw model
    # best_pipeline = joblib.load('best_pipeline.pkl')
except FileNotFoundError:
    st.error("Model or LabelEncoder file not found. Please ensure 'best_model.pkl' and 'label_encoder.pkl' exist in the directory.")
    st.stop() # Stops the script if files are missing

# --- App Title and Description ---
st.title('Disease Prediction App')
st.write('Enter the patient\'s symptoms to get a disease prediction.')

# --- Get feature names for user input ---
# NOTE: Replace this with your actual independent feature names.
# This is a placeholder for demonstration.
feature_names = ['health_centre', 'gender', 'age', 'weight', 'high_temperature', 'fever_48hrs', 'fever_in_the_last_7days', 'loss_of_weight', 'headache', 'nausea', 'vomiting', 'joint_pain', 'joint_swelling', 'muscle_pain', 'chest_pain', 'back_pain', 'loss_of_consciousness', 'loss_of_appetite', 'skin_rash', 'morbilliform_rash', 'bleeding', 'runny_nose', 'lethargy', 'dizzy', 'stomach_pain', 'swelling_stomach', 'throat_pain', 'cough', 'diarrhoea', 'retro_orbital_pain', 'shiver_cold_sensation', 'frequent_urination', 'Constipation', 'bleeding_nose', 'focal_convulsion', 'multiple_convulsions', 'impaired_level_of_consciousness', 'facial_flushing', 'facial_swelling', 'profuse_sweating', 'irrational_talking', 'bitter_taste_in_your_throat', 'stiffness', 'respiratory_distress', 'shock', 'disease']

# --- User input form ---
st.header('Patient Information')
with st.form(key='prediction_form'):
    # Use different Streamlit widgets for different types of features.
    # The following are examples; adapt them to your specific features.
    user_inputs = {}
    for feature in feature_names:
        user_inputs[feature] = st.text_input(f'Enter value for {feature}', value='0.0')

    submit_button = st.form_submit_button(label='Get Prediction')

# --- Prediction and Output ---
if submit_button:
    try:
        # Convert user input to a DataFrame
        input_data = pd.DataFrame([user_inputs])

        # Convert the input data to numeric types (if they are not already)
        input_data = input_data.apply(pd.to_numeric, errors='coerce')
        
        # Check for NaN values after conversion
        if input_data.isnull().values.any():
            st.warning("Invalid input detected. Please ensure all values are numeric.")
        else:
            st.subheader('Prediction Result')

            # Get the predicted probabilities
            probabilities = best_model.predict_proba(input_data)
            
            # Use np.where to handle the normalization safely
            row_sums = probabilities.sum(axis=1, keepdims=True)
            safe_probabilities = np.where(row_sums == 0, 0, probabilities / row_sums)
            safe_probabilities = np.nan_to_num(safe_probabilities)

            # Get the predicted class index
            predicted_class_index = np.argmax(safe_probabilities, axis=1)
            
            # Inverse transform to get the disease name
            predicted_disease = le.inverse_transform(predicted_class_index)[0]

            # Display the result
            st.success(f'Predicted Disease: **{predicted_disease}**')
            
            # Display probabilities
            st.write('**Predicted Probabilities:**')
            prob_df = pd.DataFrame(safe_probabilities, columns=le.classes_)
            st.bar_chart(prob_df.T)

    except ValueError as ve:
        st.error(f"Prediction Error: {ve}. Please check your input values.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

