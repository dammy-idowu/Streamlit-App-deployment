import streamlit as st
import pandas as pd
import joblib
import numpy as np
from scipy import stats

# --- Load the saved objects ---
try:
    best_model = joblib.load('best_model.pkl')
    le = joblib.load('label_encoder.pkl')
except FileNotFoundError:
    st.error("Model or LabelEncoder file not found. Please ensure 'best_model.pkl' and 'label_encoder.pkl' exist.")
    st.stop()

# --- Define custom name mappings ---
user_friendly_feature_names = {
    'health_centre': 'Health Centre',
    'age': 'Age (years)',
    'weight': 'Weight (kg)',
    'gender_encoded': 'Gender',
    'high_temperature': 'High body temperature',
    'fever_48hrs': 'Had fever for more than 48 hours',
    'fever_in_the_last_7days': 'Had fever in the last 7 days',
    'loss_of_weight': 'Loss of weight',
    'headache': 'Headache',
    'nausea': 'Nausea',
    'vomiting': 'Vomiting',
    'joint_pain': 'Joint pain',
    'joint_swelling': 'Joint swelling',
    'muscle_pain': 'Muscle pain',
    'chest_pain': 'Chest pain',
    'back_pain': 'Back pain',
    'loss_of_consciousness': 'Loss of consciousness',
    'loss_of_appetite': 'Loss of appetite',
    'skin_rash': 'Skin rash',
    'morbilliform_rash': 'Morbilliform rash',
    'bleeding': 'Bleeding',
    'runny_nose': 'Runny nose',
    'lethargy': 'Lethargy',
    'dizzy': 'Dizzy',
    'stomach_pain': 'Stomach pain',
    'swelling_stomach': 'Swelling stomach',
    'throat_pain': 'Throat pain',
    'cough': 'Cough',
    'diarrhoea': 'Diarrhoea',
    'retro_orbital_pain': 'Retro-orbital pain',
    'shiver_cold_sensation': 'Cold shivering sensation',
    'frequent_urination': 'Frequent urination',
    'Constipation': 'Constipation',
    'bleeding_nose': 'Bleeding nose',
    'focal_convulsion': 'Focal convulsion',
    'multiple_convulsions': 'Multiple convulsion',
    'impaired_level_of_consciousness': 'Impaired level of consciousness',
    'facial_flushing': 'Facial flushing',
    'facial_swelling': 'Facial swelling',
    'profuse_sweating': 'Profuse sweating',
    'irrational_talking': 'Irrational talking',
    'bitter_taste_in_your_throat': 'Bitter taste in your throat',
    'stiffness': 'Stiffness',
    'respiratory_distress': 'Respiratory distress',
    'shock': 'Shock'
}

# --- NEW DISEASE MAPPING ---
user_friendly_disease_names = {
    '4': ['Malaria', 'Dengue'],
    '8': ['Malaria', 'Thyphoid Fever'],
    '3': ['Malaria'],
    '7': ['Malaria', 'Other diseases'],
    '10': ['Malaria', 'Yellow fever'],
    '2': ['Dengue', 'Yellow fever'],
    '0': ['Dengue', 'Other diseases'],
    '5': ['Malaria', 'Dengue', 'Other diseases'],
    '11': ['Malaria', 'Yellow fever', 'Other diseases'],
    '9': ['Malaria', 'Thyphoid fever', 'Other diseases'],
    '1': ['Dengue', 'Thyphoid fever'],
    '12': ['Other diseases'],
    '6': ['Malaria', 'Dengue', 'Typhoid Fever']
}


# --- Define specific widget options and mappings ---
gender_options = {'Female': 0, 'Male': 1}
health_centre_options = {'CMA de DO': 0, 'CMA de DAFRA': 1}
boolean_options = {'False': 0, 'True': 1}

# --- Bootstrap function (same as previous code) ---
def calculate_bootstrap_confidence_interval(model, input_data, n_bootstraps=1000, confidence=0.95):
    input_df = pd.DataFrame([input_data])
    predicted_probabilities = []
    
    if len(input_df.columns) == 0:
        return 0, 0

    for _ in range(n_bootstraps):
        resampled_input = input_df.sample(n=len(input_df), replace=True)
        proba = model.predict_proba(resampled_input)
        
        row_sums = proba.sum(axis=1, keepdims=True)
        safe_proba = np.where(row_sums == 0, 0, proba / row_sums)
        predicted_probabilities.append(np.max(safe_proba, axis=1))

    lower_bound = np.percentile(predicted_probabilities, (1 - confidence) / 2 * 100)
    upper_bound = np.percentile(predicted_probabilities, (1 - (1 - confidence) / 2) * 100)
    
    return lower_bound, upper_bound

# --- Streamlit App Layout ---
st.set_page_config(page_title="Custom Disease Prediction App")
st.title('Disease Prediction App')
st.info('Your disease diagnostic assistant')
st.write('Enter the patient\'s symptoms to get a disease prediction.')

# --- Create the sidebar for basic patient info ---
st.sidebar.header('Basic Patient Information')
with st.sidebar.form(key='sidebar_form'):
    user_inputs_sidebar = {}
    selected_hc_name = st.selectbox('Health Centre', options=list(health_centre_options.keys()))
    user_inputs_sidebar['health_centre'] = health_centre_options[selected_hc_name]
    user_inputs_sidebar['age'] = st.slider('Age (years)', min_value=0, max_value=120, value=30)
    user_inputs_sidebar['weight'] = st.slider('Weight (kg)', min_value=10.0, max_value=200.0, value=70.0, step=0.5)
    selected_gender_name = st.selectbox('Gender', options=list(gender_options.keys()))
    user_inputs_sidebar['gender_encoded'] = gender_options[selected_gender_name]
    submit_button_sidebar = st.form_submit_button(label='Update Patient Info')

# --- User input form for symptoms in the main area ---
st.header('Patient Symptoms')
with st.form(key='prediction_form'):
    user_inputs_main = user_inputs_sidebar.copy()
    
    boolean_features = [
        'high_temperature', 'fever_48hrs', 'fever_in_the_last_7days',
        'loss_of_weight', 'headache', 'nausea', 'vomiting', 'joint_pain',
        'joint_swelling', 'muscle_pain', 'chest_pain', 'back_pain',
        'loss_of_consciousness', 'loss_of_appetite', 'skin_rash',
        'morbilliform_rash', 'bleeding', 'runny_nose', 'lethargy',
        'dizzy', 'stomach_pain', 'swelling_stomach', 'throat_pain',
        'cough', 'diarrhoea', 'retro_orbital_pain', 'shiver_cold_sensation',
        'frequent_urination', 'Constipation', 'bleeding_nose',
        'focal_convulsion', 'multiple_convulsions',
        'impaired_level_of_consciousness', 'facial_flushing',
        'facial_swelling', 'profuse_sweating', 'irrational_talking',
        'bitter_taste_in_your_throat', 'stiffness',
        'respiratory_distress', 'shock'
    ]
    
    for feature in boolean_features:
        display_name = user_friendly_feature_names[feature]
        selected_bool = st.selectbox(display_name, options=list(boolean_options.keys()))
        user_inputs_main[feature] = boolean_options[selected_bool]
    
    submit_button_main = st.form_submit_button(label='Get Prediction')

# --- Prediction and Output ---
if submit_button_main:
    try:
        input_data = pd.DataFrame([user_inputs_main])
        input_data = input_data.apply(pd.to_numeric, errors='coerce')
        
        if input_data.isnull().values.any():
            st.warning("Invalid input detected. Please ensure all values are valid.")
        else:
            st.subheader('Prediction Result')

            probabilities = best_model.predict_proba(input_data)
            row_sums = probabilities.sum(axis=1, keepdims=True)
            safe_probabilities = np.where(row_sums == 0, 0, probabilities / row_sums)
            safe_probabilities = np.nan_to_num(safe_probabilities)

            predicted_class_index = np.argmax(safe_probabilities, axis=1)
            original_predicted_disease_label = str(predicted_class_index[0])
            
            display_predicted_diseases = user_friendly_disease_names.get(
                original_predicted_disease_label, 
                [f'Disease with label {original_predicted_disease_label}']
            )
            
            st.success(f'Predicted Disease: **{", ".join(display_predicted_diseases)}**')
            
            lower, upper = calculate_bootstrap_confidence_interval(best_model, input_data)
            st.info(f'**Confidence Interval:** ({lower:.2f}, {upper:.2f})')
            
            st.write('**Predicted Probabilities:**')
            prob_df = pd.DataFrame(safe_probabilities, columns=le.classes_).T
            prob_df.index = prob_df.index.map(lambda c: ", ".join(user_friendly_disease_names.get(c, [c])))
            st.bar_chart(prob_df)

    except ValueError as ve:
        st.error(f"Prediction Error: {ve}. Please check your input values.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
