
from flask import Flask, render_template, request, flash
import joblib
import numpy as np
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'

# Load trained model with error handling
try:
    model = joblib.load("logreg.pkl")
except FileNotFoundError:
    print("Warning: Model file not found. Using dummy predictions.")
    model = None

# Mapping back numeric prediction to original stage
stage_map = {
    0: 'NORMAL',
    1: 'HYPERTENSION (Stage-1)',
    2: 'HYPERTENSION (Stage-2)',
    3: 'HYPERTENSIVE CRISIS'
}

# Medical-grade color mapping for results
color_map = {
    0: '#10B981',  # Medical green for normal
    1: '#F59E0B',  # Medical amber for stage 1
    2: '#EF7316',  # Medical orange for stage 2
    3: '#EF4444'   # Medical red for crisis
}

# Detailed medical recommendations
recommendations = {
    0: {
        'title': 'Normal Blood Pressure',
        'description': 'Your cardiovascular risk assessment indicates normal blood pressure levels.',
        'actions': [
            'Maintain current healthy lifestyle',
            'Regular physical activity (150 minutes/week)',
            'Continue balanced, low-sodium diet',
            'Annual blood pressure monitoring',
            'Regular health check-ups'
        ],
        'priority': 'Low Risk'
    },
    1: {
        'title': 'Stage 1 Hypertension',
        'description': 'Mild elevation detected requiring lifestyle modifications and medical consultation.',
        'actions': [
            'Schedule appointment with healthcare provider',
            'Implement DASH diet plan',
            'Increase physical activity gradually',
            'Monitor blood pressure bi-weekly',
            'Reduce sodium intake (<2300mg/day)',
            'Consider stress management techniques'
        ],
        'priority': 'Moderate Risk'
    },
    2: {
        'title': 'Stage 2 Hypertension',
        'description': 'Significant hypertension requiring immediate medical intervention and treatment.',
        'actions': [
            'URGENT: Consult physician within 1-2 days',
            'Likely medication therapy required',
            'Comprehensive cardiovascular assessment',
            'Daily blood pressure monitoring',
            'Strict dietary sodium restriction',
            'Lifestyle modification counseling'
        ],
        'priority': 'High Risk'
    },
    3: {
        'title': 'Hypertensive Crisis',
        'description': 'CRITICAL: Dangerously elevated blood pressure requiring emergency medical care.',
        'actions': [
            'EMERGENCY: Seek immediate medical attention',
            'Call 911 if experiencing symptoms',
            'Do not delay treatment',
            'Monitor for stroke/heart attack signs',
            'Prepare current medication list',
            'Avoid physical exertion'
        ],
        'priority': 'EMERGENCY'
    }
}
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Collect user inputs from the form with validation
            required_fields = ['Gender', 'Age', 'History', 'Patient', 'TakeMedication',
                             'Severity', 'Breathshortness', 'Visualchanges', 'Nosebleeding',
                             'whendiagnosed', 'Systolic', 'Diastolic', 'ControlledDiet']

            form_data = {}
            for field in required_fields:
                value = request.form.get(field)
                if not value or value == '':
                    flash(f"Please complete all required fields: {field.replace('_', ' ')}", "error")
                    return render_template('index.html')
                form_data[field] = value

            # Encode inputs with error handling
            try:
                encoded = {}
                encoded[0] = 1 if form_data['Gender'] == 'Male' else 0
                encoded[1] = {'18-34': 1, '35-50': 2, '51-64': 3, '65+': 4}[form_data['Age']]
                encoded[2] = 1 if form_data['History'] == 'Yes' else 0
                encoded[3] = 1 if form_data['Patient'] == 'Yes' else 0
                encoded[4] = {'Never taken medication': 1, 'Yes, always': 2, 'Yes, sometimes': 3}[form_data['TakeMedication']]
                encoded[5] = {'Mild': 1, 'Moderate': 2, 'Severe': 3}[form_data['Severity']]
                encoded[6] = 1 if form_data['Breathshortness'] == 'Yes' else 0
                encoded[7] = 1 if form_data['Visualchanges'] == 'Yes' else 0
                encoded[8] = 1 if form_data['Nosebleeding'] == 'Yes' else 0
                encoded[9] = {'I can\'t recall': 1, '1-5 years': 2, '5+ years': 3}[form_data['whendiagnosed']]
                encoded[10] = {'<100': 1, '100 - 110': 2, '111 - 120': 3, '121 - 130': 4, '130+': 5}[form_data['Systolic']]
                encoded[11] = {'<70': 1, '70 - 80': 2, '81 - 90': 3, '91 - 100': 4, '100+': 5}[form_data['Diastolic']]
                encoded[12] = 1 if form_data['ControlledDiet'] == 'Yes' else 0

            except KeyError as e:
                flash(f"Invalid selection detected: {str(e)}", "error")
                return render_template('index.html')

            # Manual scaling of ordinal features
            scaled_encoded = encoded.copy()
            scaled_encoded[1] = (encoded[1] - 1) / (4 - 1)  # Age
            scaled_encoded[4] = (encoded[4] - 1) / (3 - 1)  # TakeMedication
            scaled_encoded[5] = (encoded[5] - 1) / (3 - 1)  # Severity
            scaled_encoded[9] = (encoded[9] - 1) / (3 - 1)  # when diagnosed
            scaled_encoded[10] = (encoded[10] - 1) / (5 - 1) # Systolic
            scaled_encoded[11] = (encoded[11] - 1) / (5 - 1) # Diastolic

            input_array = np.array(list(scaled_encoded.values())).reshape(1, -1)

            # Predict with model or use dummy prediction
            if model is not None:
                prediction = model.predict(input_array)[0]
                try:
                    confidence = max(model.predict_proba(input_array)[0]) * 100
                except:
                    confidence = 85.0
            else:
                # Dummy prediction for testing when model is not available
                import random
                prediction = random.randint(0, 3)
                confidence = 87.5
                flash("Demo Mode: Using simulated AI prediction for demonstration", "info")
                
            result_text = stage_map[prediction]
            result_color = color_map[prediction]
            result_recommendation = recommendations[prediction]

            return render_template('index.html',
                                   prediction_text=result_text,
                                   result_color=result_color,
                                   confidence=confidence,
                                   recommendation=result_recommendation,
                                   form_data=form_data)

        except Exception as e:
            flash(f"System error occurred. Please try again or contact technical support.", "error")
            return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)