from flask import Flask, render_template, request, flash
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'

# Load trained model
try:
    model = joblib.load("logreg_model.pkl")
except FileNotFoundError:
    print("Warning: Model file not found. Using dummy predictions.")
    model = None

# Color mapping using STRING labels
color_map = {
    'NORMAL': '#10B981',
    'HYPERTENSION (Stage-1)': '#F59E0B',
    'HYPERTENSION (Stage-2)': '#EF7316',
    'HYPERTENSIVE CRISIS': '#EF4444'
}

# Detailed medical recommendations using STRING labels
recommendations = {
    'NORMAL': {
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

    'HYPERTENSION (Stage-1)': {
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

    'HYPERTENSION (Stage-2)': {
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

    'HYPERTENSIVE CRISIS': {
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
    try:
        required_fields = [
            'Gender', 'Age', 'History', 'Patient', 'TakeMedication',
            'Severity', 'Breathshortness', 'Visualchanges', 'Nosebleeding',
            'whendiagnosed', 'Systolic', 'Diastolic', 'ControlledDiet'
        ]

        form_data = {}
        for field in required_fields:
            value = request.form.get(field)
            if not value:
                flash("Please complete all required fields", "error")
                return render_template('index.html')
            form_data[field] = value

        # -------- ENCODING --------
        encoded = {
            'Gender': 1 if form_data['Gender'] == 'Male' else 0,
            'Age': {'18-34': 1, '35-50': 2, '51-64': 3, '65+': 4}[form_data['Age']],
            'History': 1 if form_data['History'] == 'Yes' else 0,
            'Patient': 1 if form_data['Patient'] == 'Yes' else 0,
            'TakeMedication': {
                'Never taken medication': 1,
                'Yes, always': 2,
                'Yes, sometimes': 3
            }[form_data['TakeMedication']],
            'Severity': {'Mild': 1, 'Moderate': 2, 'Severe': 3}[form_data['Severity']],
            'Breathshortness': 1 if form_data['Breathshortness'] == 'Yes' else 0,
            'Visualchanges': 1 if form_data['Visualchanges'] == 'Yes' else 0,
            'Nosebleeding': 1 if form_data['Nosebleeding'] == 'Yes' else 0,
            'whendiagnosed': {
                'less than one year': 1,
                '1-5 years': 2,
                '5+ years': 3
            }[form_data['whendiagnosed']],
            'Systolic': {
                '<100': 1, '100 - 110': 2, '111 - 120': 3,
                '121 - 130': 4, '130+': 5
            }[form_data['Systolic']],
            'Diastolic': {
                '<70': 1, '70 - 80': 2, '81 - 90': 3,
                '91 - 100': 4, '100+': 5
            }[form_data['Diastolic']],
            'ControlledDiet': 1 if form_data['ControlledDiet'] == 'Yes' else 0
        }

        # -------- SCALING --------
        encoded['Age'] = (encoded['Age'] - 1) / 3
        encoded['TakeMedication'] = (encoded['TakeMedication'] - 1) / 2
        encoded['Severity'] = (encoded['Severity'] - 1) / 2
        encoded['whendiagnosed'] = (encoded['whendiagnosed'] - 1) / 2
        encoded['Systolic'] = (encoded['Systolic'] - 1) / 4
        encoded['Diastolic'] = (encoded['Diastolic'] - 1) / 4

        feature_order = list(encoded.keys())
        input_array = np.array([[encoded[f] for f in feature_order]])

        # -------- PREDICTION --------
        if model:
            prediction_label = model.predict(input_array)[0]
            confidence = max(model.predict_proba(input_array)[0]) * 100
        else:
            prediction_label = 'HYPERTENSION (Stage-2)'
            confidence = 85.0
            flash("Demo Mode Prediction", "info")

        return render_template(
            'index.html',
            prediction_text=prediction_label,
            result_color=color_map[prediction_label],
            confidence=confidence,
            recommendation=recommendations[prediction_label],
            form_data=form_data
        )

    except Exception as e:
        print("ERROR:", e)
        flash(str(e), "error")
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
