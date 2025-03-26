from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler

# Load the trained model and fitted scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')  # Load the fitted scaler

# Create Flask app
app = Flask(__name__)

# Route for the homepage
@app.route('/')
def home():
    # Render the HTML form
    return render_template('index.html')

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'error': 'Invalid input: "features" key is missing'}), 400

        input_data = np.array(data['features']).reshape(1, -1)
        
        # Scale the input data using the fitted scaler
        input_data_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_data_scaled)
        
        # Return the prediction as JSON
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        # Log the error and return a 500 response
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)