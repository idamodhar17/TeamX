from flask import Flask, render_template, request, session, jsonify
import pandas as pd
import pickle

# Flask App
app = Flask(__name__)

app.secret_key = 'sdakq3wjp29q3jerwo349u1205p4ejfoq8234m0nf380623456245njdkfsgh'

try:
    with open('./model_2000.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    print("Model file not found. Please check the file path.")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        age = float(request.form['age'])
        bloodPressure = float(request.form['bloodPressure'])

        input_data = pd.DataFrame([[height, weight, age, bloodPressure]], columns=['Height_cm', 'Weight_kg', 'Age', 'Blood_Pressure_mmHg'])
        predictions = model.predict(input_data)
        disease = predictions[0]
        
        suggestion = "Maintain a healthy lifestyle with balanced nutrition and regular exercise."

        return jsonify(disease=disease, suggestion=suggestion)
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify(error="An error occurred during prediction.")

if __name__ == "__main__":
    app.run(debug=True, port="80", host="0.0.0.0")