from flask import Flask, render_template, request, session
import pandas as pd
import pickle
import ollama

# Flask App
app = Flask(__name__)

app.secret_key = 'sdakq3wjp29q3jerwo349u1205p4ejfoq8234m0nf380623456245njdkfsgh'

# ollama
ollama.chat(model='llama3.1', messages=None)

messages = [
    {
        'role': 'system',
        'content': 'you are an ai based health consultant. we provide you the disease and you will provide consultant based on that '
    }
]


# Load Model
with open('../model_2000.pkl', 'rb') as file:
    model = pickle.load(file)

# Home Page
@app.route('/')
def home():
    disease = session.get('disease', None)
    return render_template("index.html")

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    height = float(request.form['height'])
    weight = float(request.form['weight'])
    age = float(request.form['age'])
    bloodPressure = float(request.form['bloodPressure'])
    
    input_data = pd.DataFrame([[height, weight, age, bloodPressure]], columns=['Height_cm', 'Weight_kg', 'Age', 'Blood_Pressure_mmHg'])
    
    predictions = model.predict(input_data)
    disease = predictions[0]

    ollama_input = [{'role': 'user', 'content': f"the disease is {disease}, age is {age}, weight is {weight}, blood pressure {bloodPressure},height is {height} give health consultant."}]
    response = ollama.chat(model='llama3.1', messages=ollama_input)

    # just pass this to index.html
    print(f"ollama response: {response}") 

    return render_template("index.html", disease = disease)

if __name__ == "__main__":
    app.run(debug=True, port="80", host="0.0.0.0")