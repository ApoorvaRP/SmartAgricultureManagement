from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import serial

app = Flask(__name__)

# Load the RandomForest model
with open('RandomForest.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/', methods=['GET'])
def home():
    # Open serial port
    print("Request is coming")
    datas=[]
    ser = serial.Serial('COM3', 9600)
    condition=True
    while(condition==True):
        if ser.in_waiting>0:
            data = ser.readline().decode('utf-8').strip()
            datas.append(data)
            print(datas)
            if(len(datas)>3):
                condition=False
    print("Final")
    print(datas)
    humidity  = float(datas[2].split('%\t%')[0])
    temperature =  float(datas[2].split('%\t%')[1].split(': ')[1].replace('°C',''))
    print(humidity)
    print(temperature)
    # data = ser.readline().decode('utf-8').strip()
    # print("Received data:", data)
    # data_parts = data.split(',')
    # soil_moisture = data_parts[0]
    # temperature = data_parts[1]
    # humidity = data_parts[2]
    return render_template('index.html', temperature=temperature, humidity=humidity)

    # arduino_data = ser.readline().decode('utf-8')
    # print(arduino_data)
    # temperature = float(arduino_data[2].split()[0])
    # humidity = float(arduino_data[1])
    # Close serial port
    ser.close()
    return render_template('index.html', temperature=temperature, humidity=humidity)

@app.route('/recommend_crop', methods=['POST'])
def recommend_crop():
    try:
        # Extract features from form
        N = request.form.get('N', type=float)
        P = request.form.get('P', type=float)
        K = request.form.get('K', type=float)
        temperature = request.form.get('temperature', type=float)
        humidity = request.form.get('humidity', type=float)
        ph = request.form.get('ph', type=float)
        rainfall = request.form.get('rainfall', type=float)
        
        # Making a prediction
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(features)
        
        # Return the prediction
        return render_template('result.html', recommended_crop=prediction[0])
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
