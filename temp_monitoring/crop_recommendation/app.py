from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import serial

app = Flask(__name__)

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
    temperature =  float(datas[2].split('%\t%')[1].split(': ')[1].replace('°C',''))
    print(temperature)
 
    return render_template('index.html', temperature=temperature)

    ser.close()
    return render_template('index.html', temperature=temperature)

@app.route('/temp_monitor', methods=['POST'])
def temp_monitor():
    try:
        # Extract features from form
      
        temperature = request.form.get('temperature', type=float)
        category = 'Normal'
        if temperature > 30:
            category = 'High'
        elif temperature < 10:
            category = 'Low'
        return render_template('result.html', temperature=temperature, category=category)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True,port=5003)
