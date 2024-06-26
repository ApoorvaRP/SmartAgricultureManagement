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
    humidity  = float(datas[2].split('%\t%')[0])
    print(humidity)
 
    return render_template('index.html',  humidity=humidity)

    ser.close()
    return render_template('index.html', humidity=humidity)

@app.route('/hum_monitor', methods=['POST'])
def hum_monitor():
    try:
        # Extract features from form
      
        humidity = request.form.get('humidity', type=float)
        category = 'Normal'
        if humidity > 70:
            category = 'High'
        elif humidity < 30:
            category = 'Low'
        return render_template('result.html', humidity=humidity, category=category)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True,port=5005)