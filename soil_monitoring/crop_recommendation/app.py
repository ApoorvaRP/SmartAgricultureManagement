from flask import Flask, render_template, request, jsonify
import serial

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    try:
        # Open serial port
        print("Request is coming")
        datas = []
        ser = serial.Serial('COM3', 9600)
        condition = True
        while condition:
            if ser.in_waiting > 0:
                data = ser.readline().decode('utf-8').strip()
                datas.append(data)
                print(datas)
                if len(datas) > 3:
                    condition = False
        print("Final data:", datas)
        
        # Extracting soil moisture from the first received data
        soil_moisture_str = [item for item in datas if 'Soil Moisture' in item][0]  # Find the line containing 'Soil Moisture'
        soil_moisture = float(soil_moisture_str.split(': ')[1].replace('%', ''))  # Extracting the numerical value
        print("Soil Moisture:", soil_moisture)

        # Close serial port
        ser.close()
        
        return render_template('index.html', soilmoisture=soil_moisture)
    
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)})

@app.route('/soil_monitor', methods=['POST'])
def soil_monitor():
    try:
        # Extract features from form
        soilmoisture = request.form.get('soilmoisture', type=float)
        category = 'Optimal'
        if soilmoisture > 75:
            category = 'High'
        elif soilmoisture < 25:
            category = 'Low'
        elif soilmoisture > 25 and soilmoisture <= 50:
            category = 'Medium'
        return render_template('result.html', soilmoisture=soilmoisture, category=category)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5004)
