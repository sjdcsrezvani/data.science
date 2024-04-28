from flask import Flask, request, jsonify
import util
app = Flask(__name__)


@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    response = jsonify({
        'locations': util.get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route('/predict_home_price', methods=['GET','POST'])
def predict_home_price():
    area = float(request.form['total_area'])
    location = request.form['location']
    age = int(request.form['age'])
    rooms = int(request.form['rooms'])
    elevator = int(request.form['elevator'])
    parking = int(request.form['parking'])
    warehouse = int(request.form['warehouse'])
    response = jsonify({
        'estimated_price': util.get_estimated_price(area,age,rooms,elevator,parking,warehouse,location)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__ == '__main__':
    print('starting flask server')
    util.load_saved_artifacts()
    app.run()