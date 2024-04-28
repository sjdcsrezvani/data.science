import json
import pickle
import numpy as np
__locations = None
__data_columns = None
__model = None

def get_estimated_price(area,age,rooms,elavator,parking,Warehouse,location):
    global __data_columns
    global __locations
    global __model
    try:
        loc_index = __data_columns.index(location)
    except:
        loc_index = -1
    t = len(__data_columns)
    x = np.zeros(t)
    x[0] = area
    x[1] = age
    x[2] = rooms
    x[3] = elavator
    x[4] = parking
    x[5] = Warehouse
    if loc_index >= 0:
        x[loc_index]= 1

    return round(__model.predict([x])[0], 1)

def get_location_names():
    return __locations
def load_saved_artifacts():
    global __data_columns
    global __locations

    with open('./artifacts/columns.json','r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[7:]
    global __model

    with open('./artifacts/tehran_home_prices_model.pickle', 'rb') as f:
        __model = pickle.load(f)


if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price(90,10,2,0,0,0,'کوی فردوس'))