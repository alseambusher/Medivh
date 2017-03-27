import json
from PIL import Image
import numpy as np
from sklearn.svm import SVC
from flask import Flask, jsonify
from flask.ext.cors import CORS, cross_origin
from scipy import stats
import subprocess

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

data = json.load(open("clicks.json"))["data"]
im = Image.open('0_1.png')
im = im.convert('RGB')
print im.size
x_y = []
trainX = []
trainY = []
conv_size = 5
print len(data)
print im.size[0]/conv_size
print im.size[1]/conv_size
total = 0
def get_num_clicks(x, y):
    result = 0
    for click in data:
        if click[3] >= x*conv_size and click[3] <= x*conv_size + conv_size:
            if click[4] >= y*conv_size and click[4] <= y*conv_size + conv_size:
                result += 1
    if result > 0:
        global total
        total += 1
    return result

for x in range(im.size[0]/conv_size):
    for y in range(im.size[1]/conv_size):
        # print x,y, x*5, y*6
        region = im.crop((x*conv_size,y*conv_size,x*conv_size+conv_size,y*conv_size+conv_size))
        # print np.array(region)
        # print np.reshape(region, (25, 3))
        # trainX.append(stats.mode(np.reshape(region, (conv_size*conv_size, 3)))[0][0])
        # print map(int, np.rint(np.mean(np.mean(region, axis=1), axis=0)).tolist())
        trainX.append(map(int, np.rint(np.mean(np.mean(region, axis=1), axis=0)).tolist()))
        # trainX.append(np.rint(np.mean(np.mean(region, axis=1), axis=0)))
        trainY.append(get_num_clicks(x, y))
        x_y.append([x*conv_size, y*conv_size])

print total

model = SVC(C=10)
model.fit(trainX, trainY)
predictions = model.predict(trainX)
print sum(predictions!=0)

@app.route('/')
@cross_origin()
def mouse():
    process = subprocess.Popen(["/usr/local/bin/node", "../mouse-map.js/app.js"], stdout=subprocess.PIPE)
    process.wait()
    print process.returncode
    _im = Image.open('0.png')
    _im = _im.convert('RGB')
    X = []
    for x in range(_im.size[0]/conv_size):
        for y in range(_im.size[1]/conv_size):
            region = _im.crop((x*conv_size,y*conv_size,x*conv_size+conv_size,y*conv_size+conv_size))
            X.append(map(int, np.rint(np.mean(np.mean(region, axis=1), axis=0)).tolist()))

    pred = model.predict(X)
    points = []
    for p in xrange(len(pred)):
        if pred[p] != 0:
            points.append({"x": int((x_y[p][0]*1440)/1014), "y": x_y[p][1], "value": pred[p]})
            points.append({"x": int((x_y[p][0]*1440)/1014), "y": x_y[p][1], "value": pred[p]})
    return jsonify({"results": points})


if __name__ == '__main__':
    app.run(debug=True, port=8001, use_reloader=False)
