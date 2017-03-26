import tensorflow as tf
import pandas as pd
import numpy as np

from helper.vgg import Detector
from helper import load_image

import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
import config
import tempfile
import urllib
from flask import Flask, send_file, request

app = Flask(__name__)

model_path = config.model_root + 'model-4'
batch_size = 1

label_dict = pd.read_pickle(config.label_dict_path)
n_labels = len(label_dict)

images_tf = tf.placeholder( tf.float32, [None, 224, 224, 3], name="images")
labels_tf = tf.placeholder( tf.int64, [None], name='labels')

detector = Detector(config.weight_path, n_labels)
c1,c2,c3,c4,conv5, conv6, gap, output = detector.inference(images_tf)
classmap = detector.get_classmap(labels_tf, conv6)

sess = tf.InteractiveSession()
saver = tf.train.Saver()

saver.restore(sess, model_path)


@app.route('/image')
def image():
    path = request.args.get("path")
    print path

    with tempfile.NamedTemporaryFile(suffix=".png") as tmpfile:
        urllib.urlretrieve(path, tmpfile.name)
        current_image, shape = load_image(tmpfile.name)

    conv6_val, output_val = sess.run(
        [conv6, output],
        feed_dict={
            images_tf: np.array([current_image])
        })
    label_predictions = output_val.argmax(axis=1)

    classmap_vals = sess.run(
        classmap,
        feed_dict={
            labels_tf: label_predictions,
            conv6: conv6_val
        })

    classmap_vis = map(lambda x: ((x-x.min())/(x.max()-x.min())), classmap_vals)

    ori = skimage.transform.resize(current_image, [shape[0], shape[1]])
    vis = skimage.transform.resize(classmap_vis[0], [shape[0], shape[1]])

    plt.clf()
    plt.axis("off")
    plt.imshow(ori)
    plt.imshow(vis, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest')
    with tempfile.NamedTemporaryFile(suffix=".png") as tmpfile:
        plt.savefig(tmpfile.name, transparent=True, bbox_inches='tight', pad_inches=0)
        return send_file(tmpfile.name, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, port=8000, use_reloader=False)
