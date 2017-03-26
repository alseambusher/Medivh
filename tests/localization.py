import tensorflow as tf
import pandas as pd
import numpy as np

from helper.vgg import Detector
from helper import load_image

import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
import config

model_path = config.model_root + 'model-4'
batch_size = 1

testset = pd.read_pickle(config.testset_path)[::-1][:20]
label_dict = pd.read_pickle(config.label_dict_path)
n_labels = len(label_dict)

testset["image_path"][2569] = "/Users/alse/code/Medivh/data/Stanford40/JPEGImages/applauding_191.jpg"
testset["label"][2569] = 255
testset["label_name"][2569] = "toad"
testset["image_path"][2568] = "/Users/alse/code/Medivh/data/Stanford40/JPEGImages/applauding_191.jpg"
testset["label"][2568] = 256
testset["label_name"][2568] = "cutter"
print testset
images_tf = tf.placeholder( tf.float32, [None, 224, 224, 3], name="images")
labels_tf = tf.placeholder( tf.int64, [None], name='labels')

detector = Detector(config.weight_path, n_labels)
c1,c2,c3,c4,conv5, conv6, gap, output = detector.inference(images_tf)
classmap = detector.get_classmap(labels_tf, conv6)

sess = tf.InteractiveSession()
saver = tf.train.Saver()

saver.restore( sess, model_path )

for start, end in zip(
    range( 0, len(testset)+batch_size, batch_size),
    range(batch_size, len(testset)+batch_size, batch_size)):

    current_data = testset[start:end]
    current_image_paths = current_data['image_path'].values
    current_images = np.array(map(lambda x: load_image(x), current_image_paths))

    good_index = np.array(map(lambda x: x is not None, current_images))

    current_data = current_data[good_index]
    current_image_paths = current_image_paths[good_index]
    current_images = np.stack(current_images[good_index])
    current_labels = current_data['label'].values
    current_label_names = current_data['label_name'].values

    conv6_val, output_val = sess.run(
            [conv6, output],
            feed_dict={
                images_tf: current_images
                })

    label_predictions = output_val.argmax( axis=1 )
    print current_image_paths
    print label_predictions
    acc = (label_predictions == current_labels).sum()

    classmap_vals = sess.run(
            classmap,
            feed_dict={
                labels_tf: label_predictions,
                conv6: conv6_val
                })

    # classmap_answer = sess.run(
    #         classmap,
    #         feed_dict={
    #             labels_tf: current_labels,
    #             conv6: conv6_val
    #             })

    classmap_vis = map(lambda x: ((x-x.min())/(x.max()-x.min())), classmap_vals)
    # classmap_vis = map(lambda x: ((x-x.min())/(x.max()-x.min())), classmap_answer)

    for vis, ori,ori_path, l_name in zip(classmap_vis, current_images, current_image_paths, current_label_names):
        print (l_name)
        ori = skimage.transform.resize( ori, [451,300] )
        vis = skimage.transform.resize( vis, [451,300] )
        plt.imshow( ori )
        plt.imshow( vis, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest' )
        plt.show()

#        vis_path = '../results/'+ ori_path.split('/')[-1]
#        vis_path_ori = '../results/'+ori_path.split('/')[-1].split('.')[0]+'.ori.jpg'
#        skimage.io.imsave( vis_path, vis )
#        skimage.io.imsave( vis_path_ori, ori )

