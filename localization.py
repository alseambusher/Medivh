import tensorflow as tf
import numpy as np
import pandas as pd

from helper.vgg import Detector
from helper import load_image
import os
import config
from helper.read_data import caltech256

model_path = config.model_root
trainset, testset, n_labels = caltech256()

learning_rate = tf.placeholder( tf.float32, [])
images_tf = tf.placeholder( tf.float32, [None, 224, 224, 3], name="images")
labels_tf = tf.placeholder( tf.int64, [None], name='labels')

detector = Detector(config.weight_path, n_labels)

p1,p2,p3,p4,conv5, conv6, gap, output = detector.inference(images_tf)
loss_tf = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( output, labels_tf ))

weights_only = filter( lambda x: x.name.endswith('W:0'), tf.trainable_variables() )
weight_decay = tf.reduce_sum(tf.pack([tf.nn.l2_loss(x) for x in weights_only])) * config.weight_decay_rate
loss_tf += weight_decay

sess = tf.InteractiveSession()
saver = tf.train.Saver( max_to_keep=50 )

optimizer = tf.train.MomentumOptimizer( learning_rate, config.momentum)
grads_and_vars = optimizer.compute_gradients( loss_tf )
grads_and_vars = map(lambda gv: (gv[0], gv[1]) if ('conv6' in gv[1].name or 'GAP' in gv[1].name) else (gv[0]*0.1, gv[1]), grads_and_vars)
train_op = optimizer.apply_gradients(grads_and_vars)
tf.global_variables_initializer().run()

# if pretrained_model_path:
#     print "Pretrained"
#     saver.restore(sess, pretrained_model_path)

testset.index  = range( len(testset) )

f_log = open('results/log.txt', 'w')

iterations = 0
loss_list = []
for epoch in range(config.n_epochs):

    trainset.index = range( len(trainset) )
    trainset = trainset.ix[ np.random.permutation( len(trainset) )]

    for start, end in zip(
        range( 0, len(trainset)+config.batch_size, config.batch_size),
        range(config.batch_size, len(trainset)+config.batch_size, config.batch_size)):

        current_data = trainset[start:end]
        current_image_paths = current_data['image_path'].values
        current_images = np.array(map(lambda x: load_image(x), current_image_paths))

        good_index = np.array(map(lambda x: x is not None, current_images))

        current_data = current_data[good_index]
        current_images = np.stack(current_images[good_index])
        current_labels = current_data['label'].values

        _, loss_val, output_val = sess.run(
                [train_op, loss_tf, output],
                feed_dict={
                    learning_rate: config.init_learning_rate,
                    images_tf: current_images,
                    labels_tf: current_labels
                    })

        loss_list.append( loss_val )

        iterations += 1
        if iterations % 5 == 0:
            print "======================================"
            print "Epoch", epoch, "Iteration", iterations
            print "Processed", start, '/', len(trainset)

            label_predictions = output_val.argmax(axis=1)
            acc = (label_predictions == current_labels).sum()

            print "Accuracy:", acc, '/', len(current_labels)
            print "Training Loss:", np.mean(loss_list)
            print "\n"
            loss_list = []

    n_correct = 0
    n_data = 0
    for start, end in zip(
            range(0, len(testset)+config.batch_size, config.batch_size),
            range(config.batch_size, len(testset)+config.batch_size, config.batch_size)
            ):
        current_data = testset[start:end]
        current_image_paths = current_data['image_path'].values
        current_images = np.array(map(lambda x: load_image(x), current_image_paths))

        good_index = np.array(map(lambda x: x is not None, current_images))

        current_data = current_data[good_index]
        current_images = np.stack(current_images[good_index])
        current_labels = current_data['label'].values

        output_vals = sess.run(
                output,
                feed_dict={images_tf:current_images})

        label_predictions = output_vals.argmax(axis=1)
        acc = (label_predictions == current_labels).sum()

        n_correct += acc
        n_data += len(current_data)

    acc_all = n_correct / float(n_data)
    f_log.write('epoch:'+str(epoch)+'\tacc:'+str(acc_all) + '\n')
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
    print 'epoch:'+str(epoch)+'\tacc:'+str(acc_all) + '\n'
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

    saver.save( sess, os.path.join( model_path, 'model'), global_step=epoch)

    config.init_learning_rate *= 0.99
