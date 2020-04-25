import tensorflow as tf
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import gc

Global_Step = 0

def batch(path, batch_size, n):
    img, label = [], []
    for i in range(batch_size):
        img.append(path[i+(n-1)*batch_size][0])
        label.append(path[i+(n-1)*batch_size][1])
        
    return img, label

IMG_H = 720
IMG_W = 1280
IMG_C = 3
ImageNum = 300

NpyPath = input("\n\tPlease Type Absolute-Path of your TEST mp4's npy files. (Exit Key is Ctrl+C.) \n\nPath (Ex: C:/peekaboo): ")

Picture=np.load(NpyPath+"_pic.npy")
Label=np.load(NpyPath+"_lbl.npy")

trainlist, testlist = [], []

for i in range(ImageNum):
    testlist.append([Picture[i],Label[i]])

num_class = 2

with tf.Graph().as_default() as g:
    X = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, IMG_C])
    Y = tf.placeholder(tf.int32, [None])
    
    with tf.variable_scope('CNN'):
        net = tf.layers.conv2d(X, 20, 3, (2, 2), padding='same', activation=tf.nn.relu)
        net = tf.layers.max_pooling2d(net, 2, 2)
        net = tf.layers.conv2d(net, 40, 3, (2, 2), padding='same', activation=tf.nn.relu)
        net = tf.layers.max_pooling2d(net, 2, 2)
        net = tf.layers.flatten(net)

        out = tf.layers.dense(net, num_class)
        
    with tf.variable_scope('Loss'):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels= Y, logits=out))
        
    train = tf.train.AdamOptimizer(1e-3).minimize(loss)
    saver = tf.train.Saver()

acc = 0
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.latest_checkpoint('logs')
    if checkpoint:
        saver.restore(sess, checkpoint)
    for i in range(len(testlist)):
        batch_data, batch_label = batch(testlist, 1, i)
        logit = sess.run(out, feed_dict = {X:batch_data})
        if np.argmax(logit[0]) == batch_label[0]:
            acc += 1
        else:
            print(logit[0], batch_label[0])

    print(acc/len(testlist))
