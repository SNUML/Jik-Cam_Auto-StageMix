import tensorflow as tf
import numpy as np
import random
import os
import matplotlib.pyplot as plt

Picture=np.load(r"C:/pycho_pic.npy")
Label=np.load(r"C:/pycho_lbl.npy")

ImageNum = Picture.shape[0]
TestRatio = 0.1
TrainNum = int(float(ImageNum)*(1-TestRatio))
batch_size = 10

s = np.arange(Picture.shape[0])
np.random.shuffle(s)

Picture = Picture[s]
Label = Label[s]

trainlist, testlist = [], []
for i in range(TrainNum):
    trainlist.append([Picture[i],Label[i]])
        
for i in range(TrainNum,ImageNum):
    testlist.append([Picture[i],Label[i]])

IMG_H = Picture.shape[1]
IMG_W = Picture.shape[2]
IMG_C = Picture.shape[3]

def batch(path, batch_size, n):
    img, label = [], []
    for i in range(batch_size):
        img.append(path[i+(n-1)*batch_size][0])
        label.append(path[i+(n-1)*batch_size][1])
        
    return img, label

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


with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(int(TrainNum/batch_size)):
        batch_data, batch_label = batch(trainlist, batch_size,i )
        _, l = sess.run([train, loss], feed_dict = {X: batch_data, Y: batch_label})
        print(i, l)
        
    saver.save(sess, 'logs/model.ckpt', global_step = i+1)

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