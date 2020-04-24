import tensorflow as tf
import numpy as np
import random
import os
import matplotlib.pyplot as plt
Picture = np.load(r"D:/snuml/jik-cam/video/1_pic.npy")
Label = np.load(r"D:/snuml/jik-cam/video/1_lbl.npy")
for i in range(2,6):
    P=np.load(r"D:/snuml/jik-cam/video/"+str(i)+"_pic.npy")
    L=np.load(r"D:/snuml/jik-cam/video/"+str(i)+"_lbl.npy")
    Picture = np.concatenate((Picture,P), axis=0)
    Label = np.concatenate((Label,L), axis=0)
    print(i)

combine = list(zip(Picture, Label))
random.shuffle(combine)
Picture, Label = zip(*combine)

print("shuffle")

trainlist, testlist = [], []
for i in range(250):
    trainlist.append([Picture[i],Label[i]])
        
for i in range(250,500):
    testlist.append([Picture[i],Label[i]])

IMG_H = 720
IMG_W = 1280
IMG_C = 3

def batch(path, batch_size):
    img, label = [], []
    for i in range(batch_size):
        img.append(path[0][0])
        label.append(path[0][1])
        path.pop(0)
        
    return img, label

num_class = 2

with tf.Graph().as_default() as g:
    X = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, IMG_C])
    Y = tf.placeholder(tf.int32, [None])
    
    with tf.variable_scope('CNN'):
        net = tf.layers.conv2d(X, 20, 3, (2, 2), padding='same', activation=tf.nn.relu)
        net = tf.layers.max_pooling2d(net, 2, 2)
        net = tf.layers.conv2d(net, 40, 3, (2, 2), padding='same', activation=tf.nn.relu)
        net = tf.layers.flatten(net)
     
        out = tf.layers.dense(net, num_class)
        
    with tf.variable_scope('Loss'):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels= Y, logits=out))
        
    train = tf.train.AdamOptimizer(1e-3).minimize(loss)
    saver = tf.train.Saver()

np.sum([np.product(var.shape) for var in g.get_collection('trainable_variables')]).value

batch_size = 25
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        batch_data, batch_label = batch(trainlist, batch_size )
        _, l = sess.run([train, loss], feed_dict = {X: batch_data, Y: batch_label})
        print(i, l)
        
    saver.save(sess, 'logs/model.ckpt', global_step = i+1)

acc = 0
l = len(testlist)
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.latest_checkpoint('logs')
    if checkpoint:
        saver.restore(sess, checkpoint)
        print("success")
    for i in range(l):
        batch_data, batch_label = batch(testlist, 1)
        logit = sess.run(out, feed_dict = {X:batch_data})
        if np.argmax(logit[0]) == batch_label[0]:
            acc += 1
        print(i,logit[0],np.argmax(logit[0]), batch_label[0])
            
    print(acc/l)