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

# For Saving precious memory, let's divide into two parts.
# 
# Step 1
IMG_H = None
IMG_W = None
IMG_C = None
ImageNum = None # 300

for i in range(3):
    NpyPath = input("\n\tPlease Type Absolute-Path of your " +str(i)+ "th mp4's npy files \n\tEXCEPT name extension. (Exit Key is Ctrl+C.) \n\nPath (Ex: C:/peekaboo): ")
    
    Picture=np.load(NpyPath+"_pic.npy")
    Label=np.load(NpyPath+"_lbl.npy")

    s = np.arange(Picture.shape[0])
    np.random.shuffle(s)

    Picture = Picture[s]
    Label = Label[s]

    for j in range(3):
        np.save('./_picMixed_'+str(i)+'_'+str(j),Picture[Picture.shape[0]//3*i:Picture.shape[0]//3*(i+1)])
        np.save('./_lblMixed_'+str(i)+'_'+str(j),Label[Label.shape[0]//3*i:Label.shape[0]//3*(i+1)])

    ### For global
    IMG_H = Picture.shape[1]
    IMG_W = Picture.shape[2]
    IMG_C = Picture.shape[3]
    ImageNum = Picture.shape[0]
    ###

    del Picture
    del Label

    Picture = None
    Label = None
    gc.collect()

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

Picture = None
Label = None

for ii in range(3):
    for j in range(3):
        if(j==0):
            Picture = np.load('./_picMixed_'+str(j)+'_'+str(ii)+'.npy')
            Label = np.load('./_lblMixed_'+str(j)+'_'+str(ii)+'.npy')
        else:
            Picture = np.append(Picture,np.load('./_picMixed_'+str(j)+'_'+str(ii)+'.npy'),axis=0)
            Label = np.append(Label,np.load('./_lblMixed_'+str(j)+'_'+str(ii)+'.npy'),axis=0)

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

    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        if (ii != 0):
            checkpoint = tf.train.latest_checkpoint('logs')
            if checkpoint:
                saver.restore(sess, checkpoint)
        for i in range(TrainNum//batch_size):
            batch_data, batch_label = batch(trainlist, batch_size,i )
            _, l = sess.run([train, loss], feed_dict = {X: batch_data, Y: batch_label})
            print(i, l)
            Global_Step += 1
        
        saver.save(sess, 'logs/model.ckpt', global_step = Global_Step)

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

    del Picture
    del Label

    Picture = None
    Label = None
    gc.collect()

