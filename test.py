# !usr/bin/python
# Author:das
# -*-coding: utf-8 -*-
import tensorflow as tf
import forward
import backward
BATCH_SIZE =100
STEPS = 10
from tensorflow.examples.tutorials.mnist import input_data
MODEL_NAME="mnist_model"
MODEL_SAVE_PATH='./model/'
def test(mnist):
    x= tf.placeholder(dtype=tf.float32,shape=[None,784])
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    y =forward.forward(x,None)

    ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
    ema_restore = ema.variables_to_restore()
    saver = tf.train.Saver(ema_restore)

    corret_pre = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy =tf.reduce_mean(tf.cast(corret_pre,tf.float32))
    #saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            for i in range(STEPS):
                #X,Y_ = mnist.train.next_batch(batch_size=BATCH_SIZE)
                accuracy_v=sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
                print(accuracy_v)
                #l1,l2 ,accuracy_v = sess.run([tf.argmax(y,1),tf.argmax(y_,1),accuracy], feed_dict={x: X, y_: Y_})
                #print(l1)
                #print(l2)
                #print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_v))
        else:
            print('No checkpoint file found')
            return
if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    test(mnist)