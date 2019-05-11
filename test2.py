# !usr/bin/python
# Author:das
# -*-coding: utf-8 -*-
import tensorflow as tf
import forward
import backward
BATCH_SIZE =10
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

    loss = tf.losses.mean_squared_error(labels=y_,predictions=y)
    loss = loss + tf.add_n(tf.get_collection('losses'))
    #saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            for i in range(STEPS):
                X,Y_ = mnist.test.next_batch(batch_size=BATCH_SIZE)
                #accuracy_v=sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
                #print(sess.run(y))
                loss_v= sess.run(loss, feed_dict={x: X, y_: Y_})
                print(loss_v)
                #print(l2)
                #print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_v))
        else:
            print('No checkpoint file found')
            return
if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    test(mnist)