# Python 3.6.0
# tensorflow 1.1.0

import os
import os.path as path

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

from tensorflow.examples.tutorials.mnist import input_data

MODEL_NAME = 'mnist_convnet'
NUM_STEPS = 3000
BATCH_SIZE = 16

def model_input(input_node_name, keep_prob_node_name):
    x = tf.placeholder(tf.float32, shape=[None, 28*28], name=input_node_name)
    keep_prob = tf.placeholder(tf.float32, name=keep_prob_node_name)
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    return x, keep_prob, y_

def build_model(x, keep_prob, y_, output_node_name):
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    # 28*28*1

    conv1 = tf.layers.conv2d(x_image, 64, 3, 1, 'same', activation=tf.nn.relu)
    # 28*28*64
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2, 'same')
    # 14*14*64

    conv2 = tf.layers.conv2d(pool1, 128, 3, 1, 'same', activation=tf.nn.relu)
    # 14*14*128
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2, 'same')
    # 7*7*128

    conv3 = tf.layers.conv2d(pool2, 256, 3, 1, 'same', activation=tf.nn.relu)
    # 7*7*256
    pool3 = tf.layers.max_pooling2d(conv3, 2, 2, 'same')
    # 4*4*256

    flatten = tf.reshape(pool3, [-1, 4*4*256])
    fc = tf.layers.dense(flatten, 1024, activation=tf.nn.relu)
    dropout = tf.nn.dropout(fc, keep_prob)
    logits = tf.layers.dense(dropout, 10)
    outputs = tf.nn.softmax(logits, name=output_node_name)

    # loss
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))

    # train step
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # accuracy
    correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    merged_summary_op = tf.summary.merge_all()

    return train_step, loss, accuracy, merged_summary_op

def train(x, keep_prob, y_, train_step, loss, accuracy,
        merged_summary_op, saver):
    print("training start...")

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        tf.train.write_graph(sess.graph_def, 'out',
            MODEL_NAME + '.pbtxt', True)

        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter('logs/',
            graph=tf.get_default_graph())

        for step in range(NUM_STEPS):
            batch = mnist.train.next_batch(BATCH_SIZE)
            if step % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %f' % (step, train_accuracy))
            _, summary = sess.run([train_step, merged_summary_op],
                feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            summary_writer.add_summary(summary, step)

        saver.save(sess, 'out/' + MODEL_NAME + '.chkp')

        test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels,
                                    keep_prob: 1.0})
        print('test accuracy %g' % test_accuracy)

    print("training finished!")

def export_model(input_node_names, output_node_name):
    freeze_graph.freeze_graph('out/' + MODEL_NAME + '.pbtxt', None, False,
        'out/' + MODEL_NAME + '.chkp', output_node_name, "save/restore_all",
        "save/Const:0", 'out/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")

def main():
    if not path.exists('out'):
        os.mkdir('out')

    input_node_name = 'input'
    keep_prob_node_name = 'keep_prob'
    output_node_name = 'output'

    x, keep_prob, y_ = model_input(input_node_name, keep_prob_node_name)

    train_step, loss, accuracy, merged_summary_op = build_model(x, keep_prob,
        y_, output_node_name)
    saver = tf.train.Saver()

    train(x, keep_prob, y_, train_step, loss, accuracy,
        merged_summary_op, saver)

    export_model([input_node_name, keep_prob_node_name], output_node_name)

if __name__ == '__main__':
    main()
