import tensorflow as tf
from convmnist import createWeight, createBias, createConv, createPool
import numpy as np
import time
import models.tutorials.image.cifar10.cifar10 
import models.tutorials.image.cifar10.cifar10_input

MAX_STEPS = 3000
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
DATA_DIR = "./cifar10_data/cifar-10-batches-bin"

def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, tf.float32, stddev = stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name = "weight_loss")
        tf.add_to_collection('losses', weight_loss)
    return var

def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits = logits, labels = labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'))

with tf.name_scope("placeholders"):
    image_holder = tf.placeholder(tf.float32, [BATCH_SIZE, 24, 24, 3])
    label_holder = tf.placeholder(tf.float32, [BATCH_SIZE])

with tf.name_scope("first_conv"):
    weight1 = variable_with_weight_loss([5, 5, 3, 64], stddev = 5e-2, wl = 0.0)
    kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], 'SAME')
    bias1 = tf.Variable(tf.constant(0.0, shape = [64]))
    conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
    pool1 = tf.nn.max_pool(conv1, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
    norm1 = tf.nn.lrn(pool1, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)

with tf.name_scope("second_conv"):
    weight2 = variable_with_weight_loss([5, 5, 64, 64], stddev = 5e-2, wl = 0.0)
    kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], 'SAME')
    bias2 = tf.Variable(tf.constant(0.1, [64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
    norm2 = tf.nn.lrn(conv2, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)
    pool2 = tf.nn.max_pool(norm2, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')

with tf.name_scope("first_fully_connected"):
    reshape = tf.reshape(pool2, [BATCH_SIZE, -1])
    dim = reshape.get_shape()[1].value
    weight3 = variable_with_weight_loss([dim, 384], stddev = 0.04, wl = 0.004)
    bias3 = tf.Variable(tf.constant(0.1, shape = [384]))
    local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

with tf.name_scope("second_fully_connected"):
    weight4 = variable_with_weight_loss([384, 192], stddev = 0.04, wl = 0.004)
    bias4 = tf.Variable(tf.constant(0.1, shape = [192]))
    local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

with tf.name_scope("third_fully_connected"):
    weight5 = variable_with_weight_loss([192, 10], stddev = 1/192.0, wl = 0.0)
    bias5 = tf.Variable(tf.constant(0.0, shape = [10]))
    logits = tf.add(tf.matmul(local4, weight5) + bias5)

with tf.name_scope("loss"):
    loss = loss(logits, label_holder)

with tf.name_scope("trianer"):
    trainer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

if __name__ == '__main__':
    cifar_10.maybe_download_extract()
    images_train, labels_train = cifar10_input.distorted_inputs(
        data_dir = DATA_DIR, batch_size = BATCH_SIZE)
    images_test, labels_test = cifar10_input.inputs(eval_data = True,
                                                    data_dir = DATA_DIR,
                                                    batch_size = BATCH_SIZE)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer.run()
    tf.train.start_queue_runners()
    writer = tf.summary.FileWriter("./logs/convcifar10/", sess.graph)
    for step in range(MAX_STEPS):
        start_time = time.time()
        image_batch, label_batch = sess.run([images_train, labels_train])
        _, loss_value = sess.run(trainer, loss,
            feed_dict = {image_holder: image_batch, label_holder: label_batch})
        duration = time.time() - start_time
        if step % 10 == 0:
            examples_per_sec = BATCH_SIZE / duration
            sec_per_batch = float(duration)

            print("loss: %f, examples/s: %f, s/batch:%f" \
                % (loss_value, examples_per_sec, sec_per_batch))
    
    num_examples = 10000
    import math
    num_iter = int(math.ceil(num_examples / BATCH_SIZE))
    true_count = 0
    total_sample_count = num_iter * BATCH_SIZE
    step = 0
    while step < num_iter:
        image_batch, label_batch = sess.run(images_test, labels_test)
        predictions = sess.run(top_k_op, 
            feed_dict = {image_holder: image_batch, label_holder: label_batch})
        true_count = np.sum(predictions)
        step += 1
    
    precision = true_count / total_sample_count
    print("presition @ 1 = %.3f" % precision)


