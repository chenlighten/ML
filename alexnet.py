from datetime import datetime
import math
import time
import tensorflow as tf

batch_size = 32
num_batches = 100
parameters = []

def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

def conv_layer(input, scope, kernel_shape, conv_strides, pool_ksize=None, pool_strides=None, do_pool=True):
    with tf.name_scope(scope):
        kernel = tf.Variable(tf.truncated_normal(kernel_shape, 
            dtype = tf.float32, stddev = 1e-1), name = 'weights')
        conv = tf.nn.conv2d(input, kernel, conv_strides, 'SAME')
        biases = tf.Variable(tf.constant(0.0, shape = [kernel_shape[3]], dtype = tf.float32), 
            name = 'biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name = scope)
        print_activations(conv1)
        parameters.append(kernel)
        parameters.append(biases)

        if do_pool:
            lrn1 = tf.nn.lrn(conv1, 4, bias = 1.0, alpha = 0.001 / 9, 
                beta = 0.75, name = 'lrn')
            pool1 = tf.nn.max_pool(lrn1, pool_ksize, pool_strides, 'VALID', name = 'pool')
            print_activations(pool1)
            return pool1
        else:
            return conv1

images = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])

random_images = tf.Variable(tf.random_normal(shape = [batch_size, 224, 224, 3], 
            dtype=tf.float32, stddev=0.1))

pool1 = conv_layer(random_images, 'conv1', [11, 11, 3, 64], [1, 4, 4, 1],
    [1, 3, 3, 1], [1, 2, 2, 1], True)

pool2 = conv_layer(pool1, 'conv2', [5, 5, 64, 192], [1, 1, 1, 1],
    [1, 3, 3, 1], [1, 2, 2, 1], True)

conv3 = conv_layer(pool2, 'conv3', [3, 3, 192, 384], [1, 1, 1, 1], 
    do_pool=False)

conv4 = conv_layer(conv3, 'conv4', [3, 3, 384, 256], [1, 1, 1, 1], 
    do_pool=False)

conv5 = conv_layer(conv4, 'conv5', [3, 3, 256, 256], [1, 1, 1, 1], 
    do_pool=False)

pool5 = tf.nn.max_pool(conv5, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID', name='pool5')
print_activations(pool5)

def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_steps_burn_in + num_batches):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_batches:
            if not i % 10:
                print('%s: step %d, duration = %.3f' % 
                    (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration*duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn*mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %f.3f sec / batch' %
        (datetime.now(), info_string, num_batches, mn, sd))

if __name__ == '__main__':
    # with tf.Graph().as_default():
        image_size = 224
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        time_tensorflow_run(sess, conv5, "Forward")
        objective = tf.nn.l2_loss(conv5)
        grad = tf.gradients(objective, parameters)
        time_tensorflow_run(sess, grad, "Forward_backward")

