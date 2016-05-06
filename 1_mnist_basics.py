import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Images of 28x28 pixels, calculate the total of pixels
size = 28
pixels = size**2

# Number of possible results. In this case form 0 to 9.
number_of_results = 10


# Create placeholder to store MNIST data
x = tf.placeholder(tf.float32, [None, pixels])


########################
# Regression algorythm #
########################
# Implement regression as y = Wx + b

# We use TF variables in order to move the calcs into TF instead of Python
W = tf.Variable(tf.zeros([pixels, number_of_results]))
b = tf.Variable(tf.zeros([number_of_results]))

y = tf.nn.softmax(tf.matmul(x, W) + b)


####################
# Learning process #
####################

y_ = tf.placeholder(tf.float32, [None, number_of_results])

# Use cross_entropy to teach the model what is a bad result
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# Tell the model to minimize the bad results obtained from cross_entropy process
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# Init process
init = tf.initialize_all_variables()

# Launch the model
sess = tf.Session()
sess.run(init)

# Execute the training in batches of 100 samples
print "Starting model training..."
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

print " -> Done!"

# Do the tests
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print "Testing test samples in 10 groups of 1000 samples..."
for i in range(10):
    print " - Accuracy level of group ", i
    batch_x = mnist.test.images[i*1000:(i+1)*1000]
    batch_y_ = mnist.test.labels[i*1000:(i+1)*1000]

    # Execute test and print the result
    print "    ", sess.run(accuracy, feed_dict={x: batch_x, y_: batch_y_})

print " -> Done!"

print "Testing all data at a time..."
print "Global accuracy level: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
