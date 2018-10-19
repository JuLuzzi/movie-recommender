import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse


def start():
    print "module is running"

    # reading the ratings data
    ratings = pd.read_csv('dataset/ratings.csv', header=1, engine='python')

    # Lets pivot the data to get it at a user level
    ratings_pivot = pd.pivot_table(ratings[[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]], values=2, index=0, columns=1)\
        .fillna(0)

    # creating train and test sets
    x_train, x_test = train_test_split(ratings_pivot, train_size=0.8)

    # Deciding how many nodes wach layer should have
    n_nodes_inpl = 3706
    n_nodes_hl1 = 256
    n_nodes_outl = 3706

    # first hidden layer has 784*32 weights and 32 biases
    hidden_1_layer_vals = {'weights': tf.Variable(tf.random_normal([n_nodes_inpl + 1, n_nodes_hl1]))}
    # first hidden layer has 784*32 weights and 32 biases
    output_layer_vals = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1 + 1, n_nodes_outl]))}

    # user with 3706 ratings goes in
    input_layer = tf.placeholder('float', [None, 3706])
    # add a constant node to the first layer
    # it needs to have the same shape as the input layer for me to be
    # able to concatenate it later
    input_layer_const = tf.fill([tf.shape(input_layer)[0], 1], 1.0)
    input_layer_concat = tf.concat([input_layer, input_layer_const], 1)
    # multiply output of input_layer wth a weight matrix
    layer_1 = tf.nn.sigmoid(tf.matmul(input_layer_concat, hidden_1_layer_vals['weights']))
    # adding one bias node to the hidden layer
    layer1_const = tf.fill([tf.shape(layer_1)[0], 1], 1.0)
    layer_concat = tf.concat([layer_1, layer1_const], 1)
    # multiply output of hidden with a weight matrix to get final output
    output_layer = tf.matmul(layer_concat, output_layer_vals['weights'])
    # output_true shall have the original shape for error calculations
    output_true = tf.placeholder('float', [None, 3706])
    # define our cost function
    meansq = tf.reduce_mean(tf.square(output_layer - output_true))
    # define our optimizer
    learn_rate = 0.1  # how fast the model should learn
    optimizer = tf.train.AdagradOptimizer(learn_rate).minimize(meansq)

    # initialising variables and starting the session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # defining batch size, number of epochs and learning rate
    batch_size = 100  # how many images to use together for training
    hm_epochs = 200  # how many times to go through the entire dataset
    tot_images = x_train.shape[0]  # total number of images

    # running the model for a 200 epochs taking 100 users in batches
    # total improvement is printed out after each epoch
    training(batch_size, hm_epochs, tot_images, input_layer, meansq, optimizer, output_layer, output_true, sess, x_test, x_train)

    # pick a user
    sample_user = x_test.iloc[99, :]
    # get the predicted ratings
    sample_user_pred = sess.run(output_layer, feed_dict={input_layer: [sample_user]})
    print sample_user_pred


def training(batch_size, hm_epochs, tot_images, input_layer, meansq, optimizer, output_layer, output_true, sess, x_test, x_train):
    for epoch in range(hm_epochs):
        epoch_loss = 0  # initializing error as 0

        for i in range(int(tot_images / batch_size)):
            epoch_x = x_train[i * batch_size: (i + 1) * batch_size]
            _, c = sess.run([optimizer, meansq], feed_dict={input_layer: epoch_x, output_true: epoch_x})
            epoch_loss += c

        output_train = sess.run(output_layer, feed_dict={input_layer: x_train})
        output_test = sess.run(output_layer, feed_dict={input_layer: x_test})

        print('MSE train', mse(output_train, x_train), 'MSE test', mse(output_test, x_test))
        print('Epoch', epoch, '/', hm_epochs, 'loss:', epoch_loss)


if __name__ == "__main__":
    print "Decide what to do"
    start()
