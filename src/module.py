import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split as ts
from sklearn.metrics import mean_squared_error as me


def start():
    print 'reading dataset'
    ratings = pd.read_csv('dataset/ratings.csv', header=1, engine='python')

    ratings = pd.pivot_table(ratings[[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]], values=2, index=0, columns=1)\
        .fillna(0)

    train, test = ts(ratings, train_size=0.8)

    # how many nodes
    nodes_in = 4000
    nodes_hidden = 256
    nodes_outl = 4000

    hidden_layer = {'weights': tf.Variable(tf.random_normal([nodes_in + 1, nodes_hidden]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([nodes_hidden + 1, nodes_outl]))}
    input_layer = tf.placeholder('float', [None, 4000])

    input_layer_const = tf.fill([tf.shape(input_layer)[0], 1], 1.0)
    input_layer_concat = tf.concat([input_layer, input_layer_const], 1)

    # multiply output of input_layer wth a weight matrix
    layer_1 = tf.nn.sigmoid(tf.matmul(input_layer_concat, hidden_layer['weights']))
    layer1_const = tf.fill([tf.shape(layer_1)[0], 1], 1.0)
    layer_concat = tf.concat([layer_1, layer1_const], 1)

    # multiply output of hidden_layer with a weight matrix
    output_layer = tf.matmul(layer_concat, output_layer['weights'])

    output = tf.placeholder('float', [None, 4000])

    cost_function = tf.reduce_mean(tf.square(output_layer - output))

    # how fast should learn
    learn_rate = 0.1
    optimizer = tf.train.AdagradOptimizer(learn_rate).minimize(cost_function)

    init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)

    batch_size = 100
    epochs = 200
    images = train.shape[0]

    training(batch_size, epochs, images, input_layer, cost_function, optimizer, output_layer, output, session, test,
             train)

    user = test.iloc[99, :]

    pred = session.run(output_layer, feed_dict={input_layer: [user]})

    # TODO guardar las predicciones
    print pred


def training(batch_size, epochs, tot_images, input_layer, cost_function, optimizer, output_layer, output_true, sess,
             test, train):
    for epoch in range(epochs):
        error = 0

        for i in range(int(tot_images / batch_size)):
            epoch = train[i * batch_size: (i + 1) * batch_size]
            _, c = sess.run([optimizer, cost_function], feed_dict={input_layer: epoch, output_true: epoch})
            error += c

        output_train = sess.run(output_layer, feed_dict={input_layer: train})
        output_test = sess.run(output_layer, feed_dict={input_layer: test})

        print('Train ', me(output_train, train), ' test', me(output_test, test))
        print('Epoch ', epoch, '/', epochs, ' loss:', error)


if __name__ == "__main__":
    # user = raw_input("Enter user id: ")
    # start(user)
    start()
