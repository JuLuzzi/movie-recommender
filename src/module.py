import pandas as pd
import tensorflow as tf
import matplotlib
from sklearn.model_selection import train_test_split as ts
from sklearn.metrics import mean_squared_error as me
matplotlib.use('TkAgg')


def start():
    print 'reading dataset'
    ratings = pd.read_csv('dataset/ratings.dat', sep="::", header=None, engine='python')

    ratings_pivot = pd.pivot_table(ratings[[0, 1, 2]], values=2, index=0, columns=1).fillna(0)

    train, test = ts(ratings_pivot, train_size=0.8)

    # how many nodes
    nodes_in = 3706
    nodes_hidden = 256
    nodes_out = 3706

    hidden_layer = {'weights': tf.Variable(tf.random_normal([nodes_in + 1, nodes_hidden]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([nodes_hidden + 1, nodes_out]))}
    input_layer = tf.placeholder('float', [None, 3706])

    input_layer_const = tf.fill([tf.shape(input_layer)[0], 1], 1.0)
    input_layer_concat = tf.concat([input_layer, input_layer_const], 1)

    # multiply output of input_layer wth a weight matrix
    layer_1 = tf.nn.sigmoid(tf.matmul(input_layer_concat, hidden_layer['weights']))
    layer1_const = tf.fill([tf.shape(layer_1)[0], 1], 1.0)
    layer_concat = tf.concat([layer_1, layer1_const], 1)

    # multiply output of hidden with a weight matrix to get final output
    output_layer = tf.matmul(layer_concat, output_layer['weights'])

    output = tf.placeholder('float', [None, 3706])

    cost_function = tf.reduce_mean(tf.square(output_layer - output))

    optimizer = 0.1
    optimizer = tf.train.AdagradOptimizer(optimizer).minimize(cost_function)

    init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)

    batch_size = 100
    epochs = 200
    images = train.shape[0]

    training(batch_size, cost_function, epochs, images, input_layer, optimizer, output, output_layer, session, test,
             train)

    user = test.iloc[99, :]

    pred = session.run(output_layer, feed_dict={input_layer: [user]})
    # TODO guardar las predicciones para no entrenarla cada vez
    print pred


def training(batch_size, cost_function, epochs, images, input_layer, optimizer, output, output_layer, session, test,
             train):
    for epoch in range(epochs):
        error = 0

        for i in range(int(images / batch_size)):
            epoch = train[i * batch_size: (i + 1) * batch_size]
            _, c = session.run([optimizer, cost_function], feed_dict={input_layer: epoch, output: epoch})
            error += c

        output_train = session.run(output_layer, feed_dict={input_layer: train})
        output_test = session.run(output_layer, feed_dict={input_layer: test})

        print('Train', me(output_train, train), 'test', me(output_test, test))
        print('Epoch', epoch, '/', epochs, 'error:', error)


if __name__ == "__main__":
    # user = raw_input("Enter user id: ")
    # start(user)
    start()
