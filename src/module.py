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
    nodes_inpl = 4000
    nodes_hl1 = 256
    nodes_outl = 4000

    hidden_layer = {'weights': tf.Variable(tf.random_normal([nodes_inpl + 1, nodes_hl1]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([nodes_hl1 + 1, nodes_outl]))}
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
    sess = tf.Session()
    sess.run(init)

    batch_size = 100
    hm_epochs = 200
    images = train.shape[0]

    training(batch_size, hm_epochs, images, input_layer, cost_function, optimizer, output_layer, output, sess, test,
             train)

    # TODO pasarlo por parametro
    user = test.iloc[99, :]

    user_pred = sess.run(output_layer, feed_dict={input_layer: [user]})
    # TODO devolver las top 5 predicciones
    print user_pred


def training(batch_size, hm_epochs, tot_images, input_layer, meansq, optimizer, output_layer, output_true, sess, test,
             train):
    for epoch in range(hm_epochs):
        error = 0

        for i in range(int(tot_images / batch_size)):
            epoch = train[i * batch_size: (i + 1) * batch_size]
            _, c = sess.run([optimizer, meansq], feed_dict={input_layer: epoch, output_true: epoch})
            error += c

        output_train = sess.run(output_layer, feed_dict={input_layer: train})
        output_test = sess.run(output_layer, feed_dict={input_layer: test})

        print('Train ', me(output_train, train), ' test', me(output_test, test))
        print('Epoch ', epoch, '/', hm_epochs, ' loss:', error)


if __name__ == "__main__":
    # user = raw_input("Enter user id: ")
    # start(user)
    start()
