
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

def position_embedding(input_tensor, max_position_length=100, embedding_size=200):

    with tf.variable_scope('position_embedding', reuse=tf.AUTO_REUSE):
        # T
        seq_len = int(input_tensor.shape[-2])

        # N
        batch_size = tf.shape(input_tensor)[0]

        # T_max, E
        embedding = tf.get_variable(name='position_embedding', shape=(max_position_length, embedding_size), initializer=xavier_initializer())

        # T, E
        position_embedding = tf.slice(embedding, [0,0], [seq_len, -1])

        # 1, T, E
        position_embedding = tf.expand_dims(position_embedding, 0)

        # N, T, E
        tiled_position_embedding = tf.tile(position_embedding, [batch_size, 1, 1])

    return tiled_position_embedding

