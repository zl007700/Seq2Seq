
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

def multi_head_attention(Q, K, V, num_head, key_mask=None, query_mask=None, mask_future=False):
    # Q: N, T_q, d_model
    # K: N, T_k, d_model
    # V: N, T_k, d_model

    d_model = int(Q.shape[-1])
    d_k = d_v = int(d_model / num_head)

    W_Q =  tf.get_variable(name='weight_q', shape=(d_model, d_k), dtype=tf.float32, initializer=xavier_initializer())
    W_K =  tf.get_variable(name='weight_k', shape=(d_model, d_k), dtype=tf.float32, initializer=xavier_initializer())
    W_V =  tf.get_variable(name='weight_v', shape=(d_model, d_v), dtype=tf.float32, initializer=xavier_initializer())

    tiled_Q = tf.tile(Q, [num_head, 1, 1])
    tiled_K = tf.tile(K, [num_head, 1, 1])
    tiled_V = tf.tile(V, [num_head, 1, 1])
    tiled_key_mask = tf.tile(key_mask, [num_head, 1])
    tiled_query_mask = tf.tile(query_mask, [num_head, 1])

    heads = scale_dot_product_attention(Q=tf.einsum('ijk,kl->ijl', tiled_Q, W_Q),
                                        K=tf.einsum('ijk,kl->ijl', tiled_K, W_K),
                                        V=tf.einsum('ijk,kl->ijl', tiled_V, W_V),
                                        key_mask=tiled_key_mask,
                                        query_mask=tiled_query_mask,
                                        mask_future=mask_future)

    attention = tf.concat(tf.split(heads, num_head, 0) ,axis=-1)
    return attention


def scale_dot_product_attention(Q, K, V, key_mask=None, query_mask=None ,mask_future=False):
    # Q: N*head, T, E/head, query_matrix
    # K: N*head, T, E/head, key_matrix
    # V: N*head, T, E/head, value_matrix

    d_k = int(K.shape[-1])

    # N, T_in, T_out
    attention_score = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
    attention_score /= d_k**(0.5)

    # key masks：把key中对应的Padding部分的attention_score置为负无穷,即不考虑对于padding部分的attention
    padding_digits = -100000
    if key_mask!=None:
        masks = tf.expand_dims(key_mask, 1) # (N, 1, T_k)
        masks = tf.tile(masks, [1, tf.shape(Q)[1], 1])  # (N, T_q, T_k)
        
        paddings = tf.ones_like(attention_score) * padding_digits
        masked_attention_score = tf.where(tf.equal(masks, 0), paddings, attention_score)  # (N, T_q, T_k)
        attention_score = masked_attention_score

    # query masks：把query中对应的Padding部分的attention_score设置为0，即padding部分平均考虑所有input
    if query_mask!=None:
        masks = tf.expand_dims(query_mask, 2) # (N, T_q, 1)
        masks = tf.tile(masks, [1, 1, tf.shape(K)[1]])  # (N, T_q, T_k)

        masked_attention_score = attention_score*masks
        attention_score = masked_attention_score

    # future mask: teacher forcing训练阶段只考虑t时刻之前的token的attention
    if mask_future:
        # T_in, T_out
        diag_vals = tf.ones_like(attention_score[0, :, :])
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        # N, T_in, T_out
        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(attention_score)[0], 1, 1])
        paddings = tf.ones_like(masks) * padding_digits

        masked_attention_score = tf.where(tf.equal(masks, 0), paddings, attention_score)
        attention_score = masked_attention_score

    attention_weight = tf.nn.softmax(attention_score)
    attention = tf.matmul(attention_weight, V)

    return attention

