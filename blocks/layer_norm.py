
import tensorflow as tf

def layer_norm(input_tensor, name=None):
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)
