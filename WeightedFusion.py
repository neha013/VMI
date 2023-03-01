""" """

import tensorflow as tf
class WeightedFusion(tf.keras.layers.Layer):
    """A custom keras layer to learn a weighted sum of tensors"""

    def __init__(self, **kwargs):
        super(WeightedFusion, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(name='w',
                                 shape=(1,5),
                                # initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=123),
                                # initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=123),
                                #  initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=1.,seed=123),
                                #  initializer = tf.keras.initializers.GlorotNormal(seed=123),
                                #  initializer = tf.keras.initializers.GlorotUniform(seed=123),
                                #  initializer = tf.keras.initializers.Orthogonal(seed=123),
                                initializer = tf.keras.initializers.HeNormal(seed=123),
                                # initializer = tf.keras.initializers.HeUniform(seed=123),
                                # initializer = tf.keras.initializers.Ones(seed=123),
                                # initializer = tf.keras.initializers.LecunNormal(seed=123),
                                # initializer = tf.keras.initializers.LecunUniform(seed=123),
                                 dtype='float32',
                                 trainable=True,
                                 constraint=tf.keras.constraints.min_max_norm(
                                     max_value=1, min_value=0))      
                           
        super(WeightedFusion, self).build(input_shape)

    def call(self, model_outputs):
        w1=(self.w)[0][0]
        w2=(self.w)[0][1]
        w3=(self.w)[0][2]
        w4=(self.w)[0][3]
        
        return tf.keras.activations.relu(w1 * model_outputs[0] + w2 * model_outputs[1] + w3 * model_outputs[2])
        # return tf.keras.activations.relu(1 * model_outputs[0] + 1 * model_outputs[1] + 1 * model_outputs[2])
    def get_weights(self):
        return [self.w[0][0], self.w[0][1], self.w[0][2]] 
    # def get_config(self):
       
    #     config = super(WeightedFusion,self).get_config().copy()
    #     return config
                           
        