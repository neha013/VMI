import tensorflow as tf
import tensorflow.keras.backend as K
# Add attention layer to the deep learning network
class attention((tf.keras.layers.Layer)):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                              #  initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=123),
                              #  initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=123),
                                  # initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=1.,seed=123),
                                #  initializer = tf.keras.initializers.GlorotNormal(seed=123),
                                #  initializer = tf.keras.initializers.GlorotUniform(seed=123),
                                #  initializer = tf.keras.initializers.Orthogonal(seed=123),
                                initializer = tf.keras.initializers.HeNormal(seed=123),
                                # initializer = tf.keras.initializers.HeUniform(seed=123),
                                # initializer = tf.keras.initializers.Ones(seed=123),
                                # initializer = tf.keras.initializers.LecunNormal(seed=123),
                                # initializer = tf.keras.initializers.LecunUniform(seed=123),
                                dtype='float32',
                                 trainable=True)
        
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(attention, self).build(input_shape)
 
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context
		
	