import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K



class Graph_Convolution(tf.keras.layers.Layer):
    """A custom keras layer to compute Graph Convolution"""

    def __init__(self, **kwargs):
        super(Graph_Convolution, self).__init__(**kwargs)

    def build(self, input_shape):
     
        self.w = tf.keras.layers.Dense(512, kernel_initializer = tf.keras.initializers.HeNormal(
    seed=123))
        self.num_heads = 8
        self.d_model = 512
        
    
        self.wq = tf.keras.layers.Dense(512, kernel_initializer = tf.keras.initializers.HeNormal(
    seed=123))
        self.wk = tf.keras.layers.Dense(512, kernel_initializer = tf.keras.initializers.HeNormal(
    seed=123))
        self.wv = tf.keras.layers.Dense(512, kernel_initializer = tf.keras.initializers.HeNormal(
    seed=123
))
  

        self.dense = tf.keras.layers.Dense(14, kernel_initializer =tf.keras.initializers.HeNormal(
    seed=123
))

        self.depth = self.d_model // self.num_heads
                                
        super(Graph_Convolution, self).build(input_shape)

    def call(self, X):
       
        #siNGLE-head attention    
        # K = self.wq(X)
        # Q = self.wk(X)
        # KQ = tf.matmul(Q[-1,:,:], tf.transpose(K[-1,:,:]) )/32
        # A = tf.keras.activations.softmax(KQ)

        #Multi-head attention
        k=q=v=X
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        
        q = self.split_heads(q)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v)  # (batch_size, num_heads, seq_len_v, depth)
       
         # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
         # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2,1,3])  # (batch_size, seq_len_q, num_heads, depth)
  
        s = tf.shape(q)[1]
      
        concat_attention = tf.reshape(scaled_attention, 
                                  ( -1, s, self.d_model))  # (batch_size, seq_len_q, d_model)
        
        output = self.dense(concat_attention)
        
        A = tf.keras.activations.sigmoid(output[-1,:,:])
    
        
        I = np.matrix(np.eye(A.shape[0]))
        
        A_hat = tf.math.add(A,I) #Adding Self loop
        
   #Inverse Degree Matrix (For Normalization)
        D_hat = tf.math.pow(tf.math.reduce_sum(A, axis=0), -0.5)
       
        D_hat = tf.linalg.diag(D_hat)
    #Weight Matrix
        
        p1= tf.matmul(D_hat, A_hat)
      
        p2 = tf.matmul(p1, D_hat)
     
        p3 = tf.matmul(p2, X)
       
        # self.w = K.print_tensor(self.w, message="w")
      
        # X = tf.matmul(p3, self.w) 
        Y = self.w(p3)
        
        
        # return tf.keras.activations.relu(X)
        # return tf.nn.leaky_relu(X)
        # return tf.keras.activations.elu(X)
        # return tf.keras.activations.selu(X)
        return tf.keras.activations.gelu(Y)
       
    # def compute_output_shape(self, input_shape):
    #     return (input_shape[0], self.output_dim)
      
    def split_heads(self, x):
    
        s = tf.shape(x)[1]
        x = tf.reshape(x, (-1, s, self.num_heads, self.depth))
        return x

    def scaled_dot_product_attention(self, q, k, v):


        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights 
