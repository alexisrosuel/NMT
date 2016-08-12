# -*- coding: utf-8 -*-

"""
Neural Machine Translation model
This file describes the graph used to generate a translation in the target language given a source sentence, with the loss computation and the optimizer associated 
"""

import tensorflow as tf
import math 

EMBEDDINGS_DIMENSION = 400 # Dimension of the embeddings
LSTM_SIZE = 15 # Size of the RNN (using LSTM cells)
LSTM_LAYER = 3 # Number of RNN layers (using LSTM cells)

T_ENGLISH = 0 # Updated at the beginning of the inference
S_ENGLISH = 0 # Updated at the beginning of the inference
T_FRENCH = 0 # Updated at the beginning of the inference
S_FRENCH = 0 # Updated at the beginning of the inference

BATCH_SIZE = 0 # Updated at the beginning of the inference

LEARNING_RATE = 0.01 # Initial learning rate

def inference(placeholders, corpus_params, batch_size):
    """ 
    Compute the predicted translation of the list of source sentences proposed 
    Reminder: translation from ENGLISH to FRENCH 
    
    Args:
        placeholders: list of placeholders (type is tf.float32, tf.float32, tf.bool)
        corpus_params: list of int
        batch_size: int
        
    Returns:
        4D tensor, shape=(BATCH_SIZE, 1, S_FRENCH, T_FRENCH)
    """
    
    source, target, training = placeholders
    
    global T_ENGLISH, S_ENGLISH, T_FRENCH, S_FRENCH
    T_ENGLISH, S_ENGLISH, T_FRENCH, S_FRENCH= corpus_params
    
    global BATCH_SIZE
    BATCH_SIZE = batch_size
    
    state_vector = encode(source) # Build the state_vector for each source sentence
    prediction = decode(state_vector, target, training) # Produce a translation given a state_vector
    return prediction


def produce_embeddings(source):
    """ Produce the embbedings from the one-hot vectors 
    
    Args:
        source: 4D tensor, shape=(BATCH_SIZE, 1, S_ENGLISH, T_ENGLISH)
    
    Returns:
        4D tensor, shape=(BATCH_SIZE, 1, S_ENGLISH, EMBEDDINGS_DIMENSION)
    """
    
    with tf.variable_scope('encode_dense'):
        weights = tf.get_variable(name='weights', 
                                  shape=[1,1,T_ENGLISH,EMBEDDINGS_DIMENSION], 
                                  initializer=tf.random_normal_initializer(stddev=1.0/math.sqrt(float(T_ENGLISH)))
                                  )
        
        weights_hist = tf.histogram_summary("weights-encode", weights)
        
        biases = tf.get_variable(name = 'biases',
                                 shape = [EMBEDDINGS_DIMENSION],          
                                 initializer = tf.constant_initializer(0.0))
                                 
        biases_hist = tf.histogram_summary("biases-encode", biases)
        
        embeddings = tf.nn.tanh(biases + tf.nn.conv2d(source, weights, strides = [1,1,1,1], padding='VALID'))
        
        return embeddings                
        

def embeddings_to_state_vector(embeddings):
    """ Use a RNN to produce the state vector corresponding to each sentence 
    
    Args:
        embeddings: 4D tensor, shape=(BATCH_SIZE, 1, S_ENGLISH, EMBEDDINGS_DIMENSION)
    
    Returns:
        2D tensor, shape=(BATCH_SIZE, LSTM_SIZE)
    """

    with tf.variable_scope('forward') as scope_forward:
        # remove the dimensions 1
        inputs = tf.squeeze(embeddings)
        # Permuting batch_size and n_steps
        inputs = tf.transpose(inputs,[1,0,2])
        # Reshaping to (n_steps*batch_size, n_input)
        inputs = tf.reshape(inputs, [-1, EMBEDDINGS_DIMENSION])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        inputs = tf.split(0, S_ENGLISH, inputs)

        cell_forward = tf.nn.rnn_cell.LSTMCell(LSTM_SIZE, num_proj = EMBEDDINGS_DIMENSION, state_is_tuple = True)
        state = [tf.zeros((BATCH_SIZE, sz)) for sz in cell_forward.state_size]
        
        for t in range(S_ENGLISH):
            _, state = cell_forward(inputs[t], state)
            scope_forward.reuse_variables()

        return state

def encode(source):
    """ Build the state vector from the source sentence (using one-hot representation) 
    
    Args:
        source: 4D tensor, shape=(BATCH_SIZE, 1, S_ENGLISH, T_ENGLISH)
    
    Returns:
        2D tensor, shape=(BATCH_SIZE, LSTM_SIZE)
    """

    embeddings = produce_embeddings(source)
    state_vector = embeddings_to_state_vector(embeddings)
    return state_vector 
    
    
def state_vector_to_probability(state_vector, target, training):
    """ Produce the list of token probability in the target language given a state_vector 
    
    Args:
        state_vector: 2D tensor, shape=(BATCH_SIZE, LSTM_SIZE)
        target: 4D tensor, shape=(BATCH_SIZE, 1, S_FRENCH, T_FRENCH)
        training: bool tensor
    
    Returns:
        3D tensor, shape=(BATCH_SIZE, S_FRENCH, T_FRENCH)
    """

    with tf.name_scope('transformation'):
        # remove the dimensions 1
        target = tf.squeeze(target)
        # Permuting batch_size and n_steps
        target = tf.transpose(target ,[1,0,2])
        # Reshaping to (n_steps*batch_size, n_input)
        target = tf.reshape(target , [-1, T_FRENCH]) 
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        target = tf.split(0, S_FRENCH, target )
        # S_FRENCH; BATCH; T_FRENCH
            
    with tf.variable_scope('backward') as scope_backward:     
        cell_backward = tf.nn.rnn_cell.LSTMCell(LSTM_SIZE, num_proj = EMBEDDINGS_DIMENSION, state_is_tuple = True)       
        outputs = [None] * S_FRENCH
        y = [None] * S_FRENCH
        state = state_vector
        
        weights = tf.get_variable(name='weights',
                                  shape = [EMBEDDINGS_DIMENSION, T_FRENCH],
                                  initializer = tf.random_normal_initializer(stddev =1.0/math.sqrt(float(T_FRENCH)))
                                  )                                       
        weights_hist = tf.histogram_summary("weights-decode", weights)
        
        biases = tf.get_variable(name='biases',
                                 shape=[T_FRENCH],
                                 initializer=tf.constant_initializer(0.0))
        biases_hist = tf.histogram_summary("biases-decode", biases)
        
        for t in range(S_FRENCH):
            if t:
                # Test one hot, a valider !  
                argmax = tf.argmax(outputs[t-1], dimension=1) # Get the token with highest prediction value for each example
                #range_tensor = tf.range(0, limit=BATCH_SIZE) # Build the indices list
                #indices = tf.pack([tf.cast(range_tensor, dtype=tf.int32), tf.cast(argmax, dtype=tf.int32)], axis=1)
                
                one_hot_lstm_inputs = tf.cast(tf.one_hot(argmax, depth=T_FRENCH, on_value=1, off_value=0), dtype=tf.float32) # Create the new one-hot vector
                
                
                """ If in training, feed with the target sentence, if not, feed with the choosen token (ie. here the one with the highest probability) """
                last = tf.cond(training, lambda: target[t-1], lambda: one_hot_lstm_inputs) 
                #last = tf.cond(training, lambda: target[t-1], lambda: outputs[t-1])
                #last = sortie_dense[t - 1] if training else outputs[t - 1]
            else:
                last = tf.zeros((BATCH_SIZE, T_FRENCH))

            y[t], state = cell_backward(last, state)
    
            outputs[t] = tf.nn.softmax(tf.matmul(y[t], weights) + biases)
            scope_backward.reuse_variables()
        
        return tf.transpose(outputs, [1,0,2])
    
    
def decode(state_vector, target, training):
    """ Produce the probability of appearance for each token 
    
    Args:
        state_vector: 2D tensor, shape=(BATCH_SIZE, LSTM_SIZE)
        target: 4D tensor, shape=(BATCH_SIZE, 1, S_FRENCH, T_FRENCH)
        training: bool tensor
    
    Returns:
        3D tensor, shape=(BATCH_SIZE, S_FRENCH, T_FRENCH)
    """
    
    outputs = state_vector_to_probability(state_vector, target, training)
    return outputs

def conpute_loss(scores, target):
    """ Compute the perplexity of the batch 
    
    Args:
        scores: 3D tensor, shape=(BATCH_SIZE, 1, S_FRENCH, T_FRENCH)
        target: 4D tensor, shape=(BATCH_SIZE, 1, S_FRENCH, T_FRENCH)
        
    Returns:
        tf.float32 tensor
    """
    
    with tf.name_scope('loss_computation'):
        sortie_loss = tf.squeeze(target)    
        scores = tf.squeeze(scores) 
        
        loss = tf.mul(scores, sortie_loss)
        loss = tf.reduce_sum(loss,reduction_indices=2)
        loss = tf.clip_by_value(loss, clip_value_min=1e-10, clip_value_max=1.0)
        loss = tf.log(loss)
        loss = tf.reduce_sum(loss, reduction_indices=1)
        loss = tf.reduce_mean(loss)
        
        l2_weights = 0.01
        with tf.variable_scope('encode_dense', reuse=True):
            w = tf.get_variable('weights')
            b = tf.get_variable('biases')
            loss -= l2_weights *tf.nn.l2_loss(w)
            loss -= l2_weights *tf.nn.l2_loss(b)
            
        with tf.variable_scope('backward', reuse=True):
            w = tf.get_variable('weights')
            b = tf.get_variable('biases')
            loss -= l2_weights *tf.nn.l2_loss(w)
            loss -= l2_weights *tf.nn.l2_loss(b)
        return -loss
    
    
def training(loss):
    """ Produce the training operator, using the Adam Optimizer 
    
    Args:
        loss: tf.float32 tensor
        
    Returns:
        training operator
    """
    
    tf.scalar_summary(loss.op.name,loss)
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)    
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op=optimizer.minimize(loss,global_step=global_step)
    return train_op
