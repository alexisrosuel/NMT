# -*- coding: utf-8 -*-

"""
Neural Machine Translation model. This file load the dataset, prepare the and manage the tensorflow graph/session, and run the training/test phases
"""

import pretreatment
import treatment
import NMT

import tensorflow as tf
import numpy as np

import os
import time
import math

DIRECTORY = os.path.dirname(os.path.realpath(__file__))

N_ENGLISH = 0 # total numer of english sentences in the corpus --> updated later in compute_N function
N_FRENCH = 0 # total numer of french sentences in the corpus --> updated later in compute_N function

T_ENGLISH = 0 # size of the english dictionnary (ie. number of different token)--> updated later in compute_T function
T_FRENCH = 0 # size of the french dictionnary (ie. number of different token)--> updated later in compute_T function

S_ENGLISH = 0 # size of the english sentences (uniformized to the maximum length sentence --> updatd later in compute_S function
S_FRENCH = 0 # size of the french sentences (uniformized to the maximum length sentence --> updatd later in compute_S function

CORPUS_ENGLISH = list() # list of the english token
CORPUS_FRENCH = list() # list of the french token

BATCH_SIZE = 64 
SENTENCE_LENGTH = 1

MAX_STEP = 1000 # number of training iteration


#==============================================================================
# COMPUTATION OF THE GLOBAL VARIABLES (T_ENGLISH, T_FRENCH, S_ENGLISH, S_FRENCH, N_ENGLISH, N_FRENCH)  
#==============================================================================


def compute_T(X, language):
    """ Conpute the T_FRENCH or T_ENGLISH values """
    X_flat = [item for sublist in X for item in sublist]
    
    if language == 'english':    
        global T_ENGLISH
        T_ENGLISH = len(list(set(X_flat)))
    elif language == 'french':
        global T_FRENCH
        T_FRENCH = len(list(set(X_flat)))

def compute_S(X, language): 
    """Conpute the S_FRENCH and S_ENGLISH values """
    if language == 'english':    
        global S_ENGLISH 
        S_ENGLISH = len(X[0])
    elif language == 'french':
        global S_FRENCH 
        S_FRENCH = len(X[0])
        
def compute_N(X, language): 
    """Conpute the N_FRENCH and N_ENGLISH values """
    if language == 'english':    
        global N_ENGLISH 
        N_ENGLISH = len(X)
    elif language == 'french':
        global N_FRENCH 
        N_FRENCH = len(X)
        

#==============================================================================
# COMPUTATION OF THE GLOBAL VARIABLES (T_ENGLISH, T_FRENCH, S_ENGLISH, S_FRENCH, N_ENGLISH, N_FRENCH)  
#==============================================================================

#==============================================================================
# BATCH GENERATION
#==============================================================================

def batch(source, target):
    """ Take randomly BATCH_SIZE elements from the source sentences and the target sentences """
    indices = np.random.choice(range(source.shape[0]), size=BATCH_SIZE)
    source = np.take(source, indices, axis=0)
    target = np.take(target, indices, axis=0)
    return (source, target)
    
#==============================================================================
# BATCH GENERATION
#==============================================================================
        
#==============================================================================
# TENSORFLOW PLACEHOLDER DEFINITION
#==============================================================================
        
def placeholder_input():
    """ Define the placeholder for the source sentence, the target sentence, and the condition for executing the training phase or not """
    source_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 1, S_ENGLISH, T_ENGLISH), name='source')
    target_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 1, S_FRENCH, T_FRENCH), name='target')
    training_placeholder = tf.placeholder(tf.bool, shape=[], name='training')
    return source_placeholder, target_placeholder, training_placeholder
    
#==============================================================================
# TENSORFLOW PLACEHOLDER DEFINITION
#============================================================================== 
    
def run_prediction(data, sess, placeholders, scores):
    """ conpute the prediction of a target sentence given a source sentence """    
    X, Y = data
    source_pl, target_pl, training_pl = placeholders
    split_set = int(math.floor(0.9*len(X))) # Value for spliting the set in 90% training / 10% test 
    
    X_batch, Y_batch = batch(X[:split_set:], Y[:split_set:]) # Take a batch of sentences from the training set
    feed_dict = {source_pl: X_batch, target_pl: Y_batch, training_pl: False}
    prediction = sess.run(scores, feed_dict=feed_dict)  
    X_batch = np.squeeze(X_batch)    
    prediction = np.squeeze(prediction)
    
    source_sentences = treatment.sentences_from_one_hot(X_batch, CORPUS_ENGLISH)
    target_sentences = treatment.sentences_from_one_hot(prediction, CORPUS_FRENCH)

	# Print the pairs source -> target
    for source, target in zip(source_sentences, target_sentences): 
        print(source + " --> " + target)
    
    print("================================================================")
   
    X_batch, Y_batch = batch(X[split_set+1:], Y[split_set+1:]) # Take a batch of sentences from the test set
    feed_dict = {source_pl: X_batch, target_pl: Y_batch, training_pl: False}
    prediction = sess.run(scores, feed_dict=feed_dict)  
    
    X_batch = np.squeeze(X_batch)    
    prediction = np.squeeze(prediction)
        
    
    source_sentences = treatment.sentences_from_one_hot(X_batch, CORPUS_ENGLISH)
    target_sentences = treatment.sentences_from_one_hot(prediction, CORPUS_FRENCH)

	# Print the pairs source -> target
    for source, target in zip(source_sentences, target_sentences): 
        print(source + " --> " + target)
 
 

def run_training(data, sess, placeholders, scores, loss, train_op, summary_op, summary_writer):        
    """ Training phase: for each generated batch, apply the training operator"""

    X, Y = data
    source_pl, target_pl, training_pl = placeholders

    start_time = time.time()
    divide_set = int(math.floor(0.9*len(X))) # Value for spliting the set in 90% training / 10% test 
    
    for step in range(MAX_STEP):
        X_batch, Y_batch = batch(X[:divide_set], Y[:divide_set]) # Take a batch from the training set
        feed_dict = {source_pl: X_batch, target_pl: Y_batch, training_pl: True}
        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
        
        if step % 5 == 0:
            duration = time.time() - start_time
            print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
            summary_str = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()
            
        if step % 10 == 0:
            """ Log runtime activity """
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        
            """ Evaluate performace on the test set """
            X_batch, Y_batch = batch(X[divide_set+1:], Y[divide_set+1:])
            feed_dict = {source_pl: X_batch, target_pl: Y_batch, training_pl: False}
            loss_value = sess.run(loss, feed_dict=feed_dict)
            
            summary_str = sess.run(summary_op, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
            summary_writer.add_run_metadata(run_metadata, 'step%d' % step)
            summary_writer.add_summary(summary_str, step)
            print('===== Step %d: test loss = %.2f (%.3f sec)' % (step, loss_value, duration))
    
    
    
    
def set_up_data():
    X, Y = pretreatment.import_dataset()
    
    
    
    
   
    
    print('Applying cleansing...')
    X = pretreatment.pretreatment(X)
    Y = pretreatment.pretreatment(Y)
    
    
    for sentence in X:
        del sentence[SENTENCE_LENGTH+1:]
        
    
    
    for sentence in Y:
        del sentence[SENTENCE_LENGTH+1:]
    
    print('Computing the corpus sizes...')
    compute_T(X, 'english')
    compute_T(Y, 'french')
    compute_S(X, 'english')
    compute_S(Y, 'french')
    compute_N(X, 'french')
    compute_N(Y, 'english')
    
    print('English corpus: %d tokens' % T_ENGLISH)
    print('French corpus: %d tokens' % T_FRENCH)
    print('English sentence length: %d' % S_ENGLISH)
    print('French sentence length: %d' % S_FRENCH)
    print('Number of sentences (both english and french): %d / %d' % (N_ENGLISH, N_FRENCH))
    
    print('Converting in one hot vectors')
    global CORPUS_ENGLISH, CORPUS_FRENCH
    params_ENGLISH = (N_ENGLISH, S_ENGLISH, T_ENGLISH)
    params_FRENCH = (N_FRENCH, S_FRENCH, T_FRENCH)
    X, CORPUS_ENGLISH= treatment.convert_to_one_hot(X, params_ENGLISH)
    Y, CORPUS_FRENCH= treatment.convert_to_one_hot(Y, params_FRENCH)
    
    return (X, Y)
    
    
    
    
def main():
    X, Y = set_up_data() # Load and prepare the dataset : X -> source sentences, Y -> target sentences 
    
    with tf.Graph().as_default():
        
        source_pl, target_pl, training_pl = placeholder_input()

        data = (X, Y)
        placeholders = (source_pl, target_pl, training_pl)
        corpus_params = (T_ENGLISH, S_ENGLISH, T_FRENCH, S_FRENCH)
        
        """ Defining tensorflow graph """
        scores = NMT.inference(placeholders, corpus_params, BATCH_SIZE)
        loss = NMT.conpute_loss(scores, target_pl)
        train_op = NMT.training(loss)
        
        print('Lauching session...')
        sess = tf.Session()        
        summary_op = tf.merge_all_summaries()
        sess.run(tf.initialize_all_variables())
        summary_writer = tf.train.SummaryWriter(DIRECTORY, sess.graph)
        
        saver = tf.train.Saver()
        
    
        print('Running training...')
        run_training(data, sess, placeholders, scores, loss, train_op, summary_op, summary_writer)
        
        print('Saving model...')        
        saver.save(sess, 'my-model', global_step=MAX_STEP)
    
        #saver.restore(sess, 'my-model-150')
        print('Conputing predictions...')
        run_prediction(data, sess, placeholders, scores)
        
        

if __name__ == '__main__':
    main()
