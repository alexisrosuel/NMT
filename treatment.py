# -*- coding: utf-8 -*-

"""
This file provide the functions to convert sentence in one-hot representation, or one-hot representation in sentence 
"""

import numpy as np

def convert_to_one_hot(X, params):
    """ Convert each token into its corresponding one-hot vector representation 
    
    Args:
        X: List of tokenised sentences (each sentence is a list of token)
        params: list of parameters (number of sentences, length of the sentences, and number of different tokens
        
    Returns:
        4D numpy array, shape=(N,1,S,T)
        List of the different token
    """    
    
    N, S, T = params
    X_flat = [item for sublist in X for item in sublist]
    union_token = list(set(X_flat))
    
    tensor = np.zeros(shape=(N,1,S,T), dtype=np.int8)
    for i in range(len(X)):
    	for j in range(len(X[i])):
         indice = union_token.index(X[i][j])
         tensor[i,0,j,indice] = 1
    return tensor, union_token


def convert_one_hot_to_string(X, corpus):
    """ Convert the one-hot vector representation into the corresponding word 
    
    Args:
        X: 3D numpy array, shape=(N,S,T)
        corpus: List of the different tokens
        
    Returns:
        List of sentences, where each sentence is a list of token
    """
    
    result = list()
    for sentence in X:
        result_one_sentence = list()
        for word in sentence:
           result_one_sentence.append(corpus[np.argmax(word)])
        result.append(result_one_sentence)
    return result
                
def list_to_sentence(token_list):
    """ Produce a sentence given the list of token to assemble 
    
    Args:
        token_list: List of token
    
    Returns:
        String (concatenation of the token with ' ' between each)"""
    return " ".join(token_list)


def sentences_from_one_hot(prediction, corpus):
    """ Convert the prediction of the NMT from the one_hot representation to the corresponding string 
    
    Args:
        prediction: 3D numpy array
        corpus: list of the different tokens
    
    Returns:
        List of string
    """
    
    a_afficher = 30 # Number of setences to keep for display

    list_of_tokens = convert_one_hot_to_string(prediction, corpus) # Convert each token into from its one-hot representation to the real token given the right corpus

    sentences = map(list_to_sentence, list_of_tokens[:a_afficher])
    
    return sentences
