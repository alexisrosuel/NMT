# -*- coding: utf-8 -*-

"""
This file import the data and apply the appropriate cleansing rules
"""

import re
import unicodedata
from collections import Counter


import nltk
# nltk.download() # Only for the first run

k = 2 # Minimum number of occurence for a token to be learned

def import_dataset():
    """ Import the source and target sentences 
    
    Args:
    	_
    	
    Returns:
    	List of sentences
    	List of sentences
    """
    
    print("Opening documents.")
    english_set = open("Dataset/europarl-v7.fr-en.en", "r")
    french_set = open("Dataset/europarl-v7.fr-en.fr", "r")
    
    print("Reading documents...")
    X = english_set.read(50000000).decode("utf-8")
    Y = french_set.read(50000000).decode("utf-8")
    
    print("Closing documents.")
    english_set.close()
    french_set.close()
    
    # Split the content of the documents into sentences
    X = X.split('\n') 
    Y = Y.split('\n')
    
    nb_sentences= min(len(X),len(Y)) # Keep the same number of sentences in both languages
    
    return(X[:nb_sentences],Y[:nb_sentences])



def tokenize(corpus):
    """ Tokenize a corpus 
    
    Args:
    	corpus: List of sentences
    	
    Returns:
    	List tokenised sentences (each sentence is a list of token)
    """
    print("Tokenisation in progress...")
    tokenised_corpus= map(nltk.word_tokenize,corpus)
    print("Tokenisation finished.")
    return tokenised_corpus

def remove_less_used_words(X):
    """ Remove a token if it appears less than k time. Also remove the empty token 
    
    Args:
    	X: List of tokenised sentences (each sentence is a list of token)
    
    Returns:
    	List of tokenised sentences (each sentence is a list of token)
    """
    
    
    print('Removing token not enough used...')
    X_flat = [item for sublist in X for item in sublist]
    
    """ Create a dictionary {token1: token1_occurence, token2: token2_occurence, etc.} """    
    dictionary_occurences = Counter(X_flat)
    
    for i in range(len(X)):
        to_be_deleted = 0 # Count the number of empty string to delete after the following processing
        for j in range(len(X[i])):
            if X[i][j] == '':
                X[i].insert(len(X[i]), X[i].pop(j)) # Deplace the empty string at the end of the current sentence
                j = j - 1
                to_be_deleted += 1
            elif dictionary_occurences[X[i][j]] <= k:
                X[i][j] = 'UNK'
        X[i] = X[i][:-to_be_deleted] # Delete the empty string deplaced at the end of the sentence
    print('Removing completed.')
    
    return X
     
    
def remove_special_character(X):
    """ Remove all the special characters from the sentences. This includes :
        1. Accentuation
        2. Uppercase
        3. Punctionation
        4. Numbers 
        
    Args:
    	X: List of tokenised sentences (each sentence is a list of token)
    	
    Returns:
    	List of tokenised sentences (each sentence is a list of token)
    """

# CHIFFRE pour chiffre

    print('Removing special characters...')


    for i in range(len(X)):
        for j in range(len(X[i])):
            # Accents    
            token = unicodedata.normalize('NFD', X[i][j]).encode('ascii', 'ignore') 
            token = token.decode('utf-8')
    
            # Special characters
            token = re.sub(r'[^\w\s]','',token)
    
            # Uppercase 
            token = token.lower()
            X[i][j] = token
            
   		 	# Numbers
   		 	# ...
    
   		 	# Punctuation

    
   
    return X


def standardize_sentence_length(X):
    """ Add the 'EOS' token at the end of each sentence, and make all the sentences the same size (here the choosen size is the longest sentence) by adding a token 'FILL' until the end 
    
    Args:
    	X: List of tokenised sentences (each sentence is a list of token)
    
    Returns:
    	List of tokenised sentences (each sentence is a list of token)
    """

    longueur_max = max(map(len,X))+1 # Compute the sentence size 

    for sentence in X:
        #sentence.append('EOS')
        if len(sentence) < longueur_max:
            sentence.extend(['FILL' for i in range(1 + longueur_max - len(sentence))])

    return X


def pretreatment(X):

    """ Apply the following rules to the sentences :
        1. tokenisation
        2. special character removed
        3. not enough frequent token removed
        4. 'EOS' token added and sentences length standardized (to the max)
    
    Args:
    	X: List of sentences
    
    Returns:
    	List of tokenised sentences (each sentence is a list of token)
    """


    X = tokenize(X)
    X = remove_special_character(X)
    X = remove_less_used_words(X)

    return X

