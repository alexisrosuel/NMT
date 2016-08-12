# -*- coding: utf-8 -*-

"""
This file import the data and apply the appropriate cleansing rules
"""

import re
import unicodedata

import nltk
# nltk.download() # Only for the first run

k = 2 # Minimum number of occurence for a token to be learned

def import_dataset():
    """ Import the source and target sentences """
    print("Opening documents.")
    english_set = open("Dataset/europarl-v7.fr-en.en", "r")
    french_set = open("Dataset/europarl-v7.fr-en.fr", "r")
    
    print("Reading documents...")
    X = english_set.read(1000000).decode("utf-8")
    Y = french_set.read(1000000).decode("utf-8")
    
    print("Closing documents.")
    english_set.close()
    french_set.close()
    
    # Split the content of the documents into sentences
    X = X.split('\n') 
    Y = Y.split('\n')
    
    nb_sentences= min(len(X),len(Y)) # Keep the same number of sentences in both languages
    
    return(X[:nb_sentences],Y[:nb_sentences])



def tokenize(corpus):
    """ Tokenize a corpus """
    print("Tokenisation in progress...")
    tokenised_corpus= map(nltk.word_tokenize,corpus)
    print("Tokenisation finished.")
    return tokenised_corpus

def remove_less_used_words(X):
    """ Remove a token if it appears less than k time. Also remove the empty token """
    #A faire la suppresion des tokens vides !
    
    print('Removing token not enough used...')
    X_flat = [item for sublist in X for item in sublist]
    
    """ Create a dictionary {token1: token1_occurence, token2: token2_occurence, etc.} """
    union_token = list(set(X_flat))
    occurences = map(X_flat.count,union_token)
    dictionary_occurences = dict(zip(union_token, occurences))
    
    for sentence in X:
    	for token in sentence:
    		if token == '':
    			del token
    		elif dictionary_occurences[token] <= k:
    			token = 'UNK'

#    A OPTIMISER
    already_deleted = list() # List of the token which must be replaced by 'UNK'
    for i in range(len(X)):
        for j in range(len(X[i])):
            if X[i][j] == '':
                del X[i][j] #If it is the empty token -> delete 
            elif X[i][j] in already_deleted:
                X[i][j] = 'UNK' # If already identified as a token to remove -> 'UNK' 
            else:
			# Else, if a new unused token is discovered -> remove and add to the list already_deleted
                indice = union_token.index(X[i][j]) 
                if occurences[indice] <= k:
                    already_deleted.append(X[i][j])
                    X[i][j] = 'UNK' 

# Essayer avec un dict
#    A OPTIMISER
    already_kept = list() # List of the token which must be replaced by 'UNK'
    for i in range(len(X)):
        for j in range(len(X[i])):
            if X[i][j] == '':
                del X[i][j] #If it is the empty token -> delete 
            elif X[i][j] not in already_kept:
                X[i][j] = 'UNK' # If already identified as a token to remove -> 'UNK' 
            else:
			# Else, if a new unused token is discovered -> remove and add to the list already_deleted
                indice = union_token.index(X[i][j]) 
                if occurences[indice] > k:
                    already_kept.append(X[i][j])
                else:
                	X[i][j] = 'UNK' 

    print('Removing completed.')
    
    return X
     
    
def remove_special_character(X):
    """ Remove all the special characters from the sentences. This includes :
        1. Accentuation
        2. Uppercase
        3. Punctionation
        4. Numbers 
    """

# CHIFFRE pour chiffre

    print('Removing special characters...')

# Essai fonction
    for sentence in X:
        for token in sentence:
			# Accents    
            token = unicodedata.normalize('NFD', token).encode('ascii', 'ignore') 
            token = token.decode('utf-8')
    
    		# Special characters
            token = re.sub(r'[^\w\s]','',token)
    
   		 	# Uppercase 
            token = token.lower()
   	 
   		 	# Numbers
   		 	# ...
    
   		 	# Punctuation

    
    #X = map(remove_special_character_sentence,X)
    return X


def standardize_sentence_length(X):
    """ Add the 'EOS' token at the end of each sentence, and make all the sentences the same size (here the choosen size is the longest sentence) by adding a token 'FILL' until the end """

    longueur_max = max(map(len,X))+1 # Compute the sentence size 

    for sentence in X:
        sentence.append('EOS')
        if len(sentence) < longueur_max:
            sentence.extend(['FILL' for i in range(1 + longueur_max - len(sentence))])

    return X


def pretreatment(X):

    """ Apply the following rules to the sentences :
        1. tokenisation
        2. special character removed
        3. not enough frequent token removed
        4. 'EOS' token added and sentences length standardized (to the max) 
    """
# faire la separation en longueur ici ! 

    X = tokenize(X)
    X = remove_special_character(X)
    #X = remove_less_used_words(X)
    X = standardize_sentence_length(X)
    
    return X

