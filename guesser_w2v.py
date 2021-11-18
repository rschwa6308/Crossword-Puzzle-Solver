import os
import json
import pickle
from pprint import pprint
from typing import List, Tuple, set 
import numpy as np
import math
import gensim.downloader
import gensim.models
from scipy import spatial
from sklearn.neighbors import NearestNeighbors

'''
    This class trains and/or generates a guesser for the cross-word puzzle game
    using gensim word2vec combined with sklearn KNearestHeighbors

'''
MODEL_PATH = 'trained_model.p'
    

class W2VGuesser:
    def __init__(self):
        self.model = None
        self.matrix = None
        self.nn_model = None
        self.dim = None
        self.answers = None

    # function average word2vec vector
    def avg_feature_vector(words, model, num_features, ind2key_set):
        feature_vec = np.zeros((num_features, ), dtype='float32')
        n_words = 0
        for word in words:
            if word in ind2key_set:
                n_words += 1
                feature_vec = np.add(feature_vec, model[word])
        if (n_words > 0):
            feature_vec = np.divide(feature_vec, n_words)
        return feature_vec

    # define cosine similarity score
    def sim_score(v1,v2):
        return 1 - spatial.distance.cosine(v1, v2)
    
    # define vectorizer
    def word2vec_vectorizer(self,data,model,num_features,ind2key_set):
        vec_data = []
        for sentence in data:
            sentence = [word for word in sentence if len(word)>1]
            vec_data.append(self.avg_feature_vector(sentence,model,num_features,ind2key_set))
    
        return vec_data

    def train(self, clues_train, answers, self_train_w2v=False, dim=100) -> None:
        # clean data
        new_clues_train = []
        for clue in clues_train:
            clue = clue.replace('\'', '')
            clue = clue.replace('"', '')
            clue = clue.replace(':', '')
            new_clues_train.append([w.lower() for w in clue.split(' ')])
        clues_train = new_clues_train

        if self_train_w2v:
            # build word2vec
            model = gensim.models.word2vec.Word2Vec(new_clues_train, vector_size=dim, min_count=1)
            self.model = model.wv
        else:
            # use pre-trained model
            model = gensim.downloader.load('glove-wiki-gigaword-100')
            self.model = model

        self.answers = answers
        self.dim = dim
        self.matrix = self.word2vec_vectorizer(new_clues_train,self.model,dim,set(model.wv.index_to_key))
        self.nn_model = NearestNeighbors().fit(self.matrix)

    def guess(self, clue: str, slot: str, max_guesses: int=5) -> List[Tuple[str, float]]:
        clue = clue.replace('\'', '')
        clue = clue.replace('"', '')
        clue = clue.replace(':', '')
        clue_vector =  self.word2vec_vectorizer([clue],self.model,self.dim,set(self.model.index_to_key))
        distances, indices = self.nn_model.kneighbors(clue_vector,n_neighbors=max_guesses)
        raw_guesses = [self.answers[i] for i in indices[0]]

        def valid(g):
            o = True
            o &= len(g) == len(slot)
            o &= g.lower() not in clue.lower()
            return o
    
        # convert distances to confidences
        guesses = [
            (g, self.distance_to_confidence(d))
            for g, d in zip(raw_guesses, distances[0]) if valid(g)
        ]

        # if a guess appears multiple times, interpret confidences as independent probabilities and combine
        unique_guesses = set(g for g, _ in guesses)
        guesses_combined = [
            (g, 1 - math.prod(1-conf for g_, conf in guesses if g_==g))
            for g in unique_guesses
        ]

        return list(sorted(guesses_combined, key=lambda item: item[1], reverse=True))

    def save(self):
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'matrix': self.matrix,
                'nn_model': self.nn_model,
                'dim': self.dim,
                'answers': self.answers
            }, f)

    @classmethod
    def load(cls):
        with open(MODEL_PATH, 'rb') as f:
            params = pickle.load(f)
            guesser = W2VGuesser()
            guesser.model= params['model']
            guesser.matrix = params['matrix']
            guesser.answers = params['answers']
            guesser.dim = params['dim']
            guesser.nn_model = params['nn_model']
            return guesser
