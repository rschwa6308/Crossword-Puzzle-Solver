import pickle
from typing import List, Tuple
import math
import numpy as np
import random
from functools import lru_cache

from ngram_searching import ngram_search



class Guesser:
    def load(self):
        """Initialize the guesser, reading from disk if needed"""
        pass
    
    @lru_cache(maxsize=10**3)
    def guess(self, clue: str, slot: str, max_guesses: int=5) -> List[Tuple[str, float]]:
        """Get a list of guesses represented as `(guess, confidence)` pairs (sorted best to worst)"""
        pass



class BasicGuesser(Guesser):
    def load(self):
        self.answers_train, self.vectorizer, self.model = pickle.load(open("trained_model.p", "rb"))
    
    @staticmethod
    def distance_to_confidence(dist):
        """map a clue embedding vector distance to a confidence value"""
        return 0.5 * math.e ** (-dist)
    
    @lru_cache(maxsize=10**3)
    def guess(self, clue, slot, max_guesses):
        clue_vector = self.vectorizer.transform([clue])

        # if clue vector is all 0's, we have never seen any of the words in the clue before
        # so we cannot even try to make a guess (yet)
        if clue_vector[0].nnz == 0:
            # TODO: default to n-gram search or something
            return []
        
        distances, indices = self.model.kneighbors(clue_vector, n_neighbors=20)
        raw_guesses = [self.answers_train[i] for i in indices[0]]
        # print([clues_train[i] for i in indices[0]])

        def valid(g):
            o = True
            if len(slot):
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



class HybridGuesser(Guesser):
    ngram_threshold = 0.10      # threshold at which to revert to raw n-gram searching

    def load(self):
        self.answers_train, self.vectorizer, self.model = pickle.load(open("trained_model.p", "rb"))
    
    @staticmethod
    def distance_to_confidence(dist):
        """map a clue embedding vector distance to a confidence value"""
        return 0.5 * math.e ** (-dist)
    
    @lru_cache(maxsize=10**3)
    def tfidf_guess(self, clue, slot):
        clue_vector = self.vectorizer.transform([clue])

        # if clue vector is all 0's, we have never seen any of the words in the clue before
        # so we cannot even try to make a guess (yet)
        if clue_vector[0].nnz == 0:
            # TODO: default to n-gram search or something
            return []
        
        distances, indices = self.model.kneighbors(clue_vector, n_neighbors=20)
        raw_guesses = [self.answers_train[i] for i in indices[0]]
        # print([clues_train[i] for i in indices[0]])

        def valid(g):
            o = True
            if len(slot):
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
    
    @lru_cache(maxsize=10**3)
    def guess(self, clue: str, slot: str, max_guesses: int=5) -> List[Tuple[str, float]]:
        tfidf_guesses = self.tfidf_guess(clue, slot)
        if len(tfidf_guesses) == 0 or max(conf for _, conf in tfidf_guesses) < self.ngram_threshold:
            # search for n-grams that fit the slot
            # for now, just find a single match (chosen arbitrarily)
            ngrams = [ngram for n in range(1, 3) for ngram in ngram_search(n, slot, single=True)]
            if not ngrams: return []
            ngram = random.choice(ngrams)
            guess = "".join(ngram)
            print("ngram guess:", guess)
            return [(guess, 0.05)]  # arbitrary (low) confidence score
        else:
            return tfidf_guesses



# Some testing...
if __name__ == "__main__":
    guesser = BasicGuesser()
    guesser.load()

    print(guesser.guess("opposite of NNE", slot="   "))