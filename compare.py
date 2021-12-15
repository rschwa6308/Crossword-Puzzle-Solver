from typing import List, Optional, Tuple
from collections import defaultdict
import pickle
import json
import argparse
from os import path

import torch

from typing import Union, Dict

from sklearn.feature_extraction.text import TfidfVectorizer

from qanta_util.qbdata import QantaDatabase
from tfidf_guesser_test import StubDatabase

from sgd import kBIAS

MODEL_PATH = 'tfidf.pickle'
BUZZ_NUM_GUESSES = 10
BUZZ_THRESHOLD = 0.3


import pickle
from typing import List, Tuple
import math
import numpy as np
import random
from functools import lru_cache
# from helpers import EPSILON

EPSILON = 1e-6      # machine epsilon for fuzzy floating-point comparisons

from ngram_searching import ngram_search


class TfidfGuesser:
    """
    Class that, given a query, finds the most similar question to it.
    """

    def __init__(self):
        """
        Initializes data structures that will be useful later.
        """

        # You may want to add addtional data members

        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.i_to_ans = None  # sol

    def train(self, training_data: Union[StubDatabase, QantaDatabase], limit=-1) -> None:
        """
        Use a tf-idf vectorizer to analyze a training dataset and to process
        future examples.

        Keyword arguments:
        training_data -- The dataset to build representation from
        limit -- How many training data to use (default -1 uses all data)
        """

        questions = [x.text for x in training_data.guess_train_questions]
        answers = [x.page for x in training_data.guess_train_questions]

        if limit > 0:
            questions = questions[:limit]
            answers = answers[:limit]

        # Your code here
        self.i_to_ans = {i: ans for i, ans in enumerate(answers)}  # sol
        self.tfidf_vectorizer = TfidfVectorizer(  # sol
            ngram_range=(1, 3), min_df=2, max_df=.9  # sol
        ).fit(questions)  # sol
        self.tfidf_matrix = self.tfidf_vectorizer.transform(questions)  # sol

    def guess(self, questions: List[str], max_n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        """
        Given the text of questions, generate guesses (a list of both both the page id and score) for each one.

        Keyword arguments:
        questions -- Raw text of questions in a list
        max_n_guesses -- How many top guesses to return
        """

        guesses = []
        representations = self.tfidf_vectorizer.transform(questions)  # sol
        guess_matrix = self.tfidf_matrix.dot(representations.T).T  # sol
        guess_indices = (-guess_matrix).toarray().argsort(axis=1)[:, 0:max_n_guesses]  # sol
        guesses = []  # sol
        for i in range(len(questions)):  # sol
            idxs = guess_indices[i]  # sol
            guesses.append([(self.i_to_ans[j], guess_matrix[i, j]) for j in idxs])  # sol

        return guesses

    def confusion_matrix(self, evaluation_data: QantaDatabase, limit=-1) -> Dict[str, Dict[str, int]]:
        """
        Given a matrix of test examples and labels, compute the confusion
        matrixfor the current classifier.  Should return a dictionary of
        dictionaries where d[ii][jj] is the number of times an example
        with true label ii was labeled as jj.

        :param evaluation_data: Database of questions and answers
        :param limit: How many evaluation questions to use
        """

        # Finish this function to build a dictionary with the
        # mislabeled examples.  You'll need to call the guess
        # function.

        questions = [x.text for x in evaluation_data.guess_dev_questions]
        answers = [x.page for x in evaluation_data.guess_dev_questions]

        if limit > 0:
            questions = questions[:limit]
            answers = answers[:limit]

        d = defaultdict(dict)
        data_index = 0  # sol
        guesses = [x[0][0] for x in self.guess(questions, max_n_guesses=1)]  # sol
        for gg, yy in zip(guesses, answers):  # sol
            d[yy][gg] = d[yy].get(gg, 0) + 1  # sol
            data_index += 1  # sol
            if data_index % 100 == 0:  # sol
                print("%i/%i for confusion matrix" % (data_index,  # sol
                                                      len(guesses)))  # sol
        return d

    def save(self):  # sol
        with open(MODEL_PATH, 'wb') as f:  # sol
            pickle.dump({  # sol
                'i_to_ans': self.i_to_ans,  # sol
                'tfidf_vectorizer': self.tfidf_vectorizer,  # sol
                'tfidf_matrix': self.tfidf_matrix  # sol
            }, f)  # sol

    @classmethod  # sol
    def load(self):  # sol
        """ #sol
        Load the guesser from a saved file #sol
        """  # sol

        with open(MODEL_PATH, 'rb') as f:  # sol
            params = pickle.load(f)  # sol
            guesser = TfidfGuesser()  # sol
            guesser.tfidf_vectorizer = params['tfidf_vectorizer']  # sol
            guesser.tfidf_matrix = params['tfidf_matrix']  # sol
            guesser.i_to_ans = params['i_to_ans']  # sol
            return guesser  # sol


# You won't need this for this homework, but it will generate the data for a
# future homework; included for reference.
def write_guess_json(guesser, filename, fold, run_length=200, censor_features=["id", "label"]):
    """
    Returns the vocab, which is a list of all features.

    """

    vocab = [kBIAS]

    print("Writing guesses to %s" % filename)
    num = 0
    with open(filename, 'w') as outfile:
        total = len(fold)
        for qq in fold:
            num += 1
            if num % (total // 80) == 0:
                print('.', end='', flush=True)

            runs = qq.runs(run_length)
            guesses = guesser.guess(runs[0], max_n_guesses=5)
            scores = [guess[1] for guess in guesses]

            for raw_guess, rr in zip(guesses[0], runs[0]):
                gg, ss = raw_guess
                guess = {"id": qq.qanta_id,
                         "guess:%s" % gg: 1,
                         "run_length": len(rr) / 1000,
                         "score": ss,
                         "label": qq.page == gg,
                         "category:%s" % qq.category: 1,
                         "year:%s" % qq.year: 1}

                for ii in guess:
                    # Don't let it use features that would allow cheating
                    if ii not in censor_features and ii not in vocab:
                        vocab.append(ii)

                outfile.write(json.dumps(guess, sort_keys=True))
                outfile.write("\n")
    print("")
    return vocab


def product(nums):
    p = 1
    for n in nums: p *= n
    return p


class Guesser:
    def load(self):
        """Initialize the guesser, reading from disk if needed"""
        pass

    @lru_cache(maxsize=10 ** 3)
    def guess(self, clue: str, slot: str, max_guesses: int = 5) -> List[Tuple[str, float]]:
        """Get a list of guesses represented as `(guess, confidence)` pairs (sorted best to worst)"""
        pass

class HybridGuesser(Guesser):
    ngram_threshold = 0.05  # threshold at which to revert to raw n-gram searching

    def load(self):
        self.answers_train, self.vectorizer, self.model = pickle.load(open("trained_model.p", "rb"))

    @staticmethod
    def distance_to_confidence(dist):
        """map a clue embedding vector distance to a confidence value"""
        return 0.5 * math.e ** (-dist)

    @lru_cache(maxsize=10 ** 3)
    def tfidf_guess(self, clue, slot_length):
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
            if slot_length:
                o &= len(g) == slot_length
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
            (g, 1 - product(1 - conf for g_, conf in guesses if g_ == g))
            for g in unique_guesses
        ]

        return list(sorted(guesses_combined, key=lambda item: item[1], reverse=True))

    def guess(self, clue: str, slot: str, max_guesses: int = 5) -> List[Tuple[str, float]]:
        tfidf_guesses = self.tfidf_guess(clue, len(slot))
        if len(tfidf_guesses) == 0 or max(conf for _, conf in tfidf_guesses) < self.ngram_threshold:
            # search for n-grams that fit the slot
            # for now, just find a single match (chosen arbitrarily)
            ngrams = [ngram for n in range(1, 3) for ngram in ngram_search(n, slot, single=True)]
            if not ngrams: return []
            ngram = random.choice(ngrams)
            guess = "".join(ngram)
            print("ngram guess:", guess)
            return [(guess, self.ngram_threshold + EPSILON)]  # arbitrary (low) confidence score
        else:
            return tfidf_guesses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--guesstrain", default="data/small.guesstrain.json", type=str)
    parser.add_argument("--guessdev", default="data/small.guessdev.json", type=str)

    parser.add_argument("--buzztrain", default="data/small.buzztrain.json", type=str)
    parser.add_argument("--buzzdev", default="data/small.buzzdev.json", type=str)
    parser.add_argument("--limit", default=-1, type=int)
    parser.add_argument("--vocab", default="", type=str)

    parser.add_argument("--buzztrain_predictions", default="crossword.guess.buzztrain.jsonl", type=str)
    parser.add_argument("--buzzdev_predictions", default="crossword.guess.buzzdev.jsonl", type=str)

    flags = parser.parse_args()

    print("Loading %s" % flags.guesstrain)
    guesstrain = QantaDatabase(flags.guesstrain)
    guessdev = QantaDatabase(flags.guessdev)

    # tfidf_guesser = TfidfGuesser()
    # tfidf_guesser.train(guesstrain, limit=flags.limit)
    # tfidf_guesser.save()  # sol

    guesser = HybridGuesser()
    guesser.load()

    # confusion = tfidf_guesser.confusion_matrix(guessdev, limit=-1)
    # print("Errors:\n=================================================")
    # for ii in confusion:
    #     for jj in confusion[ii]:
    #         if ii != jj:
    #             print("%i\t%s\t%s\t" % (confusion[ii][jj], ii, jj))

    for qa in guessdev.guess_dev_questions:
        # question = qa.first_sentence
        question = qa.text
        answer = qa.answer

        print("--------------------------------------------------------------------")
        print("question: ", question)
        print("answer: ", answer)

        print("crossword_tfidf guess: ", guesser.guess(question, slot=(" "*len(answer.strip())) ))

    # if flags.buzztrain_predictions:
    #     print("Loading %s" % flags.buzztrain)
    #     buzztrain = QantaDatabase(flags.buzztrain)
    #     vocab = write_guess_json(tfidf_guesser, flags.buzztrain_predictions, buzztrain.buzz_train_questions)
    #
    # if flags.vocab:
    #     with open(flags.vocab, 'w') as outfile:
    #         for ii in vocab:
    #             outfile.write("%s\n" % ii)
    #
    # if flags.buzzdev_predictions:
    #     assert flags.buzztrain_predictions, "Don't have vocab if you don't do buzztrain"
    #     print("Loading %s" % flags.buzzdev)
    #     buzzdev = QantaDatabase(flags.buzzdev)
    #     write_guess_json(tfidf_guesser, flags.buzzdev_predictions, buzzdev.buzz_dev_questions)
