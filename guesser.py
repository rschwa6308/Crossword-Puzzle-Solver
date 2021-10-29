import pickle


answers_train, vectorizer, model = pickle.load(open("trained_model.p", "rb"))


# Define a guesser function
def guess(clue, slot_length=None, max_guesses=5, max_guesses_raw=30):
    clue_vector = vectorizer.transform([clue])
    distances, indices = model.kneighbors(clue_vector, n_neighbors=max_guesses_raw)
    raw_guesses = [answers_train[i] for i in indices[0]]
    # print([clues_train[i] for i in indices[0]])

    def valid(g):
        o = True
        if slot_length:
            o &= len(g) == slot_length
        o &= g.lower() not in clue.lower()
        return o
    
    guesses = [g for g in raw_guesses if valid(g)]
    return guesses[:max_guesses]

    # TODO:
    # - include a confidence with each guess
    # - use repeated guesses and distances to determine confidence

