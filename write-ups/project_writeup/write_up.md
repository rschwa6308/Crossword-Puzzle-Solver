# CMSC 470 Final Project: Shortz Circuit
Russell Schwartz, Chenqi Zhu, Henrique Corte, Ben Tompkins, Dan Song


## Project Overview
The goal of our project is to build an automatic crossword puzzle solver, trained on a data set of ~14,500 New York Times crosswords dating back to 1976.
We used two separate implementations to get our solver to work: A guesser, and a solver.

## Framework
The `Puzzle` class makes it easy to interact with a crossword puzzle, which is nontrivial since the slot-identifier scheme is idiosyncratic. There are also lots of helper functions for visualizing a puzzle in the terminal while it is being solved. 

![](terminal_screenshot.png)


## Guesser Implementations
The `guess` function should take as input the clue and the current contents of the slot and then generate a set of guesses which fit the slot, and each have an associated confidence score. Our current implementation uses vanilla TF-IDF trained on a huge number of clues (602,694). 

When given a test clue, we find the `k` clues that are most similar (experimenting with both cosine similarity and L2-norm) and then filter for those that are compatible with the slot. Then, we use the similarity scores combined with the presence of repeats to calculate confidence scores. This looks like (from a `Guesser` class):

### TF-IDF
Crossword puzzles love to repeat clue-answer pairs, so this approach works well. On our test set, the correct answer appeared in the top 5 best guesses ~60% of the time.

If TFIDF does not return any guesses above a meager confidence threshold, then the guesser resorts to n-gram searching, where it simply tries to find any sequence of English words that fit the slot.

Here is our TF-IDF Guesser Implementation:
```python
@lru_cache(maxsize=10**3)
def tfidf_guess(self, clue, slot_length):
    clue_vector = self.vectorizer.transform([clue])

    if clue_vector[0].nnz == 0:
        return []
    
    distances, indices = self.model.kneighbors(clue_vector, n_neighbors=20)
    raw_guesses = [self.answers_train[i] for i in indices[0]]

    def valid(g):
        o = True
        if slot_length:
            o &= len(g) == slot_length
        o &= g.lower() not in clue.lower()
        return o
    
    guesses = [
        (g, self.distance_to_confidence(d))
        for g, d in zip(raw_guesses, distances[0]) if valid(g)
    ]

    unique_guesses = set(g for g, _ in guesses)
    guesses_combined = [
        (g, 1 - product(1-conf for g_, conf in guesses if g_==g))
        for g in unique_guesses
    ]

    return list(sorted(guesses_combined, key=lambda item: item[1], reverse=True))
```

### Word2Vec Guesser Attempt
Word2Vec is a useful tool in NLP which maps each word to a vector based on its association with the documents. It is good at detecting the ‘similarity’ between different words as two similar words would result in two similar vectors in the n-dimensional vector space and vice versa. We initially believe this would be a good implementation of guesser as the clues are comprised of short sentences with fewer words than the quizzes we have learned throughout the semester. In theory, Word2Vec would be good at matching two similar clues together by calculating the closeness (cosine similarity) between 2 vectors.

Gensim is a library containing a good implementation of Word2Vec trainer and various pre-trained models. We used modules from this library to train and test the Word2Vec guesser. As a clue has multiple words, we applied an average function `avg_feature_vector` on each clue to obtain the vector representation of each clue. The vectors could also be clustered using `KNearestNeighbors` method and the `guess` function utilizes this property to obtain the nearest n guesses fast. 

The following code defines the W2VGuesser guess functionS:

```python
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

    guesses = [
        (g, self.distance_to_confidence(d))
        for g, d in zip(raw_guesses, distances[0]) if valid(g)
    ]

    unique_guesses = set(g for g, _ in guesses)
    guesses_combined = [
        (g, 1 - math.prod(1-conf for g_, conf in guesses if g_==g))
        for g in unique_guesses
    ]

    return list(sorted(guesses_combined, key=lambda item: item[1], reverse=True))
```
As word vectors lives in n-dimensional vector space, it is possible to project each vector onto 2-dimensional plane to observe the closeness between different vectorized clues. To project a vector of higher dimension to a lower dimension space, one common method is Principal Component Analysis (PCA). PCA employs Singular Value Decomposition (SVD) to extract the m-dimensional data from n-dimensional data(n$\geq$m) by preserving data corresponding to m-largest singular values.

![](w2v_pca.png)

In this example where the testing clue was ‘radiator output’, the Word2Vec guesser successfully distinguished the testing data that matched to the clue (labeled red) with other confusing puzzles(blue) that also include the word ‘output’. We can see under the PCA the correct clue vector is not aligned with the incorrect ones on the x-axis.

![](w2v_pca2.png)

While the Word2Vec guesser may be effective at distinguishing the wrong clues, in some other cases it was not able to find the correct one. This example above illustrates that despite the guesser separating ‘African language group’ from other culture-language related confusing clues it did not find any correct vector close to the testing clue.

The result of this attempt, however, was less than ideal. The Word2Vec implementation of guesser only achieved an accuracy of roughly 45% compared to what we had for 55% in the baseline guesser. It is possible that in the crossword puzzle, the same answer was asked in completely different way resulting in the vector being far from each other.

### ELMo Attempt

We decided to try ELMo word embeddings to generate a more powerful representation of the clues for the word puzzles. We believed that these embeddings would be able to represent clues significantly better and boost our overall performance.

For our implementation, we used: 
1. We used AllenNLP’s pretrained ELMo model to generate word embeddings for our 800,000+ training samples. 
2. We ran into the problem of different dimensionality for clues of different lengths, so we used the same approach as DANS and averaged the ELMo embeddings for each clue.
3. Finally, we ran the average embeddings through a nearest neighbor model.

From running the average ELMo embeddings through Nearest Neighbors, we were able to achieve about 60% accuracy. This was roughly a 6-7% improvement on our previous TF-IDF model.
How could we improve this model?
we believe our accuracy was only 60% because the Nearest Neighbor model was not powerful enough to capture a lot of the differences in clues.
Next steps:
We believe that using a neural model would significantly improve our accuracy, more specifically a deep averaging network.
We were not able to do this due to lack of computation power, 860,000 samples and each embedding having 1000+ dimensions.


#
In the end, we decided to go with TF-IDF as our final guesser. While this was our baseline, the other models did not significantly improve accuracy, but instead greatly increased overhead in encoding and training. In the end we get around 60-70% accuracy for the TF-IDF guesser.


## Solver Implementations
A relatively naive solution:

```python
class BasicSolverThreshold(Solver):
    """
    Only fill in a slot if the guess confidence is above a threshold, which decreases over with time.
    Run until threshold hits a minimum degeneracy point (say, 5% confidence)
    """

    guesser_class: Type[Guesser] = BasicGuesser

    def solve(self, puzzle: Puzzle) -> bool:
        threshold = 0.75 

        while not puzzle.grid_filled() and threshold >= 0.05:
            stuck = True
            for ident in puzzle.get_identifiers():
                current_slot = puzzle.read_slot(ident)
                if " " not in current_slot: continue

                clue = puzzle.get_clue(ident)
                gs = self.guesser.guess(clue, puzzle.read_slot(ident), max_guesses=5)

                for g, conf in gs:
                    if compatible(current_slot, g) and conf >= threshold:
                        puzzle.write_slot(ident, g)
                        stuck = False
                        break
            
            threshold *= 0.5
        
        return not stuck
```

This algorithm has the highly restrictive property that they solve the puzzle "in ink" so to speak, meaning that once a slot is written to it is never changed. Despite being dumb, they work alright.

Here are some performance metrics for a test suite of 100 randomly chosen test puzzles.
```
{'average_fill_accuracy': 0.606,
 'average_fill_percentage': 0.745,
 'solver': <class 'solvers.BasicSolverThreshold'>}
```

`average_fill_accuracy` represents the percentage of filled cells that were correct

`average_fill_percentage` represents the percentage of the grid that was filled at all (at present, these solvers leave a slot blank if they have never seen any of the words in the clue before)


A smarter solution:

```python

class CellConfidenceSolver(Solver):
    """
    Each filled cell has associated confidence score (derived from guess confidence).
    Low confidence cells can be overwritten by subsequent guesses.
    """

    guesser_class: Type[Guesser] = HybridGuesser

    def solve(self, puzzle: Puzzle):
        self.confidence_grid = [
            [None if cell == "." else 0.0 for cell in row]
            for row in puzzle.grid
        ]

        conf_threshold = 0.90

        converged = False
        while not converged:
            converged = True
            for ident in puzzle.get_identifiers():
                current_slot = puzzle.read_slot(ident)
                slot_coords = puzzle.cells_map[ident]
                slot_confidence_avg = average(self.confidence_grid[y][x] for x, y in slot_coords)

                clue = puzzle.get_clue(ident)
                gs = self.guesser.guess(clue, puzzle.read_slot(ident), max_guesses=5)

                for g, conf in gs:
                    slot_confidence_avg_changed = average(
                        self.confidence_grid[y][x]
                        for (x, y), old, new in zip(slot_coords, current_slot, g)
                        if old != new
                    )
                    # overwrite the current slot if several conditions are met
                    if all([
                        g != current_slot,
                        conf > conf_threshold,
                        conf > slot_confidence_avg_changed + EPSILON,
                    ]):
                        # transfer guess confidence to cell confidence
                        for (x, y), old, new in zip(slot_coords, current_slot, g):
                            if old != new:
                                self.confidence_grid[y][x] = conf                               # cell contradicted
                            else:
                                old_conf = self.confidence_grid[y][x]
                                self.confidence_grid[y][x] = 1 - (1 - old_conf)*(1 - conf)      # cell corroborated
                        
                        # overwrite contents of slot with the new guess
                        puzzle.write_slot(ident, g)
                        
                        converged = False
                        break
            
            conf_threshold *= 0.5   # exponential decay
```

This solver works by maintaining a grid of confidence values (one for each cell). For example:

![](confidence_grid.png)

Confidence in a cell is inherited from the confidence in the guess. Confidence values can be increased or decreased through corroboration or contradiction. In the case that a cell with confidence $c$ is corroborated by a new guess with confidence $c'$, we update according to the following rule:

$$c \gets 1 - (1 - c)(1 - c')$$

Which corresponds to $\mathrm{Prob}(c \lor c')$ if we interpret confidence scores as independent probabilities.

Performance metrics:

```
{'average_fill_accuracy': 0.704,
 'average_fill_percentage': 0.934,
 'solver': <class 'solvers.CellConfidenceSolver'>}
```

With these metrics, we believe our model and techniques for guessing and solving to work. While we never reached the gold-standard of 94%, we reached to around 70% just using TF-IDF and leveraging cell confidences.
These are great results that dont require an extreme amount of overhead to build encodings and teach models. 

## Error Analysis
### Effects of Question Type
In NYT Crossword puzzles, there are 4 forms of clues: Questions, Fill-in-the-Blank, Clues with Specifiers, and General Clues. **Questions** usually come in the form of jokes, puns, or sayings
(i.e., Go for the gold? -> Pan, Win; where the first word is correct, second is our model's guess). **Fill-in-the-blank** is self-explanatory (i.e., ___-la-la -> Ooh, Tra). Next are clues with specifiers, where the clue ends with a categorical hint in parentheses (i.e., Pizzeria ___ (fast-food chain) -> UNO, KFC).
Lastly are general clues, which are clues that usually ask for synonyms or related terms (i.e., Cell, e.g. -> Phone, Pager)

A potential issue with some of the clues is the model may not be aware of the different types and their rules. For instance, the model will see “Look ___ (probe)” and guess “Test.” Here the model sees “probe” and a good four-letter word that matches is “test”. However, the model is unaware that “___” means that the guessed word belongs in this context. So in actuality, the answer is “Into.”

This also goes for question clues that require a little more real-knowledge and cleverness that our tf-idf models may fail to see.

![](clue_type_accuracy.png)

Based on a sample of 100 random crosswords, our system had the most success answering Fill-in-the-Blank clues along with mild success with specifier clues.
Our models had the most problems guessing clues with questions. This makes sense as these clues require the most real-world knowledge as they are often jokes, puns, or fairly ambiguous. 
In the future, we suggest leveraging a model that can work around these real-world dependencies for more accurate guesses.

### Slot and Clue Length

Other findings in the error analysis looked at the slot and clue length and seeing if these number impacted accuracy. We found that slot length does not affect the accuracy of the guess; incorrect guesses did not skew longer/shorter than correct guesses. Both correct guesses and incorrect guesses averaged to a little over 4. 
However, we also found that clue length does affect the accuracy of the guess. Longer clues lead to more inaccurate results. This makes sense as longer clues usually means more words for the model to get distracted with. Also, in longer clues there is more of a probability that the clue contains a word that the model has never seen before.


## Who did what?
**Russell Schwartz**
- Worked on data collection, main framework, baseline TFIDF guesser, and solver algorithms.
- Found, processed, and cleaned data set
- Ran statistics on data set
- Wrote Puzzle, Guesser, and Solver classes
- Wrote testing and puzzle visualization utilities
- Implemented TF-IDF baseline
- Developed Threshold Solver Algorithm
- Developed Cell Confidence Solver Algorithm

**Henrique Corte**
- Due to the low accuracy from tf-idf and word2vec, I worked on implementing averaged ELMo embeddings to NN model.
- Error analysis of the ELMo model and analyzed what clue-pairs the model under performed for.
- Found ways to boost accuracy, including averaging the embeddings, batching training samples, cleaning test data, data selection to improve embedding speed and computational performance.
- Attempted to run these embeddings through different models, including logistic regression and deep networks - both of which were unsuccessful due to absurd computing time.

**Chenqi Zhu**
- Responsible for the Word2Vec representation of the clues. 
- Constructed W2VGuesser class and associated functions.
- Attempted using W2V as embedding in DAN but failed at making the model compatible with tensorflow.
- Visualization of Word2Vec using PCA and analysis on some examples.

**Ben Tompkins**
- Research and Development on the guesser, specifically the fallback when the guesser receives a clue where it doesn't recognize any of the worlds. 
  - Word2Word Generator model -> doesn't rely on having to see the words; provided good guesses but were incompatible and not the right length.
  - Wikipedia fallback that would look the clue up on Wikipedia if it didn't recognize the words -> This increased fill percentage but did not increase accuracy. 
  - Ended up going with n-gram implementation
- Error Analysis on the final model.

**Daniel Song**
- Worked on and tested final Guesser Implementation.
- Experimented with other pretrained model options through Huggingface and Transformers.
- Worked to make Crossword submission compatible with QA Gradescope submission.


