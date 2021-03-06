from typing import List, Tuple
import wikipedia

from guessers import BasicGuesser


class QuizBowlModel:
    THRESHOLD = 0.3

    def __init__(self):
        self.guesser = BasicGuesser()
        self.guesser.load()

    def convert_to_QA_answer(self, guess):
        try:
            page_title = wikipedia.search(guess, results=1)[0]
        except Exception:
            page_title = guess.title()

        return page_title.replace(" ", "_")

    def guess_and_buzz(self, question_text: List[str]) -> List[Tuple[str, bool]]:
        """
        Use the crossword guesser to generate general QA guesses.

        Try slot lengths of varying length, returning the guess that has the highest confidence score.
        Buzz if that confidence score is above a reasonable threshold.
        """
        guess_buzzes = []

        for question in question_text:
            best_guess = ("???", -1.0)
            for slot_length in range(4, 15):
                guesses = self.guesser.guess(question, slot=" "*slot_length, max_guesses=1)
                if guesses:
                    guess = guesses[0]
                    if guess[1] > best_guess[1]:
                        best_guess = guess
            
            guess_buzzes.append((
                self.convert_to_QA_answer(best_guess[0]),
                best_guess[1] > self.THRESHOLD
            ))
        
        return guess_buzzes



# if __name__ == "__main__":
#     m = QuizBowlModel()
#     print(m.guess_and_buzz([
#         "first man on the moon",
#         "adam and eve garden"
#     ]))
