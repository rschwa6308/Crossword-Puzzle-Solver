from typing import Type
from guessers import BasicGuesser, Guesser
from puzzle import Puzzle
from helpers import pprint_grid, clear_console



class Solver:
    guesser_class: Type[Guesser] = None

    def __init__(self):
        self.guesser = self.guesser_class()
        self.guesser.load()

    def solve(self, puzzle: Puzzle) -> bool:
        """
        Attempt to solve the given puzzle, returning whether we succeeded in filling the grid or not.
        Modifies the given puzzle in place.
        """
    
    def print_update_animation_frame(self, puzzle, guess, ident, confidence, clear=True):
        if clear: clear_console()

        print(f"Solver:   {type(self).__name__}")
        print(f"Puzzle:   {puzzle.puzzle_dict['dow']}, {puzzle.puzzle_dict['date']}\n")

        print(f"Writing {guess:^15} to slot {ident}", end="")
        # print(f"Confidence: {confidence:^15.0%} ", end="")
        # print(f"Writing {guess:^15} to slot {ident} with {confidence:.0%} confidence\t", end="")

        if guess == puzzle.get_answer(ident):
            print("\t(CORRECT ✓)")
        else:
            print("\t(INCORRECT ✗)")
        
        message = f"{confidence:.0%} " + "😕😐😀"[int(confidence*3)] + " "
        print(f"        {message:^14}\n")
        
        pprint_grid(puzzle.grid, puzzle.get_acc_grid())
        print()




def compatible(a, b):
    if len(a) != len(b): return False

    for a_char, b_char in zip(a, b):
        if a_char == " " or b_char == " ": continue
        if a_char != b_char: return False

    return True



class BasicSolver(Solver):
    """
    The most obvious possible strategy:

    Iterate over the slots in order, writing (in ink) our best guess that is compatible with the current contents of the slot
    Repeat until either grid is filled or we get stuck

    Uses the `BasicGuesser`
    """

    guesser_class: Type[Guesser] = BasicGuesser

    def solve(self, puzzle: Puzzle) -> bool:
        stuck = False
        while not puzzle.grid_filled() and not stuck:
            stuck = True
            for ident in puzzle.get_identifiers():
                current_slot = puzzle.read_slot(ident)
                if " " not in current_slot: continue

                clue = puzzle.get_clue(ident)
                gs = self.guesser.guess(clue, puzzle.read_slot(ident), max_guesses=5)

                for g, conf in gs:
                    if compatible(current_slot, g):
                        puzzle.write_slot(ident, g)
                        stuck = False
                        self.print_update_animation_frame(puzzle, g, ident, conf)
                        # time.sleep(0.5)
                        break
        
        return not stuck



class BasicSolverThreshold(Solver):
    """
    A variant of `BasicSolver`:

    Only fill in a slot if the guess confidence is above a threshold, which decreases over with time.

    Run until threshold hits a minimum degeneracy point (say, 5% confidence)
    """

    guesser_class: Type[Guesser] = BasicGuesser

    def solve(self, puzzle: Puzzle) -> bool:
        threshold = 0.75    # on the first pass, only fill in those that we are quite confident in

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
                        self.print_update_animation_frame(puzzle, g, ident, conf)
                        # time.sleep(0.5)
                        break
            
            threshold *= 0.5    # exponential decay
        
        return not stuck



"""
Some thoughts on a general strategy:
 - iteratively fill slots, using some heuristic to choose which one to guess next
 - heuristic might be a combination of guesser confidence and number of slot-intersections
 - if we get stuck, backtrack up the tree and try again with the next best guess

pseudocode:

def solve(puzzle):
    if puzzle is solved: return puzzle

    best_slot = puzzle.heuristic()
    clue = puzzle.clues(best_slot)
    context = puzzle.contexts(best_slot)
    for g in guess(clue, context):
        res = solve(puzzle with g in best_slot)
        if res is not STUCK: return res
    
    return STUCK
"""