from typing import Type
from guessers import BasicGuesser, Guesser
from puzzle import Puzzle
from helpers import pprint_confidence_grid, pprint_grid, clear_console
from pprint import pprint



class Solver:
    guesser_class: Type[Guesser] = None

    def __init__(self):
        self.guesser = self.guesser_class()
        self.guesser.load()

    def solve(self, puzzle: Puzzle):
        """
        Attempt to solve the given puzzle.
        Modifies the given puzzle in place.
        """



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

    def solve(self, puzzle: Puzzle):
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
                        clear_console()
                        print(f"\nWriting {g:^15} to slot {ident}\tconf: {conf:.0%}\t", end="")
                        print("(CORRECT ✓)\n" if g == puzzle.get_answer(ident) else "(INCORRECT ✗)\n")
                        puzzle.write_slot(ident, g)
                        stuck = False
                        pprint_grid(puzzle.grid, puzzle.get_acc_grid())
                        # print("\n"*5)
                        # time.sleep(0.5)
                        break



class BasicSolverThreshold(Solver):
    """
    A variant of `BasicSolver`:

    Only fill in a slot if the guess confidence is above a threshold, which decreases over with time.

    Run until threshold hits a minimum degeneracy point (say, 5% confidence)
    """

    guesser_class: Type[Guesser] = BasicGuesser

    def solve(self, puzzle: Puzzle):
        threshold = 0.75    # on the first pass, only fill in those that we are quite confident in

        while not puzzle.grid_filled() and threshold >= 0.05:
            for ident in puzzle.get_identifiers():
                current_slot = puzzle.read_slot(ident)
                if " " not in current_slot: continue

                clue = puzzle.get_clue(ident)
                gs = self.guesser.guess(clue, puzzle.read_slot(ident), max_guesses=5)

                for g, conf in gs:
                    if compatible(current_slot, g) and conf >= threshold:
                        clear_console()
                        print(f"\nWriting {g:^15} to slot {ident}\tconf: {conf:.0%}\t", end="")
                        print("(CORRECT ✓)\n" if g == puzzle.get_answer(ident) else "(INCORRECT ✗)\n")
                        puzzle.write_slot(ident, g)
                        pprint_grid(puzzle.grid, puzzle.get_acc_grid())
                        break
            
            threshold *= 0.5    # exponential decay
        


class CellConfidenceSolver(Solver):
    """
    A first attempt at solving "in pencil".
    
    Each filled cell has associated confidence score (derived from guess confidence).
    Low confidence cells can be overwritten by subsequent guesses.

    WIP
    """
    guesser_class: Type[Guesser] = BasicGuesser

    def solve(self, puzzle: Puzzle):
        # TODO: initialize confidence grid
        confidence_grid = [
            [None if cell == "." else 0.0 for cell in row]
            for row in puzzle.grid
        ]

        while not puzzle.grid_filled():
            for ident in puzzle.get_identifiers():
                current_slot = puzzle.read_slot(ident)
                slot_coords = puzzle.cells_map[ident]
                # if " " not in current_slot: continue

                clue = puzzle.get_clue(ident)
                gs = self.guesser.guess(clue, puzzle.read_slot(ident), max_guesses=5)

                for g, conf in gs:
                    # TODO: overwriting logic here
                    if compatible(current_slot, g):
                        clear_console()
                        print(f"\nWriting {g:^15} to slot {ident}\tconf: {conf:.0%}\t", end="")
                        print("(CORRECT ✓)\n" if g == puzzle.get_answer(ident) else "(INCORRECT ✗)\n")
                        puzzle.write_slot(ident, g)
                        # transfer guess confidence to cell confidence, optionally "normalizing" by slot length
                        for x, y in slot_coords:
                            # confidence_grid[y][x] = conf / len(slot_coords) ** 0.5
                            confidence_grid[y][x] = conf

                        pprint_grid(puzzle.grid, puzzle.get_acc_grid())
                        pprint_confidence_grid(confidence_grid)
                        break



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