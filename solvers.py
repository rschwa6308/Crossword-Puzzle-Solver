from typing import Type
from guessers import BasicGuesser, Guesser, HybridGuesser
from puzzle import Puzzle
from helpers import pprint_confidence_grid, pprint_grid, clear_console, EPSILON
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



def compatibility_score(a, b):
    if len(a) != len(b): return 0.0

    matches = 0
    for a_char, b_char in zip(a, b):
        if a_char == " " or b_char == " ":
            matches += 1
        if a_char == b_char:
            matches += 1
    
    return matches / len(a)


def compatible(a, b):
    return compatibility_score(a, b) == 1.0
    # if len(a) != len(b): return False

    # for a_char, b_char in zip(a, b):
    #     if a_char == " " or b_char == " ":
    #         continue
    #     if a_char != b_char: return False

    # return True



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
                        puzzle.write_slot(ident, g)
                        stuck = False
                        self.print_update_animation_frame(puzzle, g, ident, conf)
                        # time.sleep(0.5)
                        break



class BasicSolverThreshold(Solver):
    """
    A variant of `BasicSolver`:

    Only fill in a slot if the guess confidence is above a threshold, which decreases over with time.

    Run until threshold hits a minimum degeneracy point (say, 5% confidence)

    Uses the `HybridGuesser`
    """

    guesser_class: Type[Guesser] = HybridGuesser

    def solve(self, puzzle: Puzzle):
        conf_threshold = 0.75       # on the first pass, only fill in those that we are quite confident in

        while not puzzle.grid_filled() and conf_threshold >= 0.05:
            for ident in puzzle.get_identifiers():
                current_slot = puzzle.read_slot(ident)
                if " " not in current_slot: continue

                clue = puzzle.get_clue(ident)
                gs = self.guesser.guess(clue, puzzle.read_slot(ident), max_guesses=5)

                for g, conf in gs:
                    if compatible(current_slot, g) and conf >= conf_threshold:
                        puzzle.write_slot(ident, g)
                        stuck = False
                        self.print_update_animation_frame(puzzle, g, ident, conf)
                        # time.sleep(0.5)
                        break
            
            conf_threshold *= 0.5   # exponential decay
        


def average(nums):
    nums = list(nums)
    if len(nums) == 0: return 0
    return sum(nums) / len(nums)



class CellConfidenceSolver(Solver):
    """
    A first attempt at solving "in pencil".
    
    Each filled cell has associated confidence score (derived from guess confidence).
    Low confidence cells can be overwritten by subsequent guesses.

    WIP
    """
    guesser_class: Type[Guesser] = HybridGuesser

    def print_update_animation_frame(self, puzzle, guess, ident, confidence, clear=True):
        super().print_update_animation_frame(puzzle, guess, ident, confidence, clear)
        pprint_confidence_grid(self.confidence_grid)

    def solve(self, puzzle: Puzzle):
        # TODO: initialize confidence grid
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
                # if " " not in current_slot: continue

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
                        # compatibility_score(g, current_slot) > 0.25,
                        conf > conf_threshold,
                        conf > slot_confidence_avg_changed + EPSILON,
                    ]):
                        # transfer guess confidence to cell confidence
                        for (x, y), old, new in zip(slot_coords, current_slot, g):
                            if old != new:
                                self.confidence_grid[y][x] = conf                               # cell changed
                            else:
                                old_conf = self.confidence_grid[y][x]
                                self.confidence_grid[y][x] = 1 - (1 - old_conf)*(1 - conf)      # cell corroborated
                        
                        # overwrite contents of slot with the new guess
                        puzzle.write_slot(ident, g)
                        
                        converged = False

                        # print("old:", slot_confidence_avg, "new: ", conf)
                        self.print_update_animation_frame(puzzle, g, ident, conf)
                        break
            
            conf_threshold *= 0.5   # exponential decay



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