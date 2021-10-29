import os
import json
import random
from pprint import pprint

from puzzle import Puzzle
from guesser import guess
from helpers import get_puzzle_file_paths, pprint_grid


DATA_PATH = "data/nyt_crosswords-cleaned"


"""
General Strategy:
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


def compatible(a, b):
    if len(a) != len(b): return False

    for a_char, b_char in zip(a, b):
        if a_char == " " or b_char == " ": continue
        if a_char != b_char: return False

    return True



def solve_puzzle(puzzle: Puzzle):
    """
    The most obvious possible strategy:

    Iterate over the slots in order, writing (in ink) our best guess that is compatible with the current contents of the slot
    Repeat until either grid is filled or we get stuck

    Return whether we succeeded in filling the grid or not
    """
    stuck = False
    while not puzzle.grid_filled() and not stuck:
        stuck = True
        for ident in puzzle.get_identifiers():
            current_slot = puzzle.read_slot(ident)
            if " " not in current_slot: continue

            clue = puzzle.get_clue(ident)
            gs = guess(clue, puzzle.get_slot_length(ident), max_guesses=5)
            # print(f"guesses for {ident}. {clue}: {gs}")

            for g in gs:
                if compatible(current_slot, g):
                    print(f"Writing {g:^15} to slot {ident}\t", end="")
                    if g == puzzle.get_answer(ident):
                        print("(CORRECT ✓)\n")
                    else:
                        print("(INCORRECT ✗)\n")
                    puzzle.write_slot(ident, g)
                    stuck = False
                    pprint_grid(puzzle.grid, puzzle.get_acc_grid())
                    print()
                    break
    
    return not stuck



if __name__ == "__main__":
    # choose a random puzzle from an ODD day (EVEN days used for training)
    puzzle_paths = [puzz for puzz in get_puzzle_file_paths(DATA_PATH) if int(puzz[-7:-5]) % 2 == 1]
    puzzle_path = os.path.join(DATA_PATH, random.choice(puzzle_paths))
    print(f"Opening {puzzle_path}\n")
    with open(puzzle_path) as f:
        puzzle_dict = json.load(f)


    # # a particular puzzle
    # with open(os.path.join(DATA_PATH, "2010", "01", "01.json")) as f:
    #     puzzle_dict = json.load(f)

    
    # pprint(puzzle_dict)

    puzzle = Puzzle(puzzle_dict)
    # pprint_grid(puzzle.solved_grid)

    res = solve_puzzle(puzzle)
    print(f"Filled grid:\t{res}")
    print(f"Fill accuracy:\t{puzzle.grid_accuracy():.0%}")
