import os
import json
from pprint import pprint

from puzzle import Puzzle
from helpers import pprint_grid


DATA_PATH = "CMSC470/crossword_puzzles/data/nyt_crosswords-cleaned"


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

def solve_puzzle(puzzle):
    pass

if __name__ == "__main__":
    with open(os.path.join(DATA_PATH, "2010", "01", "01.json")) as f:
        puzzle_dict = json.load(f)
    
    pprint(puzzle_dict)

    puzzle = Puzzle(puzzle_dict)
    pprint_grid(puzzle.solved_grid)

    solve_puzzle(puzzle)
