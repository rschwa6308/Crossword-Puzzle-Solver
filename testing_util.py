import os
import random
import json
import time
from typing import Callable, Type

from puzzle import Puzzle
from solvers import Solver



DATA_PATH = "data/nyt_crosswords-cleaned"


def get_puzzle_file_paths(data_path):
    with open(os.path.join(data_path, "manifest.txt")) as f:
        paths = [l.strip() for l in f.readlines()]
    
    return paths


# only test on puzzles from ODD days (EVEN days reserved for training)
puzzle_paths = [puzz for puzz in get_puzzle_file_paths(DATA_PATH) if int(puzz[-7:-5]) % 2 == 1]



def test_solver(solver_class: Type[Solver], num_puzzles, puzzle_filter:Callable[[Puzzle], bool]=lambda p: True, seed=None):
    """
    Run the given solver on the specified number of random puzzles.

    Return a dict with relevant statistics
    """
    solver = solver_class()

    fill_percentages = []
    fill_accuracies = []
    scores = []

    if seed:
        random.seed(seed)

    # temp_pat = ["Monday", "Wednesday", "Sunday"]

    trials_complete = 0
    while trials_complete < num_puzzles:
        puzzle_path = os.path.join(DATA_PATH, random.choice(puzzle_paths))

        try:
            with open(puzzle_path) as f:
                puzzle_dict = json.load(f)
            puzzle = Puzzle(puzzle_dict)
        except Exception as e:
            # print(f"An error occurred while building Puzzle object: {e}")
            # print("Skipping")
            continue
        
        if not puzzle_filter(puzzle):
            continue
            
        # # TEMPORARY
        # if puzzle_dict["dow"] != temp_pat[trials_complete]:
        #     continue 

        solver.solve(puzzle)

        fill_per = puzzle.grid_fill_percentage()
        fill_acc = puzzle.grid_accuracy()

        print(f"Fill percentage:   {fill_per:.0%}")
        print(f"Fill accuracy:     {fill_acc:.0%}")
        print()

        time.sleep(1)

        fill_percentages.append(fill_per)
        fill_accuracies.append(fill_acc)
        scores.append(fill_per * fill_acc)

        trials_complete += 1

    return {
        "solver": solver_class,
        "num_puzzles": num_puzzles,
        "average_fill_percentage": sum(fill_percentages) / num_puzzles,
        "average_fill_accuracy": sum(fill_accuracies) / num_puzzles,
        "average_score": sum(scores) / num_puzzles,
        "raw": {
            "fill_percentages": fill_percentages,
            "fill_accuracies": fill_accuracies,
            "scores": scores,
        }
    }

