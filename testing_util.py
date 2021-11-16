import os
import random
import json
from typing import Type

from puzzle import Puzzle
from solvers import Solver



DATA_PATH = "data/nyt_crosswords-cleaned"


def get_puzzle_file_paths(data_path):
    with open(os.path.join(data_path, "manifest.txt")) as f:
        paths = [l.strip() for l in f.readlines()]
    
    return paths


# only test on puzzles from ODD days (EVEN days reserved for training)
puzzle_paths = [puzz for puzz in get_puzzle_file_paths(DATA_PATH) if int(puzz[-7:-5]) % 2 == 1]



def test_solver(solver_class: Type[Solver], num_puzzles):
    """
    Run the given solver on the specified number of random puzzles.

    Return a dict with relevant statistics
    """
    solver = solver_class()

    fill_percentages = []
    fill_accuracies = []

    trials_complete = 0
    while trials_complete < num_puzzles:
        puzzle_path = os.path.join(DATA_PATH, random.choice(puzzle_paths))

        print(f"Testing {solver_class} on {puzzle_path}\n")

        with open(puzzle_path) as f:
            puzzle_dict = json.load(f)

        try:
            puzzle = Puzzle(puzzle_dict)
        except Exception as e:
            print(f"An error occurred while building Puzzle object: {e}")
            print("Skipping")
            continue

        solver.solve(puzzle)

        fill_per = puzzle.grid_fill_percentage()
        fill_acc = puzzle.grid_accuracy()

        print(f"Fill percentage:   {fill_per:.0%}")
        print(f"Fill accuracy:     {fill_acc:.0%}")

        fill_percentages.append(fill_per)
        fill_accuracies.append(fill_acc)

        trials_complete += 1

    return {
        "solver": solver_class,
        "num_puzzles": num_puzzles,
        "average_fill_percentage": sum(fill_percentages) / num_puzzles,
        "average_fill_accuracy": sum(fill_accuracies) / num_puzzles
    }

