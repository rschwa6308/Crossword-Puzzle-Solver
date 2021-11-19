import os
import sys
import json
import random
from pprint import pprint
import time

from puzzle import Puzzle
from testing_util import get_puzzle_file_paths, test_solver
from solvers import BasicSolver, BasicSolverThreshold, CellConfidenceSolver



# DATA_PATH = "data/nyt_crosswords-cleaned"


# # choose a random puzzle from an ODD day (EVEN days reserved for training)
# puzzle_paths = [puzz for puzz in get_puzzle_file_paths(DATA_PATH) if int(puzz[-7:-5]) % 2 == 1]
# puzzle_path = os.path.join(DATA_PATH, random.choice(puzzle_paths))

# print(f"Opening {puzzle_path}\n")
# with open(puzzle_path) as f:
#     puzzle_dict = json.load(f)


# # # a particular puzzle
# # with open(os.path.join(DATA_PATH, "2010", "01", "01.json")) as f:
# #     puzzle_dict = json.load(f)


# # pprint(puzzle_dict)

# puzzle = Puzzle(puzzle_dict)


# solver = BasicSolverThreshold()
# solver.solve(puzzle)


# print(f"Fill percentage:   {puzzle.grid_fill_percentage():.0%}")
# print(f"Fill accuracy:     {puzzle.grid_accuracy():.0%}")


# res1 = test_solver(BasicSolver, 100)
# res2 = test_solver(BasicSolverThreshold, 10)
res3 = test_solver(CellConfidenceSolver, 1)
# pprint(res1)
# pprint(res2)
pprint(res3)

"""
{'average_fill_accuracy': 0.5136492346624022,
 'average_fill_percentage': 0.7937445334206289,
 'solver': <class 'solvers.BasicSolver'>}
{'average_fill_accuracy': 0.6062841239766793,
 'average_fill_percentage': 0.7452190900625086,
 'solver': <class 'solvers.BasicSolverThreshold'>}
"""