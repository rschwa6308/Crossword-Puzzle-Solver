from pprint import pprint
import time
from guessers import BasicGuesser

from puzzle import Puzzle
from testing_util import get_puzzle_file_paths, test_solver
from solvers import BasicSolver, BasicSolverThreshold, CellConfidenceSolver



class CustomSolver(CellConfidenceSolver):
    """CellConfidenceSolver without n-gram searching"""
    guesser_class = BasicGuesser


SEED = 69

results = [
    test_solver(CellConfidenceSolver, 10, seed=SEED),
    test_solver(CustomSolver, 10, seed=SEED)
]

for res in results:
    pprint(res)

# res1 = test_solver(BasicSolver, 100)
# res2 = test_solver(BasicSolverThreshold, 100, seed=69)
# res3 = test_solver(CellConfidenceSolver, 100, seed=69)
# pprint(res1)
# pprint(res2)
# pprint(res3)


"""
{'average_fill_accuracy': 0.5136492346624022,
 'average_fill_percentage': 0.7937445334206289,
 'solver': <class 'solvers.BasicSolver'>}

{'average_fill_accuracy': 0.6062841239766793,
 'average_fill_percentage': 0.7452190900625086,
 'solver': <class 'solvers.BasicSolverThreshold'>}

{'average_fill_accuracy': 0.6149019741627505,
 'average_fill_percentage': 0.8845044841083854,
 'num_puzzles': 100,
 'solver': <class 'solvers.CellConfidenceSolver'>}
"""