import os
import json
from pprint import pprint
from typing import Dict, List, Tuple

from helpers import build_grid, pprint_grid

DATA_PATH = "CMSC470/crossword_puzzles/data/nyt_crosswords-cleaned"



class Puzzle:
    """
    An interface for easily interacting with a puzzle.

    NOTE: the provided "answers" do not necessarily match the populated slots given in the "grid" exactly and should be ignored
    """
    def __init__(self, puzzle_dict):
        self.puzzle_dict = puzzle_dict
        self.width, self.height = puzzle_dict["size"]["cols"], puzzle_dict["size"]["rows"]
        self.clues_map: Dict[str, str] = {}
        self.answers_map: Dict[str, str] = {}
        self.cells_map: Dict[str, List[Tuple[int, int]]] = {}

        self.solved_grid = None
        self.build_solved_grid()

        self.grid = [[cell if cell == "." else " " for cell in row] for row in self.solved_grid]

        # self.build_clues_answers_map()
        self.build_clues_map()
        self.build_cells_map()
        self.build_answers_map()

        assert len(self.clues_map) == len(self.answers_map)
        assert len(self.clues_map) == len(self.cells_map)

        for ident, cells in self.cells_map.items():
            assert len(self.answers_map[ident]) == len(cells)
        
    def build_solved_grid(self):
        self.solved_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        for i, cell in enumerate(self.puzzle_dict["grid"]):
            y, x = divmod(i, self.width)
            self.solved_grid[y][x] = cell

    # def build_clues_answers_map(self):
    #     """build map: ident -> clue and map: ident -> answer"""
    #     for clue, answer in zip(self.puzzle_dict["clues"]["across"], self.puzzle_dict["answers"]["across"]):
    #         n, c = clue.split(". ", 1)
    #         self.clues_map[f"{n}A"] = c
    #         self.answers_map[f"{n}A"] = answer

    #     for clue, answer in zip(self.puzzle_dict["clues"]["down"], self.puzzle_dict["answers"]["down"]):
    #         n, c = clue.split(". ", 1)
    #         self.clues_map[f"{n}D"] = c
    #         self.answers_map[f"{n}D"] = answer

    def build_clues_map(self):
        """build map: ident -> clue"""
        for clue in self.puzzle_dict["clues"]["across"]:
            n, c = clue.split(". ", 1)
            self.clues_map[f"{n}A"] = c

        for clue in self.puzzle_dict["clues"]["down"]:
            n, c = clue.split(". ", 1)
            self.clues_map[f"{n}D"] = c

    def build_cells_map(self):
        """build map: ident -> cells"""
        nums_grid = [
            [self.puzzle_dict["gridnums"][self.width * y + x] for x in range(self.width)]
            for y in range(self.height)
        ]

        # across
        for y in range(self.height):
            seq = []
            for x in range(self.width + 1):
                if x == self.width or self.grid[y][x] == ".":
                    if seq:
                        start = nums_grid[seq[0][1]][seq[0][0]]
                        self.cells_map[f"{start}A"] = seq
                        seq = []
                else:
                    seq.append((x, y))

        # down
        for x in range(self.width):
            seq = []
            for y in range(self.height + 1):
                if y == self.height or self.grid[y][x] == ".":
                    if seq:
                        start = nums_grid[seq[0][1]][seq[0][0]]
                        self.cells_map[f"{start}D"] = seq
                        seq = []
                else:
                    seq.append((x, y))

    def build_answers_map(self):
        for ident, coords in self.cells_map.items():
            self.answers_map[ident] = "".join(self.solved_grid[y][x] for x, y in coords)

    def get_clue(self, ident):
        "get the clue for the slot with the given identifier (e.g. 7D or 41A)"
        return self.clues_map[ident]
    
    def get_answer(self, ident):
        "get the answer for the slot with the given identifier (e.g. 7D or 41A)"
        return self.answers_map[ident]
    
    def read_slot(self, ident):
        "get the contents of the slot with the given identifier (e.g. 7D or 41A)"
        coords = self.cells_map[ident]
        return "".join(self.grid[y][x] for x, y in coords)
    
    def write_slot(self, ident, string):
        "write to the slot with the given identifier (e.g. 7D or 41A)"
        coords = self.cells_map[ident]
        if len(string) != len(coords):
            raise ValueError(f"Cannot write string of length {len(string)} (\"{string}\") to slot of length {len(coords)} ({ident})")
        for char, (x, y) in zip(string, coords):
            self.grid[y][x] = char
    
    def grid_filled(self):
        "determine if the current grid is completely filled"
        return all(" " not in row for row in self.grid)


if __name__ == "__main__":
    with open(os.path.join(DATA_PATH, "2014", "01", "01.json")) as f:
        puzzle_dict = json.load(f)
    
    pprint(puzzle_dict)

    puzzle = Puzzle(puzzle_dict)
    pprint(puzzle.cells_map)
    pprint_grid(puzzle.solved_grid)

    print(puzzle.get_clue("18A"))

    # print()

    # print(repr(puzzle.read_slot("2D")))
    # puzzle.write_slot("2D", "ABCDEFG")
    # pprint_grid(puzzle.grid)
    # print(puzzle.read_slot("2D"))

    # print(repr(puzzle.read_slot("1A")))