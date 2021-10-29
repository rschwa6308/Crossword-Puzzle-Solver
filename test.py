import os
import json

DATA_PATH = "CMSC470/crossword_puzzles/data/nyt_crosswords-cleaned"



from helpers import build_grid, pprint_grid


if __name__ == "__main__":
    with open(os.path.join(DATA_PATH, "2010", "01", "01.json")) as f:
        puzzle = json.load(f)
    
    # pprint(puzzle)

    grid = build_grid(puzzle)
    # grid[2][2] = " "
    pprint_grid(grid)