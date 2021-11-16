import os
import termcolor


def build_grid(puzzle):
    w = puzzle["size"]["cols"]
    h = puzzle["size"]["rows"]

    grid = [[None for _ in range(w)] for _ in range(h)]
    for i, cell in enumerate(puzzle["grid"]):
        y, x = divmod(i, w)
        grid[y][x] = cell
    
    return grid



WIDE_MAP = {i: i + 0xFEE0 for i in range(0x21, 0x7F)}
WIDE_MAP[0x20] = 0x3000

def char_map(c):
    if c.isalnum():
        return c.translate(WIDE_MAP)
    elif c == ".":
        return "██"
    elif c == " ":
        return "＿"
    
    return "??"


def pprint_grid(grid, accuracy_grid=None, incorrect_color="red"):
    """
    pretty-print the given grid using unicode block and double-wide chars

    if accuracy_grid is supplied, coloring incorrect cells
    
    """
    # h, w = len(grid), len(grid[0])
    
    if not accuracy_grid:
        for row in grid:
            chars = map(char_map, row)
            print("".join(chars))
    else:
        for row, acc_row in zip(grid, accuracy_grid):
            chars = map(char_map, row)
            print("".join(
                char if correct!=False else termcolor.colored(char, incorrect_color)
                for char, correct in zip(chars, acc_row)
            ))
    



def vert_cell_sep(a, b):
    if a == "." and b == ".":
        return "█"
    elif a == ".":
        return "▌"
    elif b == ".":
        return "▐"
    else:
        return "│"


def hor_cell_sep(a, b):
    if a == "." and b == ".":
        return "██"
    elif a == ".":
        return "▀▀"
    elif b == ".":
        return "▄▄"
    else:
        return "──"



def pprint_grid_complex(grid):
    h, w = len(grid), len(grid[0])

    print("┌─" + "─┬─"*(w-1) + "─┐")

    line_sep = "\n├─" + "─┼─"*(w-1) + "─┤\n"
    line_sep = "\n"

    for y in range(h):
        row = grid[y]
        cell_chars = [char_map(c) for c in row]
        line = vert_cell_sep("A", row[0])
        for x in range(w):
            line += cell_chars[x]
            if x < w-1:
                line += vert_cell_sep(row[x], row[x+1])
        line += vert_cell_sep(row[-1], "A")
        print(line)
        if y < h - 1:
            line_sep = "├" + "┼".join(
                hor_cell_sep(grid[y][x], grid[y+1][x])
                for x in range(w)
            ) + "┤"
            print(line_sep)
    print("└─" + "─┴─"*(w-1) + "─┘")



from IPython.display import clear_output


def clear_console():
    clear_output(wait=True)                             # jupyter notebook
    os.system('cls' if os.name=='nt' else 'clear')      # windows or linux
