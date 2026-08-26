import sys
from crossword import *
from PIL import Image, ImageDraw, ImageFont

class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` to make each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
            constraints: the length of the word)
        """
        for var, words in self.domains.items():
            self.domains[var] = {word for word in words if len(word) == var.length}

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        removed values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        if y not in self.crossword.neighbors(x):
            return False

        first_word_letter , second_word_letter = self.crossword.overlaps[x, y]

        revised = False
        words_to_remove = set()
        for x_word in self.domains[x]:
            there_is_an_option = False
            for y_word in self.domains[y]:
                if x_word[first_word_letter] == y_word[second_word_letter]:
                    there_is_an_option = True
                    break
            if not there_is_an_option:
                words_to_remove.add(x_word)
                revised = True

        if revised:
            self.domains[x] -= words_to_remove

        return revised

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        queue = set()
        if arcs is None:
            for key in self.domains.keys():
                for neighbor in self.crossword.neighbors(key):
                    queue.add((key, neighbor))
        else:
            queue = set(arcs)
        
        while queue:
            (x,y) = queue.pop()
            if self.revise(x,y):
                if not self.domains[x]:
                    return False
                for z in (self.crossword.neighbors(x) - {y}):
                    queue.add((z, x))

        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        if assignment.keys() != self.crossword.variables:
            return False

        for variable in self.crossword.variables:
            if not assignment[variable]:
                return False

        return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        words = list(assignment.values())
        if len(words) != len(set(words)):
            return False
        
        for var in assignment.keys():

            if len(assignment[var]) != var.length:
                return False

            for neighbor in self.crossword.neighbors(var):
                if neighbor not in assignment:
                    continue

                i, j = self.crossword.overlaps[var, neighbor]
                if assignment[var][i] != assignment[neighbor][j]:
                    return False

        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        values = self.domains[var]
        values_constrains = {value: 0 for value in values}
        
        for neighbor in self.crossword.neighbors(var):
            if neighbor in assignment:
                continue

            i, j = self.crossword.overlaps[var, neighbor]

            for value in values:
                for neighbor_value in self.domains[neighbor]:
                    if value[i] != neighbor_value[j]:
                        values_constrains[value] += 1

        sorted_values_constrains = sorted(values_constrains.keys(), key=lambda val: values_constrains[val])

        return sorted_values_constrains

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        unassigned_variables = self.crossword.variables - set(assignment.keys())

        min_len = min(len(self.domains[k]) for k in unassigned_variables)
        min_vars = [k for k in unassigned_variables if len(self.domains[k]) == min_len]

        if len(min_vars) == 1:
            return min_vars[0]

        max_degree = max(len(self.crossword.neighbors(k)) for k in min_vars)
        best_vars = [k for k in min_vars if len(self.crossword.neighbors(k)) == max_degree]

        return best_vars[0]

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)

        for value in self.order_domain_values(var, assignment):

            assignment[var] = value
            if self.consistent(assignment):

                result = self.backtrack(assignment)
                if result:
                    return result
                
            del assignment[var]

        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
