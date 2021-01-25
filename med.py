# POLISH CALCULATOR

def calc(s):
    """Evaluate expression."""
    s_items = s.split(" ")
    operations_stack = []
    nums_stack = []

    ops = "-/+*"
    for item in s_items:
        if item in ops:
            operations_stack.append(item)
        else:
            nums_stack.append(int(item))

    while len(nums_stack) > 1:
        second_term = nums_stack.pop()
        first_term = nums_stack.pop()

        op = operations_stack.pop()
        
        if op == '+':
            ans = first_term + second_term
        elif op == '-':
            ans = first_term - second_term
        elif op == '*':
            ans = first_term * second_term
        else:
            ans = first_term / second_term

        nums_stack.append(ans)

    return nums_stack[0]

print(calc("/ 6 - 4 2"))


# CHECK DETECTION

def check(king, queen):
    """Given a chessboard with one K and one Q, see if the K can attack the Q.

    This function is given coordinates for the king and queen on a chessboard.
    These coordinates are given as a letter A-H for the columns and 1-8 for the
    row, like "D6" and "B7":
    """
    k_col = king[0]
    k_row = int(king[1])

    q_col = queen[0]
    q_row = int(queen[1])

    # Same row or col check
    if q_col == k_col or q_row == k_row:
        return True

    cols = {}
    i = 1
    for letter in "ABCDEFGH":
        cols[letter] = i
        i += 1

    # Diagonal Check
    return abs(cols[q_col] - cols[k_col]) == abs(q_row - k_row)

print(check("D6", "H7"))