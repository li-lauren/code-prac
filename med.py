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

