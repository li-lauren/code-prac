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


# COINS

def coins(num_coins):
    """Find change from combinations of `num_coins` of dimes and pennies.

    This should return a set of the unique amounts of change possible.
    """
    ans = set()
    for i in range(num_coins):
        ans.add(i * 10 + (num_coins - i) * 1)

    ans.add(num_coins * 10)

    return ans

print(coins(4) == {4, 13, 22, 31, 40})

def coins_recursive(num_coins):
    def add_coins(coins_left, total, results):
        if coins_left == 0:
            results.add(total)
            return
        add_coins(coins_left - 1, 10 + total, results)
        add_coins(coins_left - 1, 1 + total, results)
    results = set()

    add_coins(num_coins, 0, results)

    return results

print(coins_recursive(4) == {4, 13, 22, 31, 40})


# Compress String

def compress_string(s):
    curr = s[0]
    ans = s[0]
    count = 1

    for ch in s[1:]:
        if ch == curr:
            count += 1
        else:
            if count > 1:
                ans += str(count)
            count = 1
            curr = ch
            ans += ch

    # account for if last character is repeated        
    if count > 1:
        ans += str(count)

    return ans

print(compress_string('balloonicorn'))


# Count employees
class Node(object):
    """Node in a tree."""

    def __init__(self, name, children=None):
        self.name = name
        self.children = children or []

    def count_employees(self):
        """Return a count of how many employees this person manages.

        Return a count of how many people that manager manages. This should
        include *everyone* under them, not just people who directly report to
        them.
        """

        if not self.children:
            return 0
        else:
            return len(self.children) + sum([child.count_employees() for child in self.children])


# Dec to Binary
def dec2bin(num):
    """Convert a decimal number to binary representation."""

    bin = ""
    while num >= 1:
        bin = str(num % 2) + bin
        num = num // 2
    
    return bin

print(dec2bin(13))

# Check Friends
class PersonNode(object):
    """A node in a graph representing a person.

    This is created with a name and, optionally, a list of adjacent nodes.
    """

    def __init__(self, name, adjacent=[]):
        self.name = name
        self.adjacent = set(adjacent)

    def __repr__(self):
        return "<PersonNode %s>" % self.name

class FriendGraph(object):
    """Graph to keep track of social connections."""

    def __init__(self):
        """Create an empty graph.

        We keep a dictionary to map people's names -> nodes.
        """

        self.nodes = {}

    def add_person(self, name):
        """Add a person to our graph.

            >>> f = FriendGraph()
            >>> f.nodes
            {}

            >>> f.add_person("Dumbledore")
            >>> f.nodes
            {'Dumbledore': <PersonNode Dumbledore>}
        """

        if name not in self.nodes:
            # Be careful not to just add them a second time -- otherwise,
            # if we accidentally added someone twice, we'd clear our their list
            # of friends!
            self.nodes[name] = PersonNode(name)

    def set_friends(self, name, friend_names):
        """Set two people as friends.

        This is reciprocal: so if Romeo is friends with Juliet, she's
        friends with Romeo (our graph is "undirected").

        >>> f = FriendGraph()
        >>> f.add_person("Romeo")
        >>> f.add_person("Juliet")
        >>> f.set_friends("Romeo", ["Juliet"])

        >>> f.nodes["Romeo"].adjacent
        {<PersonNode Juliet>}

        >>> f.nodes["Juliet"].adjacent
        {<PersonNode Romeo>}
        """

        person = self.nodes[name]

        for friend_name in friend_names:
            friend = self.nodes[friend_name]

            # Since adjacent is a set, we don't care if we're adding duplicates ---
            # it will only keep track of each relationship once. We do want to
            # make sure that we're adding both directions for the relationship.
            person.adjacent.add(friend)
            friend.adjacent.add(person)

    def are_connected(self, name1, name2):
        """Is this name1 friends with name2?"""

        seen = set()
        to_visit = []

        friend1 = self.nodes[name1]
        friend2 = self.nodes[name2]

        for node in friend1.adjacent:
            to_visit.append(node)

        while to_visit:
            person = to_visit.pop(0)
            if person is not friend2:
                seen.add(person)
                for per in person.adjacent:
                    if per not in seen:
                        to_visit.append(per)
            else:
                return True

        return False

f = FriendGraph()
f.add_person("Frodo")
f.add_person("Sam")
f.add_person("Gandalf")
f.add_person("Merry")
f.add_person("Pippin")
f.add_person("Treebeard")
f.add_person("Sauron")
f.add_person("Dick Cheney")

f.set_friends("Frodo", ["Sam", "Gandalf", "Merry", "Pippin"])
f.set_friends("Sam", ["Merry", "Pippin", "Gandalf"])
f.set_friends("Merry", ["Pippin", "Treebeard"])
f.set_friends("Pippin", ["Treebeard"])
f.set_friends("Sauron", ["Dick Cheney"])

print(f.are_connected("Frodo", "Sam"))

# Hexadecimal Conversion

def hex_convert(hex_in):
    """Convert a hexadecimal string, like '1A', into it's decimal equivalent."""

    hex_vals = {}

    for num in range(0,10):
        hex_vals[str(num)] = num
    i = 10
    for ch in "ABCDEF":
        hex_vals[ch] = i
        i += 1

    ans = 0
    L = len(hex_in)
    for i in range(L):
        ans += hex_vals[hex_in[L-i-1]] * 16**i

    return ans

print(hex_convert('FFFF'))

# Insertion Sort
def insertion_sort(lst):
    for i in range(1,len(lst)):
        j = 1
        curr = lst[i]
        while curr < lst[i-j] and i - j >= 0:
            lst[i-j+1] = lst[i-j]
            lst[i-j] = curr
            j += 1
    
    return lst

print(insertion_sort([2,4,3,1]))
