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

# Josephus Survivors

def find_survivor(num_people, kill_every):
    """Given num_people in circle, kill [kill_every]th person, return survivor."""

    class Node:
        """Doubly-linked Node."""
        def __init__(self, val, prev=None, next=None):
            self.val = val
            self.prev = prev
            self.next = next

        def remove(self):
            self.prev.next = self.next
            self.next.prev = self.prev

    head = Node(1)
    last = Node(num_people)
    head.prev = last
    last.next = head

    prev_node = head
    for i in range(2, num_people): 
        curr_node = Node(i, prev_node)
        prev_node.next = curr_node
        prev_node = curr_node

    prev_node.next = last

    node = head
    while node != node.next:
        for i in range(kill_every - 1):
            node = node.next
        node.remove()
        node = node.next
    return node.val

print(find_survivor(10,3))

# Largest Subseq Sum
def largest_sum(nums):
    """Find subsequence with largest sum."""

    best_sum = 0
    curr_sum = 0
    start_i = 0
    best_slice = None

    for i, num in enumerate(nums):
        if curr_sum + num < 0:
            curr_sum = 0
            best_slice = (start_i, i)
            start_i = i+1
        else:
            curr_sum += num
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_slice = (start_i, i+1)

    return nums[best_slice[0]:best_slice[1]]

print(largest_sum([1, 0, 3, -8, 4, -2, 3]))


# Leveret lunch

def lunch_count(garden):
    """Given a garden of nrows of ncols, return carrots eaten."""

    # Sanity check that garden is valid

    row_lens = [len(row) for row in garden]
    assert min(row_lens) == max(row_lens), "Garden not a matrix!"
    assert all(all(type(c) is int for c in row) for row in garden), \
        "Garden values must be ints!"

    # Get number of rows and columns

    nrows = len(garden)
    ncols = len(garden[0])

    def get_starting_pos(n):
        mids = []

        if n % 2 == 1:
            mids.append(n // 2)
        else:
            mids.append(int(n/ 2))
            mids.append(int(n / 2 - 1))

        return mids

    mid_rows = get_starting_pos(nrows)
    mid_cols = get_starting_pos(ncols)

    # get starting index
    curr_max = 0
    max_row = None
    max_col = None
    for i in mid_rows:
        for j in mid_cols:
            if garden[i][j] > curr_max:
                curr_max = garden[i][j]
                max_row = i
                max_col = j

    total = curr_max
    garden[max_row][max_col] = 0

    # get next pos
    def get_next_pos(pos_i, pos_j):
        curr_max = 0
        max_i = None
        max_j = None

        def update_max(i, j):
            nonlocal curr_max
            nonlocal max_i
            nonlocal max_j
            if garden[i][j] > curr_max:
                curr_max = garden[i][j]
                max_i = i
                max_j = j
        
        if pos_j - 1 >= 0:
            update_max(pos_i, pos_j - 1)
        if pos_i - 1 >= 0:
            update_max(pos_i - 1, pos_j)
        if pos_j + 1 < ncols:
            update_max(pos_i, pos_j + 1)
        if pos_i + 1 < nrows:
            update_max(pos_i + 1, pos_j)

        if curr_max == 0:
            return None
        else:
            garden[max_i][max_j] = 0
            return (curr_max, max_i, max_j)

    while True:
        y = get_next_pos(max_row, max_col)
        
        if y:
            total += y[0]
            max_row = y[1]
            max_col = y[2]
        else:
            return total


garden = [
    [2, 3, 1, 4, 2, 2, 3],
    [2, 3, 0, 4, 0, 3, 0],
    [1, 7, 0, 2, 1, 2, 3],
    [9, 3, 0, 4, 2, 0, 3],
]

print(lunch_count(garden))


# Max Path Triangle

class Node(object):
    """Basic node class that keeps track fo parents and children.

    This allows for multiple parents---so this isn't for trees, where
    nodes can only have one children. It is for "directed graphs".
    """

    def __init__(self, value):
        self.value = value
        self.children = []
        self.parents = []

    def __repr__(self):
        return str(self.value)


# Merge Sort
def merge_sort(lst):

    if len(lst) > 1:
        mid = len(lst) // 2
        left = lst[:mid]
        right = lst[mid:]

        merge_sort(left)
        merge_sort(right)

        left_i = right_i = new_i = 0

        while left_i < len(left) and right_i < len(right):
            if left[left_i] < right[right_i]:
                lst[new_i] = left[left_i]
                left_i += 1
            else:
                lst[new_i] = right[right_i]
                right_i += 1
            new_i += 1

        while left_i < len(left):
            lst[new_i] = left[left_i]
            left_i += 1
            new_i += 1

        while right_i < len(right):
            lst[new_i] = right[right_i]
            right_i += 1
            new_i += 1

    return lst

lst = [3,2,6,4,7,8,5]

print(merge_sort(lst))

# Word Break

def parse(phrase, vocab, cache=None):
    """Break a string into words.

    Input:
        - string of words without space breaks
        - vocabulary (set of allowed words)

    Output:
        set of all possible ways to break this down, given a vocab
    """

    if cache is None:
        cache = {}

    if phrase in cache:
        return cache[phrase]
    
    if not phrase:
        return []

    potential_sentences = []
    
    found_match = False
    for i in range(len(phrase)):
        chs = phrase[:i+1]
        if chs in vocab:
            res = parse(phrase[i+1:], vocab)
            if not res:
                potential_sentences.extend([chs])
            else:
                potential_sentences.extend([chs + " " + x for x in res])
            found_match = True

    if not found_match:
        return ["0"]

    ans = []

    for sentence in potential_sentences:
        if sentence[-1] != "0":
            ans.append(sentence)
    cache[phrase] = ans
    return ans

vocab = {'i', 'a', 'ten', 'oodles', 'ford', 'inner', 'to',
 'night', 'ate', 'noodles', 'for', 'dinner', 'tonight'}

sentences = parse('iatenoodlesfordinnertonight', vocab)

for s in sorted(sentences):
    print(s)


# Tree cousins
class Node(object):
    """Doubly-linked node in a tree.

        >>> na = Node("na")
        >>> nb1 = Node("nb1")
        >>> nb2 = Node("nb2")

        >>> nb1.set_parent(na)
        >>> nb2.set_parent(na)

        >>> na.children
        [<Node nb1>, <Node nb2>]

        >>> nb1.parent
        <Node na>
    """

    parent = None

    def __init__(self, data):
        self.children = []
        self.data = data

    def __repr__(self):
        return "<Node %s>" % self.data

    def set_parent(self, parent):
        """Set parent of this node.

        Also sets the children of the parent to include this node.
        """

        self.parent = parent
        parent.children.append(self)

    def cousins(self):
        """Find nodes on the same level as this node."""
        curr = self
        level = 0

        while curr.parent:
            curr = curr.parent
            level += 1

        cousins = set()

        def _cousins(node, curr_depth):
            if not node:
                return 

            if curr_depth == level:
                cousins.add(node)
                return
            
            for child in node.children:
                _cousins(child, curr_depth + 1)

        _cousins(curr, 0)

        cousins.remove(self)

        return cousins