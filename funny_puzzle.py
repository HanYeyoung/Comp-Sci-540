import heapq

def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader.

    INPUT:
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """

    distance = 0
    size = 3  # 3x3 grid

    for tile in range(1, 8):  # iterate through tile 1 to 7
        if tile in from_state:
            current_pos = from_state.index(tile)
            target_pos = to_state.index(tile)
            current_row, current_col = divmod(current_pos, size)
            target_row, target_col = divmod(target_pos, size)
            distance += abs(current_row - target_row) + abs(current_col - target_col)

    return distance

def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT:
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle.
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


def get_succ(state):
    """
    TODO: implement this function.

    INPUT:
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below).
    """
    size = 3  # grid size (3x3)
    succ_states = []

    # helper function to swap tiles
    def swap(state, i, j):
        new_state = list(state)
        new_state[i], new_state[j] = new_state[j], new_state[i]
        return new_state

    # positions of the empty grids (zeros)
    empty_positions = [i for i, x in enumerate(state) if x == 0]

    # generate successors for each zeros
    for pos in empty_positions:
        row, col = divmod(pos, size)

        # possible moves: up, down, left, write
        if row > 0 and state[pos - size] != 0:              # move up (if not swapping with another zero)
            succ_states.append(swap(state, pos, pos - size))
        if row < size - 1 and state[pos + size] != 0:       # move down
            succ_states.append(swap(state, pos, pos + size))
        if col > 0 and state[pos - 1] != 0:                 # move left
            succ_states.append(swap(state, pos, pos - 1))
        if col < size - 1 and state[pos + 1] != 0:          # move right
            succ_states.append(swap(state, pos, pos + 1))

    # remove duplicates, sort the list of successors
    succ_states = [list(t) for t in set(tuple(s) for s in succ_states)]
    return sorted(succ_states)

def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT:
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """
    index_dict = dict() # dictionary to keep track of states and indexes

    curr = state[:] # starting state
    g = 0 # starting g value
    h = get_manhattan_distance(curr, goal_state) # starting h value
    cost = g + h # total cost at starting state

    pq = [] # priority queue (OPEN list)
    visited = [] # visited list (CLOSED list)

    # push the initial state onto the priority queue
    b = (cost, curr, (g, h, -1))
    heapq.heappush(pq, b)
    visited.append(curr)
    index_dict[0] = b
    max_queue = 1

    # main loop of the A* algorithm
    while (len(pq) != 0):
        # pop the state with the lowest cost
        b = heapq.heappop(pq)
        curr = b[1]
        g = b[2][0] + 1
        parent_index = visited.index(curr)

        # check if the current state is the goal state
        if curr == goal_state:
            # backtrack to reconstruct the path
            result = [b]
            parent_index = b[2][2]
            while parent_index != -1:
                b = index_dict[parent_index]
                parent_index = b[2][2]
                result.insert(0, b)
            # print out the path
            for b in result:
                print(f'{b[1]} h={b[2][1]} moves: {b[2][0]}')
            print(f'Max queue length: {max_queue}')
            return

        # generate successors of the current state
        moves = get_succ(curr)
        for succ in moves:
            # check if the successor state has been visited or needs updating
            if succ in visited:
                i = visited.index(succ)
                b = index_dict[i]
                if b[2][0] > g:
                    h = b[2][1]
                    cost = g + h
                    b = (cost, succ, (g, h, parent_index))
                    index_dict[i] = b
                    heapq.heappush(pq, b)
            else:
                # new state
                h = get_manhattan_distance(succ, goal_state)
                cost = g + h
                b = (cost, succ, (g, h, parent_index))
                index_dict[len(visited)] = b
                visited.append(succ)
                heapq.heappush(pq, b)
        # update maximum length of the priority queue
        max_queue = max(max_queue, len(pq))

if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    print_succ([2, 5, 1, 4, 0, 6, 7, 0, 3])
    print()
    solve([4, 3, 0, 5, 1, 6, 7, 2, 0])
    print()