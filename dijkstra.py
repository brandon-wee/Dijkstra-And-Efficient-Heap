class TreeNode:
    def __init__(self, tree_id: int) -> None:
        """
        Constructor method for the TreeNode class.

        Written by Brandon Wee Yong Jing

        Input:
            tree_id: Integer representing the ID of the tree
        Return:
            None

        Time complexity:
            Best case analysis: O(1)
            Initializes all the instance variables. Hence, it is O(1)
            Worst case analysis: O(1)
            Justification is exactly the same as best case complexity.

        Space complexity:
            Input space analysis: O(1)
            Integer is a constant space data type. Hence, it is O(1) input space complexity.

            Aux space analysis: O(1)
            Declaration of int, boolean, float and empty list is constant space.
            Hence, the auxiliary space complexity is O(1).
        """
        self.tree_id = tree_id
        self.roads = []
        self.solulu = False
        self.exit = False
        self.visited = False
        self.shortest_distance: Union[int, float] = float('inf')  # Note that the shortest distance is int, but float is the only datatype that can have 'inf'.
        self.previous_tree = None
        self.index = tree_id + 1

    def add_road(self, road: 'TreeRoad') -> None:
        """ This method adds a road to the self.road list.

            Written by Brandon Wee Yong Jing.

            pre: road: TreeRoad object representing an edge between this TreeNode and another TreeNode
            post: None

            time complexity: Best case and worst case is O(1).
            space complexity: Input: O(1), Auxiliary: O(1).
        """
        self.roads.append(road)

    def __lt__(self, other: 'TreeNode') -> bool:
        """ Magic method used to perform the lesser than comparison between two TreeNode objects.
            This is used in the MinHeap to directly compare objects by their shortest distance.

            Written by Brandon Wee Yong Jing.

            pre: other: TreeNode object to compare the shortest distance
            post: boolean representing that self has shorter distance than other.

            time complexity: Best case and worst case is O(1).
            space complexity: Input: O(1), Auxiliary: O(1).
        """
        return self.shortest_distance < other.shortest_distance

    def __ge__(self, other: 'TreeNode') -> bool:
        """ Magic method used to perform the greater equal comparison between two TreeNode objects.
            This is used in the MinHeap to directly compare objects by their shortest distance.

            Written by Brandon Wee Yong Jing.

            pre: other: TreeNode object to compare the shortest distance
            post: boolean representing that self has greater or equal distance than other.

            time complexity: Best case and worst case is O(1).
            space complexity: Input: O(1), Auxiliary: O(1).
        """
        return self.shortest_distance >= other.shortest_distance

    def __str__(self) -> str:
        """ To string method. Mainly used during debugging stage.

            Written by Brandon Wee Yong Jing.

            pre: None
            post: String representing some instance variable values.

            time complexity: Best case and worst case is O(1).
            space complexity: Input: O(1), Auxiliary: O(1).
        """
        return f"TreeNode {self.tree_id} (Shortest Distance: {self.shortest_distance}) (Index: {self.index}))"

    def __repr__(self) -> str:
        """ Representation method. Mainly used during debugging stage.

            Written by Brandon Wee Yong Jing.

            pre: None
            post: String representing some instance variable values.

            time complexity: Best case and worst case is O(1).
            space complexity: Input: O(1), Auxiliary: O(1).
        """
        return str(self)


class TreeRoad:
    def __init__(self, start_tree: TreeNode, end_tree: TreeNode, travel_time: int) -> None:
        """
        Constructor method for the TreeRoad class.

        Written by Brandon Wee Yong Jing

        Input:
            start_tree: TreeNode object representing the source of the edge
            end_tree: TreeNode object representing the destination of the edge
            travel_time: Integer representing the edge weight.
        Return:
            None

        time complexity: Best case and worst case is O(1).
        space complexity: Input: O(1), Auxiliary: O(1).
        """
        self.start_tree = start_tree
        self.end_tree = end_tree
        self.travel_time = travel_time

    def __str__(self) -> str:
        """ To string method. Mainly used during debugging stage.

            Written by Brandon Wee Yong Jing.

            pre: None
            post: String representing some instance variable values.

            time complexity: Best case and worst case is O(1).
            space complexity: Input: O(1), Auxiliary: O(1).
        """
        return f"TreeRoad: {self.start_tree.tree_id} -> {self.end_tree.tree_id} (Travel Time: {self.travel_time})"

    def __repr__(self) -> str:
        """ Representation method. Mainly used during debugging stage.

            Written by Brandon Wee Yong Jing.

            pre: None
            post: String representing some instance variable values.

            time complexity: Best case and worst case is O(1).
            space complexity: Input: O(1), Auxiliary: O(1).
        """
        return str(self)


class MinHeap:
    """
    This is a modified implementation of the MaxHeap, converted into a MinHeap with extra 'update' function from FIT1008 A3.
    Original MaxHeap written by Brendon Taylor and Jackson Goerner. Modified by Brandon Wee Yong Jing.
    """

    def __init__(self, max_size: int) -> None:
        """
        Constructor method for the MinHeap class.

        Written by Brendon Taylor and Jackson Goerner, modified by Brandon Wee Yong Jing.

        Input:
            max_size: Integer of maximum space capacity of the MinHeap
        Return:
            None

        time complexity: Best case and worst case is O(max_size).
        space complexity: Input: O(1), Auxiliary: O(max_size).
        """
        self.length = 0
        self.the_array: List[Optional[TreeNode]] = [None] * (
                    max_size + 1)  # Modification to make the_array contain None or TreeNode

    def __len__(self) -> int:
        """ Length magic method.

            Written by Brendon Taylor and Jackson Goerner.

            pre: None
            post: Integer representing the length of the MinHeap.

            time complexity: Best case and worst case is O(1).
            space complexity: Input: O(1), Auxiliary: O(1).
        """
        return self.length

    def __repr__(self) -> str:
        """ Representation magic method. Mainly used during debugging stage.

            Written by Brendon Taylor and Jackson Goerner.

            pre: None
            post: String representation of the array.

            time complexity: Best case and worst case is O(N), where N is the length of the array.
            space complexity: Input: O(1), Auxiliary: O(N).
        """
        return str(self.the_array)

    def is_full(self) -> bool:
        """ Method to determine if MinHeap is full.

            Written by Brendon Taylor and Jackson Goerner.

            pre: None
            post: Boolean representing whether the MinHeap is full.

            time complexity: Best case and worst case is O(1).
            space complexity: Input: O(1), Auxiliary: O(1).
        """
        return self.length + 1 == len(self.the_array)

    def is_empty(self) -> bool:
        """ Method to determine if MinHeap is empty.

            Written by Brandon Wee Yong Jing.

            pre: None
            post: Boolean representing whether the MinHeap is empty.

            time complexity: Best case and worst case is O(1).
            space complexity: Input: O(1), Auxiliary: O(1).
        """
        return self.length == 0

    def rise(self, k: int) -> None:
        """
            Rise element at index k to its correct position

            Written by Brendon Taylor and Jackson Goerner, modified by Brandon Wee Yong Jing.
            pre: 1 <= k <= self.length
            post: None

            time complexity: Best case is O(1) and worst case is O(log N), where N is the length of the array.
            space complexity: Input: O(1), Auxiliary: O(1).
        """
        item = self.the_array[k]
        while k > 1 and item < self.the_array[k // 2]:  # Modification made here to swap the sign from > to <.
            self.the_array[k] = self.the_array[k // 2]
            self.the_array[k].index = k
            k = k // 2
        self.the_array[k] = item
        self.the_array[k].index = k

    def add(self, element: TreeNode) -> None:
        """
            Swaps elements while rising

            Written by Brendon Taylor and Jackson Goerner.
            pre: element: TreeNode to be added into the MinHeap.
            post: None

            time complexity: Best case is O(1) and worst case is O(log N), where N is the length of the array.
            space complexity: Input: O(1), Auxiliary: O(1).
        """
        if self.is_full():
            raise IndexError

        self.length += 1
        self.the_array[self.length] = element
        self.rise(self.length)

    def smallest_child(self, k: int) -> int:
        """
            Returns the index of k's child with the smallest value.

            Written by Brendon Taylor and Jackson Goerner, modified by Brandon Wee Yong Jing.

            pre: 1 <= k <= self.length // 2
            post: None

            time complexity: Best case and worst case is O(1).
            space complexity: Input: O(1), Auxiliary: O(1).

        """
        if 2 * k == self.length or \
                self.the_array[2 * k] < self.the_array[2 * k + 1]:
            return 2 * k
        else:
            return 2 * k + 1

    def sink(self, k: int) -> None:
        """ Make the element at index k sink to the correct position.

            Written by Brendon Taylor and Jackson Goerner, modified by Brandon Wee Yong Jing.

            pre: 1 <= k <= self.length
            post: None

            time complexity: Best case is O(1) and worst case is O(log N), where N is the length of the array.
            space complexity: Input: O(1), Auxiliary: O(1).
        """
        item = self.the_array[k]

        while 2 * k <= self.length:
            min_child = self.smallest_child(k)
            if self.the_array[min_child] >= item:  # Modification to swap the sign from <= to >=.
                break
            self.the_array[k] = self.the_array[min_child]
            self.the_array[
                k].index = k  # Modification to ensure that the index instance variable of TreeNode is consistent with position in the_array.
            k = min_child

        self.the_array[k] = item
        self.the_array[
            k].index = k  # Modification to ensure that the index instance variable of TreeNode is consistent with position in the_array.

    def get_min(self) -> TreeNode:
        """ Remove (and return) the maximum element from the heap.
            Written by Brendon Taylor and Jackson Goerner, modified by Brandon Wee Yong Jing.

            pre: None
            post: min_elt: TreeNode with minimum shortest_distance to be processed.

            time complexity: Best case is O(1) and worst case is O(log N), where N is the length of the array.
            space complexity: Input: O(1), Auxiliary: O(1).
        """
        if self.length == 0:
            raise IndexError

        min_elt = self.the_array[1]
        self.length -= 1
        if self.length > 0:
            self.the_array[1], self.the_array[self.length + 1] = self.the_array[self.length + 1], self.the_array[1]
            self.the_array[
                1].index = 1  # Modification to ensure that the index instance variable of TreeNode is consistent with position in the_array.
            self.the_array[
                self.length + 1].index = self.length + 1  # Modification to ensure that the index instance variable of TreeNode is consistent with position in the_array.
            self.sink(1)
        return min_elt

    def update(self, k: int, new_time: int, previous_tree: TreeNode) -> None:
        """ Update the kth element's position due to change in the shortest distance to maintain heap property.
            Written by Brandon Wee Yong Jing.

            pre: 1 <= k <= self.length
            new_time: Integer representing the updated shortest distance.
            previous_tree: TreeNode representing the updated previous tree.
            post: None

            time complexity: Best case is O(1) and worst case is O(log N), where N is the length of the array.
            space complexity: Input: O(1), Auxiliary: O(1).
        """
        self.the_array[k].shortest_distance = new_time
        self.the_array[k].previous_tree = previous_tree
        self.rise(k)


class TreeMap:
    def __init__(self, roads: List[tuple[int, int, int]], solulus: List[tuple[int, int, int]]) -> None:
        """
        Constructor method for the TreeMap class.
        It will initialize two instance variables:
        self.number_of_trees: Integer representing how many nodes the "Standard Universe" graph have.
        self.graph: Adjacency list representing the graph. Contains both the "Standard Universe" and "Multiverse".

        Suppose there are N vertices in the standard graph. The graph in TreeMap will contain 2N vertices, N of them
        belonging in the standard universe and the remaining N represents the Multiverse.

        Both universes will have identical connectivity, with the only difference being the ID values of the vertices
        of the multiverse graph. The only way to go from the standard universe to multiverse is to destroy a solulu tree.
        Hence, we can represent this process by creating a TreeRoad from the standard universe to multiverse, with weight
        representing the time it takes to break the solulu tree. This also means that the exits that is later considered
        in the escape method will only be found in the multiverse portion of the graph, as the only way to get to the
        multiverse is to break a solulu tree.

        Written by Brandon Wee Yong Jing

        Input:
            roads: list of tuples representing the weighted roads of the graph.
            solulus: list of tuples representing the time it takes to break solulu tree and teleportation destination.
        Return:
            None

        For this complexity analysis, let |R| be the length of roads, and |T| be the length of solulus at worst.
        Time complexity:
            Best case analysis: O(|T| + |R|)
            It takes |R| operations to calculate the number of trees, |T| to initialize the adjacency list,
            |R| to add the roads into the adjacency list, and |T| to process all the solulu trees.

            Hence, Number of opeartions = |R| + |T| + |R| + |T| = O(|T| + |R|).

            Worst case analysis: O(|T| + |R|)
            Justification is exactly the same as best case complexity.

        Space complexity:
            Input space analysis: O(|T| + |R|)
            Length of solulu + length of roads = |T| + |R| = O(|T| + |R|).

            Aux space analysis: O(|T| + |R|)
            Length of adjacency list is 2*|T|, and we are adding 2*|R| edges into adjacency list, alongside an additional
            of up to |T| edges to account for solulu edges that represent teleporting.

            Hence, the auxiliary space complexity is 2|T| + 2|R| + |T| = O(|T| + |R|).
        """
        self.number_of_trees = 0  # Declare instance variable representing number of trees.

        # Iterate through roads to find the number of vertices.
        for (u, v, w) in roads:
            self.number_of_trees = max(self.number_of_trees, u + 1, v + 1)

        # Create an adjacency list of length 2*N, where N is the number of trees.
        self.tree_nodes = [TreeNode(i) for i in range(self.number_of_trees * 2)]

        # Process every road
        for (u, v, w) in roads:
            # Create road for standard universe
            start_tree_1 = self.tree_nodes[u]
            end_tree_1 = self.tree_nodes[v]
            road_1 = TreeRoad(start_tree_1, end_tree_1, w)
            start_tree_1.add_road(road_1)

            # Create road for multiverse
            start_tree_2 = self.tree_nodes[u + self.number_of_trees]
            end_tree_2 = self.tree_nodes[v + self.number_of_trees]
            road_2 = TreeRoad(start_tree_2, end_tree_2, w)
            start_tree_2.add_road(road_2)

        # Process every solulu tree
        for (x, y, z) in solulus:
            # Create an edge from the standard universe to the multiverse
            start_tree = self.tree_nodes[x]
            end_tree = self.tree_nodes[z + self.number_of_trees]
            road = TreeRoad(start_tree, end_tree, y)
            start_tree.add_road(road)
            start_tree.solulu = True

    def escape(self, start: int, exits: List[int]) -> tuple[int, List[int]]:
        """
        Return the shortest distance from the start to any exit in exits, and also the path from start to end.

        Approach:
        It will first reset all the instance variables of the adjacency list. This is because escape may be called
        multiple times on the same graph, and if we don't reset, a previous instance of escape will interfere with the
        current instance of escape.

        It will then go through the exits list, and mark the corresponding TreeNode in the adjacency list. Do recall
        that the TreeNode that will be marked as exits is the TreeNode in the multiverse rather than the standard
        universe.

        We will then initialize a priority queue, implemented as a MinHeap, that will be used during Dijkstra.

        After that, we will perform a standard Dijkstra implementation starting from the start node, until we reach
        any node marked with exit. Once we reach a node marked with exit, we will run the backtrack method to obtain
        the path that is taken to get to the exit node.

        Written by Brandon Wee Yong Jing.

        Input:
            start: integer representing the id of the starting TreeNode
            exits: List of integers representing the id of the TreeNode that one can exit from.
        Return:
            shortest_distance: Integer representing the shortest distance between the start and the closest exit.
            path: List of integers representing the path required to get from the start to the closest exit.

        For this complexity analysis, let |R| be the number of roads in the standard universe graph,
        and |T| be the number of nodes in the standard universe graph.
        First of all, the length of exits is |E|; |E| = O(|T|), since the maximum number of exits is the number of vertices in the graph which is |T|.

        Time complexity:
            Best case analysis: O(|R|log(|T|))
            It takes O(|T|) to reset the graph into its initial state.
            After resetting, it takes |E| = O(|T|) to mark all of the exits in the graph. Once we mark all the exits, we can run Dijkstra.

            The standard implementation of Dijktra has a complexity of O(Elog(V)), where E is the number of edges and V
            is the number of vertices in the graph. For this graph, there are 2|R| + |T| = O(|R|) edges, and
            2|T| = O(|T|) vertices. Hence, the complexity of running Dijkstra is O(|R|log(|T|)).
            Note: We simplify 2|R| + |T| = O(|R|) in this case as the number of edges 2|R| is much more significant
            than the worst case of |T| edges, which represents the number of edges that would be created for the
            teleporting of a solulu tree.

            After running Dijkstra, it takes at most O(|T|) to backtrack and obtain the path.
            Therefore, the total complexity is O(|T|) + O(|T|) + O(|R|log(|T|)) + O(|T|) = O(|R|log(|T|)).

            Worst case analysis: O(|R|log(|T|))
            Justification is exactly the same as best case complexity.

        Space complexity:
            Input space analysis: O(|T|)
            Exits can have length of at most |T|, so the input space complexity is O(|T|).

            Aux space analysis: O(|T| + |R|)
            The MinHeap implementation used to implement the priority queue uses an update method so that there will
            be no duplicate TreeNodes in it. So, the priority queue will have O(|T|) nodes, with a total of O(|R|) roads.
            Hence, the auxiliary space complexity to maintain the priority queue is O(|T| + |R|).

            It will also take at most 2|T| = O(|T|) space to construct the final path from start to end.
            Hence, the auxiliary space complexity is O(|T| + |R|) + O(|T|) = O(|T| + |R|).
        """
        self.reset()  # Reset graph to its initial state

        # Mark exits
        for exit_id in exits:
            tree_node = self.tree_nodes[exit_id + self.number_of_trees]
            tree_node.exit = True

        # Standard implementation of Dijktra, using MinHeap as priority queue.
        priority_queue = MinHeap(self.number_of_trees * 2)

        # Manually create heap array, using the fact that the initial state of the heap is of the form [start, TreeNode1, TreeNode2, ...]
        my_array = list(self.tree_nodes)
        my_array[0], my_array[start] = my_array[start], my_array[0]
        # Set index of each TreeNode
        for i, tree_node in enumerate(my_array):
            tree_node.index = i + 1

        # Setup priority queue
        priority_queue.the_array = [None] + my_array
        priority_queue.length = self.number_of_trees * 2

        # Set starting node's shortest distance to 0
        self.tree_nodes[start].shortest_distance = 0

        # While priority queue is not empty
        while not priority_queue.is_empty():

            # Obtain current tree to be processed
            current_tree = priority_queue.get_min()
            current_tree.visited = True  # Set to visited

            # If we found the exit, then return result
            if current_tree.exit:
                return (current_tree.shortest_distance, self.backtrack(current_tree))

            # Otherwise, process all of its neighbouring nodes.
            for road in current_tree.roads:
                end_tree = road.end_tree  # Obtain neighbour
                if not end_tree.visited:  # Consider only if neighbouring tree is not visited

                    # Calculate time it takes to traverse from start to neighbouring tree through current tree
                    new_time = road.travel_time + current_tree.shortest_distance

                    # Add to priority queue only if the new time calculated is shorter than the one found previously
                    if end_tree.shortest_distance > new_time:
                        priority_queue.update(end_tree.index, new_time, current_tree)

        # Return null answer if no path is found
        return (0, [])

    def reset(self) -> None:
        """
            Resets all TreeNodes in adjacency list into its neutral state.
            This is done because escape may be called multiple times on the same graph, and we don't want the previous
            instance of escape to interfere with the current instance of escape.

            Written by Brandon Wee Yong Jing.

            pre: None
            post: None

            time complexity: Best case and worst case is O(N), where N is the number of TreeNodes in the adjacency list.
            space complexity: Input: O(1), Auxiliary: O(1).
        """

        # Iterate through every single TreeNode in adjacency list
        for tree_node in self.tree_nodes:
            # Reset all instance variables to its initial state
            tree_node.shortest_distance = float('inf')
            tree_node.visited = False
            tree_node.previous_tree = None
            tree_node.exit = False

    def backtrack(self, end: TreeNode) -> List[int]:
        """
            This method will backtrack from the end TreeNode until it reaches the start node, in order to reconstruct the
            path.

            Written by Brandon Wee Yong Jing.

            pre: end: TreeNode representing the final node of the path.
            post: path: List of integers representing the path

            time complexity: Best case and worst case is O(N), where N is the number of TreeNodes in the adjacency list.
            space complexity: Input: O(1), Auxiliary: O(N), where N is the number of TreeNodes in the adjacency list.
        """
        # The id of the TreeNode may be different from the expected output due to the multiverse id being the standard
        # universe id + number of trees. Hence, we use the modulus operation to obtain the true id.
        path = [end.tree_id % self.number_of_trees]

        # Continue backtracking until we reach the start, which has no previous tree.
        while end is not None:
            # This condition is to prevent duplicate tree from being added consecutively, which occurs only when
            # breaking the solulu tree teleports to itself (no teleporting).
            if path[-1] != end.tree_id % self.number_of_trees:
                path.append(end.tree_id % self.number_of_trees)

            # Proceed with backtracking.
            end = end.previous_tree

        # Reverse the backtracked path and return it
        return path[::-1]