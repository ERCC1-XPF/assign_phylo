"""Helper functions for HW3"""
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


# Problems 1-3: tree node used by neighbor joining, plotting and rerooting
class Node:
    def __init__(
        self,
        name: str,
        left: "Node",
        left_distance: float,
        right: "Node",
        right_distance: float,
        confidence: float = None,
    ):
        """A node in a binary tree produced by neighbor joining algorithm.

        Parameters
        ----------
        name: str
            Name of the node.
        left: Node
            Left child.
        left_distance: float
            The distance to the left child.
        right: Node
            Right child.
        right_distance: float
            The distance to the right child.
        confidence: float
            The confidence level of the split determined by the bootstrap method.
            Only used if you implement Bonus Problem 1.

        Notes
        -----
        The current public API needs to remain as it is, i.e., don't change the
        names of the properties in the template, as the tests expect this kind
        of structure. However, feel free to add any methods/properties/attributes
        that you might need in your tree construction.

        """
        self.name = name
        self.left = left
        self.left_distance = left_distance
        self.right = right
        self.right_distance = right_distance
        self.confidence = confidence


# Problem 1
def neighbor_joining(distances: np.ndarray, labels: list) -> Node:
    """The Neighbor-Joining algorithm.

    For the same results as in the later test dendrograms;
    add new nodes to the end of the list/matrix and
    in case of ties, use np.argmin to choose the joining pair.

    Parameters
    ----------
    distances: np.ndarray
        A 2d square, symmetric distance matrix containing distances between
        data points. The diagonal entries should always be zero; d(x, x) = 0.
    labels: list
        A list of labels corresponding to entries in the distances matrix.
        Use them to set names of nodes.

    Returns
    -------
    Node
        A root node of the neighbor joining tree.

    """
    # Create a float copy to keep the original matrix unchanged
    D = np.array(distances, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1] or D.shape[0] != len(labels):
        raise ValueError("`distances` must be a square matrix matching `labels`")

    # Every label starts as a leaf node; leaves have no children
    nodes = [Node(label, None, 0, None, 0) for label in labels]
    if len(nodes) == 1:
        return nodes[0]

    n_internal = 0
    while len(nodes) > 2:
        n = len(nodes)
        row_sums = D.sum(axis=1)

        # Calculate Q-values to determine which pair to join
        Q = (n - 2) * D - row_sums[:, None] - row_sums[None, :]

        #Exclude self-pairs when finding the minimum Q-value
        np.fill_diagonal(Q, np.inf)  # i != j

        # Find the minimum Q-value, choosing the first pair if tied
        i, j = np.unravel_index(np.argmin(Q), Q.shape)

        # Branch lengths from i and j to the new node u
        delta_i = 0.5 * D[i, j] + (row_sums[i] - row_sums[j]) / (2 * (n - 2))
        delta_j = D[i, j] - delta_i

        new_node = Node(f"internal_{n_internal}", nodes[i], delta_i, nodes[j], delta_j)
        n_internal += 1

        # Calculate distances from the new node to remaining nodes
        new_dist = 0.5 * (D[i] + D[j] - D[i, j])

        # Drop rows/cols i and j, then append u at the end of the matrix/list
        keep = [k for k in range(n) if k != i and k != j]
        new_dist = new_dist[keep]
        D = D[np.ix_(keep, keep)]
        D = np.block([
            [D, new_dist[:, None]],
            [new_dist[None, :], np.zeros((1, 1))],
        ])
        nodes = [nodes[k] for k in keep] + [new_node]

    # Last two nodes are joined by a root placed halfway between them
    half = D[0, 1] / 2
    return Node("ROOT", nodes[0], half, nodes[1], half)


# Problem 2
def plot_nj_tree(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.
        But make sure your function can work without them, for testing purposes.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree(tree=tree, ax=ax)
    >>> fig.savefig("example.png")

    Optional kwargs
    ---------------
    color: default line/label colour (default "k").
    lw: line width (default 1.5).
    fontsize: leaf label font size (default 12).
    leaf_colors: dict {leaf name: colour}. A branch takes the colour of its
        clade when every leaf below it has the same colour.
    label_func: callable mapping a leaf name to the text shown (default str).
    root_stub: length of the short line drawn left of the root
        (default 2% of the tree depth).
    show_confidence: annotate internal nodes whose `confidence` is set
        (default True).
    confidence_fontsize: font size of the confidence values
        (default 0.6 * fontsize).

    """
    if ax is None:
        ax = plt.gca()

    color = kwargs.get("color", "k")
    lw = kwargs.get("lw", 1.5)
    fontsize = kwargs.get("fontsize", 12)
    leaf_colors = kwargs.get("leaf_colors", {})
    label_func = kwargs.get("label_func", str)
    show_confidence = kwargs.get("show_confidence", True)
    confidence_fontsize = kwargs.get("confidence_fontsize", fontsize * 0.6)

    def children(node):
        return [
            (child, dist)
            for child, dist in [(node.left, node.left_distance), (node.right, node.right_distance)]
            if child is not None
        ]

    # 1. Set node positions based on branch lengths and leaf order
    # Place internal nodes at the mean y-position of their children
    coords = {}
    leaf_counter = [0]

    def layout(node, x):
        kids = children(node)
        if not kids:
            y = leaf_counter[0]
            leaf_counter[0] += 1
        else:
            y = np.mean([layout(child, x + dist) for child, dist in kids])
        coords[id(node)] = (x, y)
        return y

    layout(tree, 0.0)

    # 2. Clade colour: shared colour of all leaves below a node, else None
    clade_color = {}

    def find_clade_color(node):
        kids = children(node)
        if not kids:
            c = leaf_colors.get(node.name)
        else:
            kid_colors = {find_clade_color(child) for child, _ in kids}
            c = kid_colors.pop() if len(kid_colors) == 1 else None
        clade_color[id(node)] = c
        return c

    find_clade_color(tree)

    # 3. Draw: a vertical line spanning the children of each internal node
    #    and a horizontal line from each node to every child
    stack = [tree]
    while stack:
        node = stack.pop()
        x, y = coords[id(node)]
        kids = children(node)

        if not kids:
            ax.text(
                x, y, f"  {label_func(node.name)}",
                color=clade_color[id(node)] or color,
                fontsize=fontsize,
                verticalalignment="center",
            )
            continue

        child_ys = [coords[id(child)][1] for child, _ in kids]
        ax.plot([x, x], [min(child_ys), max(child_ys)],
                color=clade_color[id(node)] or color, lw=lw)
        for child, _ in kids:
            cx, cy = coords[id(child)]
            ax.plot([x, cx], [cy, cy], color=clade_color[id(child)] or color, lw=lw)
            stack.append(child)

        if show_confidence and node.confidence is not None and node is not tree:
            ax.text(
                x, y, f"{node.confidence:.2f} ",
                fontsize=confidence_fontsize,
                horizontalalignment="right",
                verticalalignment="bottom",
            )

    # Short stub to the left of the root
    max_depth = max(x for x, _ in coords.values())
    root_stub = kwargs.get("root_stub", 0.02 * max_depth if max_depth > 0 else 0.2)
    root_x, root_y = coords[id(tree)]
    ax.plot([root_x - root_stub, root_x], [root_y, root_y], color=color, lw=lw)

    for loc in ["top", "right"]:
        ax.spines[loc].set_visible(False)

    return ax




# Problem 3b: helper of reroot_tree 
def _find_a_parent_to_node(tree: Node, node: Node) -> tuple:
    """Utility function for reroot_tree"""
    stack = [tree]

    while len(stack) > 0:

        current_node = stack.pop()
        if node.name == current_node.left.name:
            return current_node, "left"
        elif node.name == current_node.right.name:
            return current_node, "right"

        stack += [
            n for n in [current_node.left, current_node.right] if n.left is not None
        ]

    return None



# Problem 3b
def _remove_child_from_parent(parent_node: Node, child_location: str) -> None:
    """Utility function for reroot_tree"""
    setattr(parent_node, child_location, None)
    setattr(parent_node, f"{child_location}_distance", 0.0)


# Problem 3b
def reroot_tree(original_tree: Node, outgroup_node: Node) -> Node:
    """A function to create a new root and invert a tree accordingly.

    This function reroots tree with nodes in original format. If you
    added any other relational parameters to your nodes, these parameters
    will not be inverted! You can modify this implementation or create
    additional functions to fix them.

    Parameters
    ----------
    original_tree: Node
        A root node of the original tree.
    outgroup_node: Node
        A Node to set as an outgroup (already included in a tree).
        Find it by it's name and then use it as parameter.

    Returns
    -------
    Node
        Inverted tree with a new root node.
    """
    tree = deepcopy(original_tree)

    parent, child_loc = _find_a_parent_to_node(tree, outgroup_node)
    distance = getattr(parent, f"{child_loc}_distance")
    _remove_child_from_parent(parent, child_loc)

    new_root = Node("new_root", parent, distance / 2, outgroup_node, distance / 2)
    child = parent

    while tree != child:
        parent, child_loc = _find_a_parent_to_node(tree, child)

        distance = getattr(parent, f"{child_loc}_distance")
        _remove_child_from_parent(parent, child_loc)

        empty_side = "left" if child.left is None else "right"
        setattr(child, f"{empty_side}_distance", distance)
        setattr(child, empty_side, parent)

        if tree.name == parent.name:
            break
        child = parent

    other_child_loc = "right" if child_loc == "left" else "left"
    other_child_distance = getattr(parent, f"{other_child_loc}_distance")

    setattr(child, f"{empty_side}_distance", other_child_distance + distance)
    setattr(child, empty_side, getattr(parent, other_child_loc))

    return new_root


# Problem 3c
def sort_children_by_leaves(tree: Node) -> None:
    """Sort the children of a tree by their corresponding number of leaves.

    The tree can be changed inplace.

    Parameters
    ----------
    tree: Node
        The root node of the tree.

    """
    def count_and_sort(node):
        """Sort the subtree below `node` and return its number of leaves."""
        if node is None:
            return 0
        if node.left is None and node.right is None:
            return 1

        # Sort the children first (bottom-up), counting their leaves on the way
        n_left = count_and_sort(node.left)
        n_right = count_and_sort(node.right)

        # The left child should be the one with fewer leaves; swap the
        # children together with their branch lengths
        if n_left > n_right:
            node.left, node.right = node.right, node.left
            node.left_distance, node.right_distance = node.right_distance, node.left_distance
        return n_left + n_right

    count_and_sort(tree)



def plot_nj_tree_radial(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram
    with a radial layout.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree_radial(tree=tree, ax=ax)
    >>> fig.savefig("example_radial.png")

    """
    raise NotImplementedError()


# Problem 4: Needleman-Wunsch from the previous assignment
def global_alignment(seq1, seq2, scoring_function):
    """Global sequence alignment using the Needleman–Wunsch algorithm.

    Indels should be denoted with the "-" character.

    Parameters
    ----------
    seq1: str
        First sequence to be aligned.
    seq2: str
        Second sequence to be aligned.
    scoring_function: Callable

    Returns
    -------
    str
        First aligned sequence.
    str
        Second aligned sequence.
    float
        Final score of the alignment.

    Examples
    --------
    >>> global_alignment("abracadabra", "dabarakadara", lambda x, y: [-1, 1][x == y])
    ('-ab-racadabra', 'dabarakada-ra', 5.0)

    Other alignments are not possible.

    """
    n = len(seq1)
    m = len(seq2)

    score = [[0] * (m + 1) for _ in range(n + 1)]
    pointer = [[None] * (m + 1) for _ in range(n + 1)]

    # Initialize the first col. Fill the first col by aligning seq1 prefixes to gaps
    for i in range(1, n + 1):
        score[i][0] = score[i - 1][0] + scoring_function(seq1[i - 1], "-")
        pointer[i][0] = "up"

    # Fill the first row by aligning seq2 prefixes to gaps
    for j in range(1, m + 1):
        score[0][j] = score[0][j - 1] + scoring_function("-", seq2[j - 1])
        pointer[0][j] = "left"

    # Fill the matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = score[i - 1][j - 1] + scoring_function(seq1[i - 1], seq2[j - 1])
            up = score[i - 1][j] + scoring_function(seq1[i - 1], "-")
            left = score[i][j - 1] + scoring_function("-", seq2[j - 1])

            score[i][j] = max(diagonal, up, left)

            if score[i][j] == diagonal:
                pointer[i][j] = "diag"
            elif score[i][j] == up:
                pointer[i][j] = "up"
            else:
                pointer[i][j] = "left"

    # Trace back from the bottom right cell
    aln1, aln2 = [], []
    i, j = n, m
    while i > 0 or j > 0:
        move = pointer[i][j]
        if move == "diag":
            aln1.append(seq1[i - 1])
            aln2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif move == "up":
            aln1.append(seq1[i - 1])
            aln2.append("-")
            i -= 1
        else:
            aln1.append("-")
            aln2.append(seq2[j - 1])
            j -= 1

    return "".join(reversed(aln1)), "".join(reversed(aln2)), float(score[n][m])



def scoring_function_simple(aa_i,aa_j):
    score = [-1, 1][aa_i == aa_j]
    return (score)


# Problem 4: numpy version of global_alignment for whole genomes
def global_alignment_fast(seq1, seq2, match=1, mismatch=-1, gap=-1):
    """Needleman-Wunsch with match/mismatch/linear-gap scores, vectorised with numpy.

    Gives the same alignment as `global_alignment(seq1, seq2, scoring_function_simple)`
    (same recurrence and same tie-breaking: diagonal, then up, then left), but fills
    one whole row of the table at a time, so ~30k x 30k genomes take seconds instead
    of hours. Only two rows of scores are kept; the traceback pointers are stored as
    packed bits (2 bits per cell, ~225 MB for two 30 kb genomes).

    Returns
    -------
    (str, str, float)
        First aligned sequence, second aligned sequence, alignment score.
    """
    a = np.frombuffer(seq1.encode(), dtype=np.uint8)
    b = np.frombuffer(seq2.encode(), dtype=np.uint8)
    n, m = len(a), len(b)

    gap_cost = gap * np.arange(m + 1)        # score of aligning seq2[:j] to gaps
    previous = gap_cost.astype(np.int64)      # row 0 of the score table
    n_bytes = (m + 1 + 7) // 8
    up_bits = np.zeros((n + 1, n_bytes), dtype=np.uint8)
    left_bits = np.zeros((n + 1, n_bytes), dtype=np.uint8)
    first_row_left = np.ones(m + 1, dtype=bool)
    first_row_left[0] = False
    left_bits[0] = np.packbits(first_row_left)

    best_from_above = np.empty(m + 1, dtype=np.int64)
    came_from_up = np.empty(m + 1, dtype=bool)
    for i in range(1, n + 1):
        substitution = np.where(b == a[i - 1], match, mismatch)
        diagonal = previous[:-1] + substitution
        up = previous + gap

        # Choose the higher score from diagonal or up, preferring diagonal in ties
        # Column 0 can only come from up
        best_from_above[0] = up[0]
        came_from_up[0] = True
        np.maximum(diagonal, up[1:], out=best_from_above[1:])
        np.greater(up[1:], diagonal, out=came_from_up[1:])

        # Calculate the best scores from left moves using a running maximum
        current = np.maximum.accumulate(best_from_above - gap_cost) + gap_cost
        came_from_left = current > best_from_above   # left only if strictly better

        up_bits[i] = np.packbits(came_from_up)
        left_bits[i] = np.packbits(came_from_left)
        previous = current

    def bit(bits, i, j):
        return (bits[i, j >> 3] >> (7 - (j & 7))) & 1

    # Trace back from the bottom right cell
    aln1, aln2 = [], []
    i, j = n, m
    while i > 0 or j > 0:
        if bit(left_bits, i, j):
            aln1.append("-")
            aln2.append(seq2[j - 1])
            j -= 1
        elif i > 0 and (j == 0 or bit(up_bits, i, j)):
            aln1.append(seq1[i - 1])
            aln2.append("-")
            i -= 1
        else:
            aln1.append(seq1[i - 1])
            aln2.append(seq2[j - 1])
            i -= 1
            j -= 1

    return "".join(reversed(aln1)), "".join(reversed(aln2)), float(previous[m])
