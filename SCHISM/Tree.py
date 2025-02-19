import os
import sys
import pickle
import random

import numpy as np
from itertools import chain

# A helper function to flatten a list of lists.
melt = lambda x: [subitem for item in x for subitem in item]

#----------------------------------------------------------------------#
class Node(object):
    """
    Tree data structure based on individual nodes, with hierarchical
    relationships stored within each node.

    Members:
        generationIndex: [GA interface]: the GA generation of this node/tree.
        clusterID:       ID for the cluster of mutations this node represents.
        children, parent: The children nodes and the parent of the node.
        descendants:     All the node's ancestral and descendant nodes.
    """
    __slots__ = ['clusterID', 'generationIndex', 'children', 'parent',
                 'descendants', 'samples', 'topologyRules',
                 'fitnessCoefficient',
                 'massCost', 'topologyCost', 'cost', 'fitness']

    def __init__(self, clusterID=None, samples=None, topologyRules=None,
                 fitnessCoefficient=5.0):
        self.clusterID = clusterID
        self.generationIndex = -1
        self.children, self.descendants = [], []
        self.parent = None

        self.samples = samples
        self.topologyRules = topologyRules
        self.fitnessCoefficient = fitnessCoefficient

        self.massCost, self.topologyCost, self.cost, self.fitness = 0, 0, 0, 0

    #-----------------------------------------------------------------------#
    def __iter__(self):
        for v in chain(*map(iter, self.children)):
            yield v
        yield self

    #-----------------------------------------------------------------------#
    def __getitem__(self, clusterID):
        for node in self:
            if node.clusterID == clusterID:
                return node
        return None

    #-----------------------------------------------------------------------#
    def __repr__(self):
        return '%s\t%s\t%.6f' % (self.generationIndex, self.get_newick()[1],
                                 self.fitness)

    #-----------------------------------------------------------------------#
    def copy(self):
        return pickle.loads(pickle.dumps(self, 2))

    #-----------------------------------------------------------------------#
    def copy_subtree(self, newRootIndex):
        subtree = self[newRootIndex].copy()
        subtree.parent = None
        return subtree

    #-----------------------------------------------------------------------#
    def get_nodes(self, clusterIDList=None):
        nodeList = []
        if clusterIDList is None:
            for node in self:
                nodeList.append(node)
            return nodeList
        for node in self:
            if node.clusterID in clusterIDList:
                nodeList.append(node)
        return nodeList

    #-----------------------------------------------------------------------#
    def get_indices(self, nodes=None):
        indices = []
        if nodes is None:
            for node in self:
                indices += [node.clusterID]
        elif len(nodes):
            for node in nodes:
                indices += [node.clusterID]
        return indices

    #-----------------------------------------------------------------------#
    def get_root(self):
        """Return the top node (root) of tree."""
        if self.parent is None:
            return self
        for node in self:  # iterates backwards from leaves
            if node.parent is None:
                return node

    #-----------------------------------------------------------------------#
    def get_leafs(self):
        """Return a list of the tree's leafs."""
        leafNodes = []
        for node in self:
            if not node.children and node.parent is not None:
                leafNodes.append(node)
        return leafNodes

    #-----------------------------------------------------------------------#
    def get_size(self):
        """Returns the number of connected nodes in the tree."""
        return sum(1 for _ in self)

    #-----------------------------------------------------------------------#
    def set_generation_index(self, index):
        self.generationIndex = index

    #-----------------------------------------------------------------------#
    def add_child(self, child=None):
        """Basic child add; doesn't update descendants."""
        if child is not None and child not in self.children:
            self.children.append(child)
            child.parent = self

    #-----------------------------------------------------------------------#
    def update_descendants(self):
        """Update the descendant field of nodes in the tree by bottom-up traversal."""
        if len(self.children) == 0:
            self.descendants = []
            return
        self.descendants = []
        for child in self.children:
            child.update_descendants()  # post-order traversal
            self.descendants += child.descendants + [child]
            self.descendants = list(set(self.descendants))
        return

    #-----------------------------------------------------------------------#
    def update_mass_cost(self):
        self.massCost = sum([self.samples.mass_cost(node.clusterID,
                                                     [child.clusterID for child in node.children])
                             for node in self])

    #-----------------------------------------------------------------------#
    def update_topology_cost(self):
        self.topologyCost = self.topologyRules.topology_cost(self.get_lineage_pairs())

    #-----------------------------------------------------------------------#
    def update_costs(self):
        self.update_mass_cost()
        self.update_topology_cost()
        self.cost = self.massCost + self.topologyCost

    #-----------------------------------------------------------------------#
    def update_fitness(self):
        self.update_costs()
        self.fitness = fitness_model(self.fitnessCoefficient, self.cost)

    #-----------------------------------------------------------------------#
    def update_all_data(self):
        self.update_descendants()
        self.update_fitness()

    #-----------------------------------------------------------------------#
    def mutate(self, swapFraction=0.6):
        selfClone = self.copy()
        nodesToMutate = selfClone.get_nodes()

        if random.random() < swapFraction:
            # perform swap
            node1, node2 = random.sample(nodesToMutate, 2)
            selfClone.swap_nodes(node1, node2)
        else:
            # move subtree 1-swapFraction times (on average)
            stNode, newParent = random.sample(nodesToMutate, 2)
            selfClone.move_subtree(stNode, newParent)
        return selfClone

    #-----------------------------------------------------------------------#
    def move_subtree(self, stNode, newParent):
        """
        Moves the subtree rooted at stNode to be a child of newParent.
        """
        # If new parent is already in subtree, swap the two.
        if newParent in stNode.descendants:
            stNode, newParent = newParent, stNode

        # Make sure we are moving to another part of the tree.
        if newParent == stNode.parent:
            return

        # Update immediate parents.
        oldParent = stNode.parent
        newParent.add_child(stNode)
        oldParent.children.remove(stNode)

        # Update costs/fitness and ancestor/descendant relationships.
        self.update_all_data()

    #-----------------------------------------------------------------------#
    def swap_nodes(self, node1, node2):
        """
        Swap the clusterIDs associated with two nodes and update the tree.
        """
        if node1.parent == node2.parent:
            return
        node1.clusterID, node2.clusterID = node2.clusterID, node1.clusterID
        self.update_all_data()

    #-----------------------------------------------------------------------#
    def equal_topology(self, other):
        """
        Returns True if the topologies of the two trees are the same.
        Does not consider cost/fitness values.
        """
        if other is None:
            return False
        return self.clusterID == other.clusterID and self.get_pairs() == other.get_pairs()

    #-----------------------------------------------------------------------#
    def get_pairs(self):
        """
        Get directed pairs (clusterIDs) from nodes in the tree.
        Can be used for tree comparison.
        """
        return sorted(list(set([(node.clusterID, child.clusterID)
                                 for node in self for child in node.children])))

    #-----------------------------------------------------------------------#
    def get_lineage_pairs(self):
        """
        Get directed pairs (clusterIDs) including descendants.
        """
        return sorted(list(set([(node.clusterID, desc.clusterID)
                                 for node in self for desc in node.descendants])))

    #-----------------------------------------------------------------------#
    def jaccard_similarity_index(self, other):
        if self.equal_topology(other):
            return 1.0
        selfPairs = self.get_lineage_pairs()
        otherPairs = other.get_lineage_pairs()
        overlap = len(set(selfPairs) & set(otherPairs))
        union = len(set(selfPairs) | set(otherPairs))
        return float(overlap) / union

    #-----------------------------------------------------------------------#
    def cut_node(self, node, tempNode):
        """
        Disconnect node (and its subtree) from the tree and attach it to a temporary node.
        """
        oldParent = node.parent
        tempNode.add_child(node)
        oldParent.children.remove(node)

    #-----------------------------------------------------------------------#
    def prune_leafs(self, nodeNumToPrune, tempNode):
        """
        Remove leaf nodes iteratively until nodeNumToPrune nodes have been pruned.
        """
        startSize = self.get_size()
        if nodeNumToPrune > startSize - 1:
            print("asking too many nodes (%d) to prune!" % nodeNumToPrune, file=sys.stderr)
            print(self.get_newick(), file=sys.stderr)
            print(nodeNumToPrune, file=sys.stderr)
            sys.exit()
            return
        topIndex = self.clusterID
        leafNodes = self.get_leafs()
        prunedNodes = []
        for _ in range(nodeNumToPrune):
            leafNode = random.choice(leafNodes)
            prunedNodes.append(leafNode)
            if len(leafNode.parent.children) == 1:
                leafNodes.append(leafNode.parent)  # parent becomes leaf after pruning
            self.cut_node(leafNode, tempNode)
            leafNodes.remove(leafNode)

    #-----------------------------------------------------------------------#
    def cross(self, otherNode):
        """
        Cross two trees by exchanging subtrees and reindexing.
        """
        selfOffspring, otherOffspring = self.copy(), otherNode.copy()
        clusterIndices = selfOffspring.get_indices()
        selfNodesToCross = selfOffspring.get_nodes()
        otherNodesToCross = otherOffspring.get_nodes()

        selfNodesToCross.remove(selfOffspring.get_root())
        otherNodesToCross.remove(otherOffspring.get_root())

        selfSubTree = random.choice(selfNodesToCross)
        otherSubTree = random.choice(otherNodesToCross)

        # Find which tree to prune before crossing (larger tree)
        selfSubTreeSize = selfSubTree.get_size()
        otherSubTreeSize = otherSubTree.get_size()

        nodeNumToPrune = max(selfSubTreeSize, otherSubTreeSize) - min(selfSubTreeSize, otherSubTreeSize)

        if selfSubTreeSize < otherSubTreeSize:
            toPrune = "other"
        elif selfSubTreeSize > otherSubTreeSize:
            toPrune = "self"
        else:
            toPrune = "neither"

        tempNode = Node(-1)

        # Prune the larger of the two.
        if toPrune == "self":
            selfSubTree.prune_leafs(nodeNumToPrune, tempNode)
        elif toPrune == "other":
            otherSubTree.prune_leafs(nodeNumToPrune, tempNode)

        # Swap parental attachments between the two.
        selfSubTreeParent = selfSubTree.parent
        otherSubTreeParent = otherSubTree.parent
        selfSubTreeParent.add_child(otherSubTree)
        otherSubTreeParent.add_child(selfSubTree)
        selfSubTreeParent.children.remove(selfSubTree)
        otherSubTreeParent.children.remove(otherSubTree)

        # Floating nodes after pruning.
        numSubTrees = len(tempNode.children)  # number of nodes to reattach
        reattachParents = []
        selfSubTreeNodes = selfSubTree.get_nodes()
        otherSubTreeNodes = otherSubTree.get_nodes()

        # Randomly pick one parent for each floating node from the opposite subtree.
        if toPrune == "self":
            reattachParents = [random.choice(otherSubTreeNodes) for _ in range(numSubTrees)]
        elif toPrune == "other":
            reattachParents = [random.choice(selfSubTreeNodes) for _ in range(numSubTrees)]

        # Perform reattachment.
        for index, node in enumerate(reattachParents):
            node.add_child(tempNode.children[index])

        tempNode.children = []  # Clear the temporary node.

        # Renumber cluster IDs so all are present and unique.
        allClusterIds = set(self.get_indices())

        selfMiss = list(allClusterIds - set(selfOffspring.get_indices()))
        selfMiss.sort(reverse=True)
        otherMiss = list(allClusterIds - set(otherOffspring.get_indices()))
        otherMiss.sort(reverse=True)
        selfDouble, otherDouble = otherMiss, selfMiss

        if len(selfDouble) != len(selfMiss):
            print("WARNING: Unequal number of double and missing nodes", file=sys.stderr)
            print(self.get_newick(), otherNode.get_newick(), file=sys.stderr)
            print(selfSubTree.clusterID, otherSubTree.clusterID, file=sys.stderr)
            print(selfSubTree.get_indices(), otherSubTree.get_indices(), file=sys.stderr)
            print(selfOffspring.get_indices(), otherOffspring.get_indices(), file=sys.stderr)
            print("selfDouble/otherMiss", selfDouble, "selfMiss/otherDouble", selfMiss, file=sys.stderr)
            sys.exit()

        replacedIndices = []
        for node in otherSubTree:
            if node.clusterID in selfDouble and node.clusterID not in replacedIndices:
                replacedIndices.append(node.clusterID)
                node.clusterID = selfMiss[selfDouble.index(node.clusterID)]
            if len(replacedIndices) >= len(selfDouble):
                break

        replacedIndices = []
        for node in selfSubTree:
            if node.clusterID in otherDouble and node.clusterID not in replacedIndices:
                replacedIndices.append(node.clusterID)
                node.clusterID = otherMiss[otherDouble.index(node.clusterID)]
            if len(replacedIndices) >= len(otherDouble):
                break

        selfOffspring.update_all_data()
        otherOffspring.update_all_data()
        return selfOffspring, otherOffspring

    #-----------------------------------------------------------------------#
    @staticmethod
    def random_topology(**kwargs):
        """
        Generate a random tree topology.
        """
        clusterIDs = kwargs['clusterIDs']
        numNodes = len(clusterIDs)
        rootID = random.choice(clusterIDs)
        nonRootIDs = [ID for ID in clusterIDs if ID != rootID]
        random.shuffle(nonRootIDs)

        samples = kwargs['massRules'] if 'massRules' in kwargs else None
        topologyRules = kwargs['topologyRules'] if 'topologyRules' in kwargs else None
        fitnessCoefficient = kwargs['fitnessCoefficient'] if 'fitnessCoefficient' in kwargs else 1

        randTree = Node(clusterID=rootID,
                        samples=samples,
                        topologyRules=topologyRules,
                        fitnessCoefficient=fitnessCoefficient)
        treeNodeIDs = [rootID]

        for newNodeID in nonRootIDs:
            newNode = Node(clusterID=newNodeID,
                           samples=samples,
                           topologyRules=topologyRules,
                           fitnessCoefficient=fitnessCoefficient)
            parentNodeID = random.choice(treeNodeIDs)
            randTree[parentNodeID].add_child(newNode)
            treeNodeIDs.append(newNodeID)

        randTree.update_all_data()
        return randTree

    #-----------------------------------------------------------------------#
    def string_identifier(self):
        return self.get_newick()[1]

    #-----------------------------------------------------------------------#
    @staticmethod
    def from_newick(nwkStr):
        if '(' not in nwkStr:
            # Single node
            nwkTree = Node(nwkStr)
            return nwkTree

        # More than a single node is present.
        start = nwkStr.index('(')
        end = len(nwkStr) - nwkStr[::-1].index(')') - 1
        nodeID = nwkStr[end+1:]
        nwkTree = Node(nodeID)
        inner = nwkStr[start+1:end]

        # Identify locations outside outer-level parentheses.
        status = [0] * len(inner)
        for index in range(len(inner)):
            if inner[index] == '(':
                status[index] = 1
            elif inner[index] == ')':
                status[index] = -1
        status = list(np.cumsum(status))
        free = [index for index in range(len(status)) if status[index] == 0]
        # Identify commas.
        commas = [index for index in range(len(inner)) if inner[index] == ',']
        # Commas outside outer parentheses are break points.
        breaks = sorted(list(set(free) & set(commas)))
        if breaks == []:
            child = Node.from_newick(inner)
            nwkTree.add_child(child)
            return nwkTree

        breaks.insert(0, -1)
        breaks.append(len(inner) + 1)
        childrenStr = [inner[breaks[index]+1:breaks[index+1]] for index in range(len(breaks)-1)]
        for childStr in childrenStr:
            child = Node.from_newick(childStr)
            nwkTree.add_child(child)
        return nwkTree

    #-----------------------------------------------------------------------#
    def get_newick(self):
        if len(self.children) == 0:
            return self.clusterID, str(self.clusterID)
        else:
            children = [child.get_newick() for child in self.children]
            children = sorted(children, key=lambda x: x[0])
            childrenStr = list(zip(*children))[1]
            return self.clusterID, '(' + ','.join(childrenStr) + ')' + str(self.clusterID)

#----------------------------------------------------------------------#
def fitness_model(coefficient, cost):
    return np.exp(-coefficient * cost)
