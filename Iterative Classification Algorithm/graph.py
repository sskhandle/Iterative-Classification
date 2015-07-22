from collections import defaultdict

def abstract():
    import inspect
    caller = inspect.getouterframes(inspect.currentframe())[1][3]
    raise NotImplementedError(caller + ' must be implemented in subclass')

class Graph(object):
    '''
    The base Graph class
    '''

    def __init__(self):
        '''
        Create an empty graph
        '''
        self.node_list = []
        self.edge_list = []        
    
    def add_node(self, n):
        self.node_list.append(n)        
    
    def add_edge(self, e):
        abstract()
    
    def get_neighbors(self, n):
        abstract()

class Node(object):
    def __init__(self, node_id, feature_vector = None, label = None):
        self.node_id = node_id
        self.feature_vector = feature_vector
        self.label = label


class Edge(object):
    def __init__(self, from_node, to_node, feature_vector = None, label = None):
        self.from_node = from_node
        self.to_node = to_node
        self.feature_vector = feature_vector
        self.label = label


class DirectedGraph(Graph):
    
    def __init__(self):
        super(DirectedGraph, self).__init__()
        self.out_neighbors = defaultdict(set)
        self.in_neighbors = defaultdict(set)
        self.str_class=[]

    def add_edge(self, e):
        self.edge_list.append(e)
        self.out_neighbors[e.from_node].add(e.to_node)
        self.in_neighbors[e.to_node].add(e.from_node)
    
    def get_out_neighbors(self, n):
        return self.out_neighbors[n]
    
    def get_in_neighbors(self, n):
        return self.in_neighbors[n]
    
    def get_neighbors(self, n):
        return self.out_neighbors[n].union(self.in_neighbors[n])

class UndirectedGraph(Graph):
    
    def __init__(self):
        super(UndirectedGraph, self).__init__()
        self.neighbors = defaultdict(set)        
    
    def add_edge(self, e):
        self.neighbors[e.from_node].add(e.to_node)
        self.neighbors[e.to_node].add(e.from_node)
    
    def get_neighbors(self, n):
        return self.neighbors[n]

        