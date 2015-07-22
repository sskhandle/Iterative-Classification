from graph import DirectedGraph, Node, Edge

def load_linqs_data(content_file, cites_file):
    '''
    Create a DirectedGraph object and add Nodes and Edges
    This is specific to the data files provided at http://linqs.cs.umd.edu/projects/projects/lbc/index.html
    Return two items 1. graph object, 2. the list of domain labels (e.g., ['AI', 'IR'])
    '''
    linqs_graph=DirectedGraph()
    domain_labels=[]
    id_obj_map={}

    with open(content_file, 'r') as node_file:
        for line in node_file:
            line_info=line.split('\n')[0].split('\t')
            n=Node(line_info[0],map(float,line_info[1:-1]),line_info[-1])# id, feature vector, label
            linqs_graph.add_node(n)
            if line_info[-1] not in domain_labels:
                domain_labels.append(line_info[-1])
            id_obj_map[line_info[0]]=n

    with open(cites_file,'r') as edge_file:
        for line in edge_file:
            line_info=line.split('\n')[0].split('\t')
            if line_info[0] in id_obj_map.keys() and line_info[1] in id_obj_map.keys():
                from_node=id_obj_map[line_info[1]]
                to_node=id_obj_map[line_info[0]]
                linqs_graph.add_edge(Edge(from_node,to_node))

    print "domain labels"
    print domain_labels

    return linqs_graph,domain_labels


