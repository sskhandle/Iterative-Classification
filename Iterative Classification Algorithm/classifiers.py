import numpy as np

def abstract():
    import inspect
    caller = inspect.getouterframes(inspect.currentframe())[1][3]
    raise NotImplementedError(caller + ' must be implemented in subclass')



class Aggregator(object):
    
    def __init__(self, domain_labels, directed = False):
        self.domain_labels = domain_labels # The list of labels in the domain
        self.directed = directed # Whether we should use edge directions for creating the aggregation
    
    def aggregate(self, graph, node, conditional_node_to_label_map):
        ''' Given a node, its graph, and labels to condition on (observed and/or predicted)
            create and return a feature vector for neighbors of this node.
            If a neighbor is not in conditional_node_to_label_map, ignore it.
            If directed = True, create and append two feature vectors;
            one for the out-neighbors and one for the in-neighbors.
            '''
        abstract()

class CountAggregator(Aggregator):
    '''The count aggregate'''
    
    def aggregate(self, graph, node, conditional_node_to_label_map):
        neighbor_undirected = []
        neighbor_directed_in = []
        neighbor_directed_out = []
        if self.directed:
            for x in self.domain_labels:
                neighbor_directed_in.append(0.0)
                neighbor_directed_out.append(0.0)
            for eachIn in graph.get_in_neighbors(node):
                if eachIn in conditional_node_to_label_map:
                    index = self.domain_labels.index(conditional_node_to_label_map[eachIn])
                    neighbor_directed_in[index] += 1.0
            for eachOut in graph.get_out_neighbors(node):
                if eachOut in conditional_node_to_label_map:
                    index = self.domain_labels.index(conditional_node_to_label_map[eachOut])
                    neighbor_directed_out[index] += 1.0
            return neighbor_directed_in+neighbor_directed_out
        else:
            for x in self.domain_labels:
                neighbor_undirected.append(0.0)
            for i in graph.get_neighbors(node):
                if i in conditional_node_to_label_map.keys():
                    index = self.domain_labels.index(conditional_node_to_label_map[i])
                    neighbor_undirected[index] += 1.0
            return neighbor_undirected



class ProportionalAggregator(Aggregator):
    '''The proportional aggregate'''
    
    def aggregate(self, graph, node, conditional_node_to_label_map):
        cntag = CountAggregator(self.domain_labels,self.directed)
        cnt_agg = cntag.aggregate(graph,node,conditional_node_to_label_map)
        if self.directed:
            in_neighbor_sum=sum(cnt_agg[:len(self.domain_labels)])
            out_neighbor_sum=sum(cnt_agg[len(self.domain_labels):])
            if in_neighbor_sum > 0:
                for r in range(0,len(self.domain_labels)):
                    cnt_agg[r]/=in_neighbor_sum
            if out_neighbor_sum > 0:
                for r in range(len(self.domain_labels),len(cnt_agg)):
                    cnt_agg[r]/=out_neighbor_sum
            p_list = cnt_agg
            return p_list
        else:
            total_sum = sum(cnt_agg)
            if total_sum > 0:
                for r in range(len(cnt_agg)):
                    cnt_agg[r] /= total_sum
            p_list = cnt_agg
            return p_list

class ExistAggregator(Aggregator):
    '''The exist aggregate'''
    
    def aggregate(self, graph, node, conditional_node_to_label_map):
        cntag=CountAggregator(self.domain_labels,self.directed)
        cnt_agg = cntag.aggregate(graph,node,conditional_node_to_label_map)
        for r in range(len(cnt_agg)):
            if cnt_agg[r] >= 1:
                cnt_agg[r] = 1
        ext_list = cnt_agg
        return ext_list


def get_class( kls ):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    md = __import__( module )
    for comp in parts[1:]:
        md = getattr(md, comp)
    return md

class Classifier(object):
    '''
        The base classifier object
        '''
    
    def __init__(self, scikit_classifier_name, **classifier_args):
        classifer_class=get_class(scikit_classifier_name)
        self.clf = classifer_class(**classifier_args)
    
    
    def fit(self, graph, train_indices):
        '''
            Create a scikit-learn classifier object and fit it using the Nodes of the Graph
            that are referenced in the train_indices
            '''
        abstract()
    
    def predict(self, graph, test_indices, conditional_node_to_label_map = None):
        '''
            This function should be called only after the fit function is called.
            Predict the labels of test Nodes conditioning on the labels in conditional_node_to_label_map.
            '''
        abstract()

class LocalClassifier(Classifier):
    
    def fit(self, graph, train_indices):
        
        feature_list= []
        label_list=[]
        g= graph
        n= g.node_list
        training_nodes=[n[i] for i in train_indices]
        
        for nodes in training_nodes:
            feature_list.append(nodes.feature_vector)
            label_list.append(nodes.label)
        
        self.clf.fit(feature_list, label_list)
        return
    
    
    def predict(self, graph, test_indices, conditional_node_to_label_map = None):
        
        feature_list=[]
        g= graph
        n=g.node_list
        testing_nodes = [n[i] for i in test_indices]
        
        for nodes in testing_nodes:
            feature_list.append(nodes.feature_vector)
        
        y= self.clf.predict(feature_list)
        return y


class RelationalClassifier(Classifier):
    
    def __init__(self, scikit_classifier_name, aggregator, use_node_attributes = True, **classifier_args):
        super(RelationalClassifier, self).__init__(scikit_classifier_name, **classifier_args)
        self.aggregator = aggregator
        self.use_node_attributes = use_node_attributes
    
    
    
    def fit(self, graph, train_indices):
        '''
            Create a feature list of lists (or matrix) and a label list
            (or array) and then fit using self.clf
            You need to use aggregator to create relational features.
            Note that the aggregator needs to know what to condition on,
            i.e., conditional_node_to_label_map. This should be created using only the train nodes.
            The features list might or might not include the node features, depending on
            the value of use_node_attributes.
            '''
        conditional_map={}
        for i in train_indices:
            conditional_map[graph.node_list[i]]=graph.node_list[i].label
        features=[]
        labels=[]
        for i in train_indices:
            self.feature_combination_check(graph,features,i,conditional_map)
            labels.append(graph.node_list[i].label)
        self.clf.fit(features,labels)
    
    def predict(self, graph, test_indices, conditional_node_to_label_map = None):
        '''
            This function should be called only after the fit function is called.
            Predict the labels of test Nodes conditioning on the labels in conditional_node_to_label_map.
            conditional_node_to_label_map might include the observed and predicted labels.
            This method is NOT iterative; it does NOT update conditional_node_to_label_map.
            '''
        # raise NotImplementedError('You need to implement this method')
        features=[]
        for i in test_indices:
            self.feature_combination_check(graph,features,i,conditional_node_to_label_map)
        return self.clf.predict(features)
    
    def feature_combination_check(self,graph,features,i,conditional_map):
        aggregates=self.aggregator.aggregate(graph,graph.node_list[i],conditional_map)
        feat_list=np.array([])
        if self.use_node_attributes:
            feat_list = np.array([graph.node_list[i].feature_vector])
            feat_list.tolist()
            feat_list=np.append(feat_list.tolist(),aggregates)
        else:
            feat_list=np.append(feat_list.tolist(),aggregates)
        features.append(feat_list)

class ICA(Classifier):
    
    def __init__(self, local_classifier, relational_classifier, max_iteration = 10):
        self.local_classifier = local_classifier
        self.relational_classifier = relational_classifier
        self.max_iteration = 10
    
    def fit(self, graph, train_indices):
        self.local_classifier.fit(graph, train_indices)
        self.relational_classifier.fit(graph, train_indices)
    
    
    def predict(self, graph, test_indices, conditional_node_to_label_map = None):
        '''
            This function should be called only after the fit function is called.
            Implement ICA using the local classifier and the relational classifier.
            '''
        predictclf=self.local_classifier.predict(graph,test_indices)
        self.cond_mp_upd(graph,conditional_node_to_label_map,predictclf,test_indices)
        relation_predict=[]
        temp=[]
        for eachTrail in range(self.max_iteration):
            for x in test_indices:
                temp.append(x)
                rltn_pred=list(self.relational_classifier.predict(graph,temp,conditional_node_to_label_map))
                self.cond_mp_upd(graph,conditional_node_to_label_map,rltn_pred,temp)
                temp.remove(x)
        for ti in test_indices:
            relation_predict.append(conditional_node_to_label_map[graph.node_list[ti]])
        return relation_predict

    def cond_mp_upd(self,graph,conditional_map,pred,indices):
        for x in range(len(pred)):
            conditional_map[graph.node_list[indices[x]]]=pred[x]
