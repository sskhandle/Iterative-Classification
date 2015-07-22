from data_utils import load_linqs_data
from classifiers import LocalClassifier


from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-content_file', help='The path to the content file.')
    parser.add_argument('-cites_file', help='The path to the cites file.')    
    parser.add_argument('-classifier', default='sklearn.naive_bayes.MultinomialNB', help='The underlying classifier.')
    parser.add_argument('-num_folds', type=int, default=10, help='The number of folds.')
    
    args = parser.parse_args()
    
    graph, domain_labels = load_linqs_data(args.content_file, args.cites_file)
    
    kf = KFold(n=len(graph.node_list), n_folds=args.num_folds, shuffle=True, random_state=42)    
    
    
    accuracies = []
    
    cm = None
    
    for train, test in kf:
        clf = LocalClassifier(args.classifier)
        clf.fit(graph, train)
        y_pred = clf.predict(graph, test)
        y_true = [graph.node_list[t].label for t in test]
        accuracies.append(accuracy_score(y_true, y_pred))
        if cm is None:
            cm = confusion_matrix(y_true, y_pred, labels = domain_labels)
        else:
            cm += confusion_matrix(y_true, y_pred, labels = domain_labels)

    
    print accuracies
    print "Mean accuracy: %0.4f +- %0.4f" % (np.mean(accuracies), np.std(accuracies))
    print cm
        
    
    
conditional_map={}
X=[]
for a in train_indices:
            conditional_map[graph.node_list[a]]=graph.node_list[a].label
for a in train_indices:
        if self.use_node_attributes==True:
            fi = graph.node_list[a].feature_vector
            fis = self.aggregator.aggregate(graph,graph.node_list[a],conditional_map)
            x=fi + fis
        else:
            x= self.aggregator.aggregate(graph,graph.node_list[a].node_id,conditional_map)
        X.append(x)
y=[graph.node_list[a].label for a in train_indices]
self.clf.fit(X,y)