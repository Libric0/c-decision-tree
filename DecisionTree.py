# Copyright (C) 2023 Fredrik Konrad <fredrik.konrad@posteo.net>

# This file is part of the c-decision-tree library.
# The c-decision-tree library is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

# The c-decision-tree library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with FocusLocus. If not, see <https://www.gnu.org/licenses/>.
from math import log2
import pandas as pd

class _DecisionTreeNode:

    def __init__(self, X:list, y:list, feature_names:list, feature_splitable:list):
        '''A single node of a decision Tree.
        
        - `X` is the list of instances without the target feature
        - `y` contains a target feature value for each instance
        - `feature_names` is the list of all feature names for the instances given in `X`
        - `feature_splitable` contains a boolean value for each feature, containing whether it can be split or not'''
        if len(X) != len(y):
            raise Exception("The length of X doesn't match the length of y")
        for instance in X:
            if len(instance) < len(feature_names):
                raise Exception(f"The instance {X} does not contain a value for every feature!")
            if len(instance) > len(feature_names):
                raise Exception(f"The instance {X} contains more values than there are features!")
        self.X = X
        self.y = y
        self.labels = set(y)
        self.feature_names = feature_names
        self.feature_splitable = feature_splitable

        self.split_feature = None
        self.children = None
    
    
    @property
    def data_set_size(self):
        return len(self.X)
    
    @property
    def has_children(self):
        return self.children != None
    
    @property
    def split_feature_index(self):
        return self.feature_names.index(self.split_feature)
    
    def getNodeMajorityLabel(self):
        '''Returns the majority label based on the training data'''
        max_count = 0
        max_label = None
        counts = {label:0 for label in self.labels}
        for y in self.y:
            counts[y] += 1
            if counts[y] > max_count:
                max_label = y
                max_count = counts[y]
        return max_label
        

    def getDatasetEntropy(self):
        '''Calculates the Entropy of the Dataset, regardless of the split'''
        entropy = 0.0
        for label in self.labels:
            count = self.y.count(label)
            entropy -= (count/len(self.y)) * log2(count/len(self.y))
        return entropy

    def getSplitEntropy(self, children):
        '''Takes a dict of children and calculates the weighted sum of the entropy of the children'''
        entropy = 0.0

        for feature_value in children.keys():
            child = children[feature_value]
            entropy += child.getEntropy()*child.data_set_size
        
        entropy /= self.data_set_size
        return entropy

    def getEntropy(self):
        '''Calculates the Entropy of this node. Either the weighted sum of its childrens' entropy or the dataSetEntropy'''
        if self.children == None:
            return self.getDatasetEntropy()
        else:
            return self.getSplitEntropy(self.children)
    
    def getSplit(self, feature_index):
        '''Returns a dict of DecisionTreeNodes that are this node's children given a split on the `feature_name`. Does !not! modify the decisionTree. For this, use `applySplit` with the values this returns'''
        children_X = dict()
        children_Y = dict()
        feature_values = set()
        # Sorting each instance into one of the children
        for x, y in zip(self.X, self.y):
            feature_values.add(x[feature_index])
            if children_X.get(x[feature_index]) == None:
                children_X[x[feature_index]] = []
                children_Y[x[feature_index]] = []
            
            children_X[x[feature_index]].append(x)
            children_Y[x[feature_index]].append(y)
        
        # Copying the splittable values, but turning the current feature unsplittable
        new_splitable = self.feature_splitable.copy()
        new_splitable[feature_index] = False

        # Creating child node for each feature value
        children = {feature_value: _DecisionTreeNode(X=children_X[feature_value], y=children_Y[feature_value], feature_names=self.feature_names, feature_splitable = new_splitable) for feature_value in feature_values}
        return children
    
    def getInformationGain(self, split):
        '''Returns the information gain for the given split, relative to the entropy of the dataset'''
        current_entropy = self.getDatasetEntropy()
        remaining_entropy = self.getSplitEntropy(split)

        return current_entropy - remaining_entropy

    def applySplit(self, feature_name, split):
        self.split_feature = feature_name
        self.children = split

    
    def id3(self, min_gain = 0, min_instances = 1, max_depth = None, depth = 0):
        '''Applies the id3 Algorithm to this node and it's children. Returns a complete, optimal decision tree based on the splittable features of this node.'''
        if depth == max_depth:
            return self
        best_feature = None
        best_split = None
        ## Setting best gain to min gain, so that we only consider splits with at least min_gain information gain
        best_gain = min_gain
        for i, feature in enumerate(self.feature_names):
            if not self.feature_splitable[i]:
                continue
            
            candidate = self.getSplit(i)
            candidate_gain = self.getInformationGain(candidate)
            
            if candidate_gain < best_gain or candidate_gain == 0:
                continue
            if min({len(child.X) for child in candidate.values()}) < min_instances:
                continue
            
            # If we passed through all the conditions, this is the best feature so far
            best_feature = feature
            best_split = candidate
            best_gain = candidate_gain

        if best_feature != None:
            self.applySplit(best_feature, best_split)
            for child in self.children.values():
                child.id3(min_gain, min_instances, max_depth = max_depth, depth=depth+1)
        return self

class DecisionTree:
    
    def __init__(self, min_gain = 0, min_instances = 0, max_depth = None):
        '''
        Decision Tree Classifier for Categorial Features
        - `min_gain` is the minimal gain that is needed for a split (default 0)
        - `min_instances` is the minimum number of instances allowed in a node (defualt 0)
        - `max_depth` is the maximum number of layers the tree has is allowed to create (default unlimited)'''

        self.X = []
        '''The list of instances without the target feature'''
        self.y = []
        '''Contains a target feature value for each instance'''
        self.feature_names = None
        '''A list of strings, one for each feature'''
        self.target = None
        '''The name of the target feature'''
        self.min_gain = min_gain
        '''Minimal gain that is needed for a split'''
        self.min_instances = min_instances
        '''Minimum number of instances allowed in a node'''
        self.max_depth = max_depth
        '''Maximum number of layers the tree has is allowed to create'''

        self.root = None
        '''The root node of the tree'''

    def getDatasetEntropy(self):
        '''Returns the entropy of the entire dataset.'''
        return self.root.getDatasetEntropy()
    
    def getEntropy(self):
        '''Returns the entropy of the current tree. Before training, it is equivalent to the dataset entropy.'''
        return self.root.getEntropy()
    
    def fit(self, data=None, target=None, X=None, y=None, feature_names=None, extend=False):
        '''Fits the decision tree to the data
        - `Data` is a dataframe of instances, including the target value
        - `target` is the name of the target value column. Can also be used to keep track of the target value for lists
        - `X` is a list of instances.
        - `y` is a list of target values.
        - `feature_names` is a list of feature names
        - `extend` determines whether `X`, `y` or `data` extend or overwrite the Dataset. If true, the dataset is extended (with duplicates).
        
        Usage:
        - Either pass a `DataFrame` to `data` and a string to `target`, referencing a column
        - Or pass an array of instances to `X`, an array of target values to `y`. Optionally pass a value to `target` and a list of strings to `feature_names`'''

        # If we don't have a root, we cannot extend. Therefore we automatically overwrite the dataset if there is no root
        extend = extend and self.root!=None

        # Making sure we either pass data or X,y
        if (data is None) and (X==None or y==None):
            raise Exception("No training data has been given.")
        if data is not None and (X!=None or y!=None):
            raise Exception("Values have been passed to both `data` and either `X` or `y`. Please only use either of them.")
        
        # Handling dataframes
        if data is not None:
            if not target in data.columns:
                raise Exception("Please also provide a target-attribute by passing a string to `target`")
            descriptive = data.loc[:,data.columns != target]
            descriptive_features = descriptive.columns.tolist()
            descriptive_vals = descriptive.values.tolist()
            target_vals = data.loc[:,data.columns == target].values.flatten().tolist()
            if not extend:
                self.X = descriptive_vals
                self.y = target_vals
                self.target = target
                self.feature_names = descriptive_features
            else:
                if self.feature_names != None:
                    if self.feature_names != list(descriptive_features):
                        raise Exception("The names of the existing features and the data-columns don't match.")
                else:
                    # If we started without feature names, we infer them from the dataset
                    self.feature_names = list(descriptive_features)
                self.X += descriptive_vals
                self.y += target_vals
                self.target = target

        # Handling lists of instances
        if X != None:
            if not extend:
                self.X = list(X)
                self.y = list(y)
                if feature_names != None:
                    self.feature_names = list(feature_names)
                self.target = target
            else:
                self.X += list(X)
                self.y += list(y)
                if self.feature_names != None and feature_names != None and self.feature_names != list(feature_names):
                    raise Exception("The existing and given feature names do not match. Please make sure you they are the same.")
                if self.feature_names is None and feature_names is not None:
                    self.feature_names = list(feature_names)

                if self.target != None and target != None and self.target != target:
                    raise Exception("The existing and given target names do not match. Please make sure they are the same.")
                if self.target == None:
                    self.target = target

        if self.X == []:
            raise Exception("The dataset contains no training examples")
        
        if self.feature_names != None:
            feature_names = self.feature_names
        else:
            feature_names = list(range(len(self.X[0])))
        print(self.X)
        print(self.y)
        self.root = _DecisionTreeNode(self.X,self.y, feature_names, [True]*len(feature_names))
        self.root.id3(min_gain=self.min_gain, min_instances=self.min_instances, max_depth=self.max_depth)

    def predict(self, X):
        '''Use the decision tree to predict a label for each of the given instances.'''
        ret = []
        for x in X:
            ret.append(self._predict_instance(x))
        return ret

    def _predict_instance(self, x):
        '''Use the decision tree to predict a label for a single instance'''

        currentNode:_DecisionTreeNode = self.root

        while currentNode.has_children:
            split_index = currentNode.split_feature_index

            # We cannot go deeper, if this never occured in a training example. Hence we break and return majority
            if currentNode.children.get(x[split_index]) == None:
                break
            currentNode = currentNode.children[x[split_index]]
        return currentNode.getNodeMajorityLabel()