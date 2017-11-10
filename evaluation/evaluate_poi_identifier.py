
#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train , y_test = train_test_split( features, labels, test_size = 0.3 , random_state = 42 )

clf = DecisionTreeClassifier() 
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print (accuracy_score(pred, y_test))

### evaluation metrics, counting poi and total person in test set 
poi_count = 0
person_count = 0
for poi in pred:
    
    person_count += 1

    if poi == 1:
            
        poi_count += 1

print (poi_count)
print (person_count)


### evaluation metrics finding true poi
i = 0 
true_poi = 0
for i in range(len(pred)):
    if pred[i] == 1 and y_test[i] == 1:
        true_poi += 1

print (true_poi)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print (precision_score(y_test, pred))
print (recall_score(y_test, pred))


