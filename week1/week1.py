#KNN implementation from scratch with manhattan distance
import numpy as np
from collections import Counter
import pandas as pd

#define the method of calculating distance
def manhattan(trainInstance,testInstance):
    distance = sum(abs(feature1-feature2) for feature1,feature2 in zip(trainInstance,testInstance))
    return distance

def knn_classifier(trainSet,target,testSet,k):

    result = {}

    #Iterate through the test set

    for test in range(testSet.shape[0]):
        
        distances = []

        #calculate the distance between each test point with the whole of training set.
        for train in range(trainSet.shape[0]):
            distance = manhattan(trainSet[train],testSet[test])
            distances.append((target[train],distance))

        #sort the distance in ascending order
        sortedDistances = sorted(distances,key=lambda x:x[1])
        
        #culmulate the results of the closest points to consider
        possiblePredictions = [contester for contester,_ in sortedDistances[:k]]

        #get the most popular class in the in pool of possible predictions
        prediction = Counter(possiblePredictions).most_common()[0][0]

        result[test] = prediction

    return result

data = pd.read_csv('https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv')

encoder = {
    'Virginica':3,
    'Versicolor':2,
    'Setosa':1
}

data['variety'] = data['variety'].map(encoder)

data = data.sample(frac=1,random_state=1000).reset_index()

trainSet = np.array(data.iloc[:100,1:5])
target = np.array(data['variety'])
testSet = np.array(data.iloc[100:,1:5])

testResult = knn_classifier(trainSet,target,testSet,5)

errors = 0
for i in zip(list(testResult.values()),list(data.iloc[100:,5])):
    if i[0]!=i[1]:
        errors+=1

print(f"Our KNN predicted {errors} samples wrongly")