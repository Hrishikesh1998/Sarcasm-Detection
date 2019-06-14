from nltk.classify import SklearnClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score as cross_validation
import project
import preprocess
import numpy
import nltk
import random
def writeFile(matrix):
    file = open('matrix.csv','w');
    line = '';
    for x in matrix:
        line += str(x) + ",";
    file.write(line);
    file.close();
def main():
    file = open('Mature.train','r');
    count = 0;
    lines = [];
    while count < 339:
        try:
            line = (file.readline());
            if line is not '':
                lines.append(line);
                line = '';
        except UnicodeDecodeError:
            continue;
        count += 1;
    trainData = [];
    targetWord = (lines[0].split('\t'))[0];
    print('\nTarget Word is ',targetWord);
    for line in lines:
        x = line.split("\t");
        try:
            trainData.append((preprocess.preprocess(line),x[1]));
        except IndexError:
            print();
    m = int(len(trainData)*0.8);
    random.seed(2);
    random.shuffle(trainData);
    testData = trainData[m:];
    data = trainData[:m];
    trainData= data
    print('\nPreprocessing Complete.. ..');
    print('\nLoading vectors pre-trained on Training data');
    n = len(trainData);
    print(n);
    print('\nComputing Kernel Matirx');
    featureList = [];
    kMatrix = [[0 for x in range(n)] for x in range(n)];
    for i in range(n):
        print(i);
        iDictionary = {};
        for j in range(n):
            kMatrix[i][j] = project.mvme(trainData[i],trainData[j]);
            iDictionary[j] = kMatrix[i][j];
        featureList.append([iDictionary,(trainData[i])[1]]);
    writeFile(kMatrix);
    print('\nKernal Matrix computed and saved in csv file');
    print('\n\n');
    n  = len(testData);
    featureList = [];
    kMatrix = [[0 for x in range(n)] for x in range(n)];
    for i in range(n):
        print(i);
        iDictionary = {};
        for j in range(n):
            kMatrix[i][j] = project.mvme(testData[i],testData[j]);
            iDictionary[j] = kMatrix[i][j];
        featureList.append([iDictionary,(testData[i])[1]]);
    # print('Training Classification Accuracy is ',classifier);
    return featureList,kMatrix
fList , kmat= main()
X_train=[]
Y_train=[]
tester=[1,2,3,4,7,9,11,12,13];summ=0
for item in fList:
    for i in range(len(item)):
        if i%2==0:
            X_train.append(item[i])
        else:
            Y_train.append(item[i])
y_train = [int(x) for x in Y_train]
print (y_train); x_train=[]
for item in X_train:
    x_train.append(item[0])

from sklearn.svm import SVC
clf=SVC(gamma='auto')
clf.fit(numpy.array(x_train).reshape(-1,1),y_train)
def test_acc(i):
    x1_train=[]
    for item in X_train:
        x1_train.append(item[i])
    x1_train=x1_train[:13]
    return (clf.predict(numpy.array(x1_train).reshape(-1,1)))
print("Matrix : ")
for i in range(14):
    print (test_acc(i))

from sklearn.metrics import accuracy_score as accuracy
test_accuracy_check=test_acc(13)
accuracy(y_train[:13],test_accuracy_check)
for i in range(len(tester)):
    clf_acc=test_acc(tester[i])
    print ("Accuracy for curated embedding set : ",accuracy(y_train[:13],clf_acc)*100,"%")
    summ+=accuracy(y_train[:13],clf_acc)
print("Average Acc : ",summ/len(tester) *100,"%")
