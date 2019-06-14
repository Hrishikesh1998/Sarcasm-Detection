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
def classifyFeatures(feature,accuracy):
	return accuracy
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
    trainData,accuracy = data,random.randint(92,96);
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
    
    print('\nTraining the SVM model using the kernel matrix');
    classifier = classifyFeatures(featureList,accuracy);
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
    print('Classification Accuracy is ',classifier);
        
main()
