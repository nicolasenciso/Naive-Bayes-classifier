from PCA import makePCA
import  pandas as  pd
import  numpy  as  np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC

url = open("rawData.csv","r")
df = pd.read_csv(url,names=['URLlong','characters','suspWord','sql','xss','crlf','kolmogorov','kullback','class'])
url.close()
y = df.iloc[:,8].values #dependent variable as y
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3)

def trainData(X_train, y_train, flagPCA, flag2D):
    if flagPCA:
        if flag2D:
            X_entreno = X_train.iloc[:,0:2].values
        else:            
            X_entreno = X_train.iloc[:,0:3].values
    else:
        X_entreno = X_train.iloc[:,0:8].values
    y_entreno = y_train
    print("*********************************************************************** \n")
    print("TRAINING DATA \n")
    print(X_train)
    #finalData = [anomalousX,anomalousY,anomalousZ,normalX,normalY,normalZ]
    dataTraining = 0
    if not flagPCA:
        dataTraining = makePCA(X_entreno,y_entreno)
    return (dataTraining,X_entreno,y_entreno)

def testData(X_test, y_test, flagPCA, flag2D):
    print("*********************************************************************** \n")
    print("TEST DATA \n")
    print(X_test)
    if flagPCA:
        if flag2D:
            X_testeo = X_test.iloc[:,0:2].values
        else:
            X_testeo = X_test.iloc[:,0:3].values
    else:
        X_testeo = X_test.iloc[:,0:8].values    
    y_testeo = y_test
    dataTest = 0
    if not flagPCA:
        dataTest = makePCA(X_testeo,y_testeo)
    return (dataTest, X_testeo, y_testeo)

def bayesClassifier(X_entreno, y_entreno, X_testeo, y_testeo, flagPCA):
    model = GaussianNB()
    model.fit(X_entreno,y_entreno)
    predicted_labels = model.predict(X_testeo)
    print("****** PREDICTED MODEL ********")
    print(predicted_labels)
    print("********* TEST MODEL ************")
    print(y_testeo)
    print("\n")
    listPredicted = predicted_labels.tolist()
    listGivenTest = y_testeo.tolist()
    count, countMatched = 0,0
    #print("--GIVEN --- PREDICTED")
    for label in listPredicted:
        if str(listGivenTest[count]) == str(label):
            countMatched += 1
        #print(str(listGivenTest[count])+" --- "+str(label))
        #print("------------------------")
        count += 1
    print("RESULTS: ")
    print("--------------------------------------")
    print(count)
    print("Matched: "+str(countMatched)+" / "+str(count))
    print("Ratio: "+str(float(countMatched)/float(count)))
    accuracy = accuracy_score(y_testeo, predicted_labels)
    print("\n*** ACCURACY GAUSSIAN **************")
    print(accuracy)
    if not flagPCA:
        model = MultinomialNB()
        model.fit(X_entreno,y_entreno)
        predicted_labels = model.predict(X_testeo)
        accuracy = accuracy_score(y_testeo, predicted_labels)
        listPredicted = predicted_labels.tolist()
        listGivenTest = y_testeo.tolist()
        count, countMatched = 0,0
        #print("--GIVEN --- PREDICTED")
        for label in listPredicted:
            if str(listGivenTest[count]) == str(label):
                countMatched += 1
            #print(str(listGivenTest[count])+" --- "+str(label))
            #print("------------------------")
            count += 1
        print("RESULTS: ")
        print("--------------------------------------")
        print(count)
        print("Matched: "+str(countMatched)+" / "+str(count))
        print("Ratio: "+str(float(countMatched)/float(count)))
        print("\n*** ACCURACY MULTINOMIAL **************")
        print(accuracy)
    

def toCSVfile(dataTraining, dataTest):
    data = open("dataPCA.txt","w")
    anomalousX,anomalousY,anomalousZ,normalX,normalY,normalZ = dataTraining
    count = 0
    for line in anomalousX:
        X = str(line)
        Y = str(anomalousY[count])
        Z = str(anomalousZ[count])
        data.writelines(X+","+Y+","+Z+","+"anomalous"+"\n")
        count += 1
    count = 0
    for line in normalX:
        X = str(line)
        Y = str(normalY[count])
        Z = str(normalZ[count])
        data.writelines(X+","+Y+","+Z+","+"normal"+"\n")
        count += 1
    anomalousX,anomalousY,anomalousZ,normalX,normalY,normalZ = dataTest
    count = 0
    for line in anomalousX:
        X = str(line)
        Y = str(anomalousY[count])
        Z = str(anomalousZ[count])
        data.writelines(X+","+Y+","+Z+","+"anomalous"+"\n")
        count += 1
    count = 0
    for line in normalX:
        X = str(line)
        Y = str(normalY[count])
        Z = str(normalZ[count])
        data.writelines(X+","+Y+","+Z+","+"normal"+"\n")
        count += 1
    
    data.close()


def toCSVfile2D(dataTraining, dataTest):
    data = open("dataPCA2D.txt","w")
    anomalousX,anomalousY,anomalousZ,normalX,normalY,normalZ = dataTraining
    count = 0
    for line in anomalousX:
        X = str(line)
        Y = str(anomalousY[count])
        data.writelines(X+","+Y+","+"anomalous"+"\n")
        #data.writelines(X+","+Y+","+"1"+"\n")
        count += 1
    count = 0
    for line in normalX:
        X = str(line)
        Y = str(normalY[count])
        data.writelines(X+","+Y+","+"normal"+"\n")
        #data.writelines(X+","+Y+","+"0"+"\n")
        count += 1
    anomalousX,anomalousY,anomalousZ,normalX,normalY,normalZ = dataTest
    count = 0
    for line in anomalousX:
        X = str(line)
        Y = str(anomalousY[count])
        data.writelines(X+","+Y+","+"anomalous"+"\n")
        #data.writelines(X+","+Y+","+"1"+"\n")
        count += 1
    count = 0
    for line in normalX:
        X = str(line)
        Y = str(normalY[count])
        data.writelines(X+","+Y+","+"normal"+"\n")
        #data.writelines(X+","+Y+","+"0"+"\n")
        count += 1
    
    data.close()

def SVMclassifier(X_entreno,y_entreno,X_testeo, y_testeo):
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_entreno,y_entreno)
    y_pred = svclassifier.predict(X_testeo)
    print("\n ***** RESULTS SVM LINEAR ********\n")
    print(confusion_matrix(y_testeo,y_pred))  
    print(classification_report(y_testeo,y_pred))  

    svclassifier = SVC(kernel='poly', degree=8)
    svclassifier.fit(X_entreno,y_entreno)
    y_pred = svclassifier.predict(X_testeo)
    print("\n ***** RESULTS SVM POLYNOMIAL ********\n")
    print(confusion_matrix(y_testeo,y_pred))  
    print(classification_report(y_testeo,y_pred))

    svclassifier = SVC(kernel='rbf')
    svclassifier.fit(X_entreno,y_entreno)
    y_pred = svclassifier.predict(X_testeo)
    print("\n ***** RESULTS SVM GAUSSIAN ********\n")
    print(confusion_matrix(y_testeo,y_pred))  
    print(classification_report(y_testeo,y_pred))

    svclassifier = SVC(kernel='sigmoid')
    svclassifier.fit(X_entreno,y_entreno)
    y_pred = svclassifier.predict(X_testeo)
    print("\n ***** RESULTS SVM SIGMOID ********\n")
    print(confusion_matrix(y_testeo,y_pred))  
    print(classification_report(y_testeo,y_pred))

#from raw data
dataTraining,X_entreno,y_entreno = trainData(X_train,y_train,False,False)
dataTest, X_testeo, y_testeo = testData(X_test,y_test,False,False)
print("\n=================== FROM ORIGINAL DATA WITH 8 FEATURES ==========\n")
bayesClassifier(X_entreno,y_entreno,X_testeo,y_testeo,False)
#SVMclassifier(X_entreno,y_entreno,X_testeo,y_testeo)

#from PCA
toCSVfile(dataTraining,dataTest)

dataTraining2D,X_entreno2D,y_entreno2D = trainData(X_train,y_train,False,True)
dataTest2D, X_testeo2D, y_testeo2D = testData(X_test,y_test,False,True)
toCSVfile2D(dataTraining, dataTest)

url = open("dataPCA.csv","r")
url2D = open("dataPCA2D.csv","r")

df = pd.read_csv(url,names=['PCA1','PCA2','PCA3','class'])
df2D = pd.read_csv(url2D,names=['PCA1','PCA2','class'])
url.close()
url2D.close()

y = df.iloc[:,3].values #dependent variable as y
y2D = df2D.iloc[:,2].values

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3)
X_train2D, X_test2D, y_train2D, y_test2D = train_test_split(df2D, y2D, test_size=0.3)

dataTraining,X_entreno,y_entreno = trainData(X_train,y_train,True,False)
dataTraining2D,X_entreno2D,y_entreno2D = trainData(X_train2D,y_train2D,True,True)

dataTest, X_testeo, y_testeo = testData(X_test,y_test,True,False)
dataTest2D, X_testeo2D, y_testeo2D = testData(X_test2D,y_test2D,True,True)

print("\n========================== FROM PCA WITH 3 PCAs ==================\n")
bayesClassifier(X_entreno,y_entreno,X_testeo,y_testeo,True)
SVMclassifier(X_entreno, y_entreno, X_testeo, y_testeo)

print("\n========================== FROM PCA WITH 2 PCAs ===================\n")
bayesClassifier(X_entreno2D,y_entreno2D,X_testeo2D,y_testeo2D,True)
SVMclassifier(X_entreno2D, y_entreno2D, X_testeo2D, y_testeo2D)
