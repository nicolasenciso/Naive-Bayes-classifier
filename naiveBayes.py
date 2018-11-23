from PCA import makePCA
import  pandas as  pd
import  numpy  as  np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

url = open("rawData.csv","r")
df = pd.read_csv(url,names=['URLlong','characters','suspWord','sql','xss','crlf','kolmogorov','kullback','class'])
url.close()
y = df.iloc[:,8].values #dependent variable as y
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3)

def trainData(X_train, y_train, flagPCA):
    if flagPCA:
        X_entreno = X_train.iloc[:,0:3].values
    else:
        X_entreno = X_train.iloc[:,0:8].values
    y_entreno = y_train
    print("*********************************************************************** \n")
    print("TRAINING DATA \n")
    print(X_train)
    print("\n")
    #finalData = [anomalousX,anomalousY,anomalousZ,normalX,normalY,normalZ]
    dataTraining = 0
    if not flagPCA:
        dataTraining = makePCA(X_entreno,y_entreno)
    return (dataTraining,X_entreno,y_entreno)

def testData(X_test, y_test, flagPCA):
    print("*********************************************************************** \n")
    print("TEST DATA \n")
    print(X_test)
    print("\n")
    if flagPCA:
        X_testeo = X_test.iloc[:,0:3].values
    else:
        X_testeo = X_test.iloc[:,0:8].values    
    y_testeo = y_test
    dataTest = 0
    if not flagPCA:
        dataTest = makePCA(X_testeo,y_testeo)
    return (dataTest, X_testeo, y_testeo)

def bayesClassifier(X_entreno, y_entreno, X_testeo, y_testeo, flagPCA):
    print("\n")
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
    print("\n")
    print("RESULTS: ")
    print("--------------------------------------")
    print(count)
    print("Matched: "+str(countMatched)+" / "+str(count))
    print("Ratio: "+str(float(countMatched)/float(count)))
    accuracy = accuracy_score(y_testeo, predicted_labels)
    print("*** ACCURACY GAUSSIAN **************")
    print(accuracy)
    print("\n")
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
        print("\n")
        print("RESULTS: ")
        print("--------------------------------------")
        print(count)
        print("Matched: "+str(countMatched)+" / "+str(count))
        print("Ratio: "+str(float(countMatched)/float(count)))
        print("*** ACCURACY MULTINOMIAL **************")
        print(accuracy)
        print("\n")

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


#from raw data
print("***** FROM ORIGINAL DATA WITH 8 FEATURES *******")
dataTraining,X_entreno,y_entreno = trainData(X_train,y_train,False)
dataTest, X_testeo, y_testeo = testData(X_test,y_test,False)
bayesClassifier(X_entreno,y_entreno,X_testeo,y_testeo,False)

#from PCA
print("***** FROM PCA WITH 3 PCAs *******")
toCSVfile(dataTraining,dataTest)
url = open("dataPCA.csv","r")
df = pd.read_csv(url,names=['PCA1','PCA2','PCA3','class'])
url.close()
y = df.iloc[:,3].values #dependent variable as y
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3)
dataTraining,X_entreno,y_entreno = trainData(X_train,y_train,True)
dataTest, X_testeo, y_testeo = testData(X_test,y_test,True)
bayesClassifier(X_entreno,y_entreno,X_testeo,y_testeo,True)
