from PCA import makePCA
import  pandas as  pd
import  numpy  as  np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

url = open("dataPCA.csv","r")

df = pd.read_csv(url,names=['URLlong','characters','suspWord','sql','xss','crlf','kolmogorov','kullback','class'])
url.close()
y = df.iloc[:,8].values #dependent variable as y
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3)

def trainData(X_train, y_train):
    X_entreno = X_train.iloc[:,0:8].values
    y_entreno = y_train
    print("*********************************************************************** \n")
    print("TRAINING DATA \n")
    print(X_train)
    print("\n")
    #finalData = [anomalousX,anomalousY,anomalousZ,normalX,normalY,normalZ]
    dataTraining = makePCA(X_entreno,y_entreno)
    return (dataTraining,X_entreno,y_entreno)

def testData(X_test, y_test):
    print("*********************************************************************** \n")
    print("TEST DATA \n")
    print(X_test)
    print("\n")
    X_testeo = X_test.iloc[:,0:8].values
    y_testeo = y_test
    dataTest = makePCA(X_testeo,y_testeo)
    return (dataTest, X_testeo, y_testeo)

def bayesClassifier(X_entreno, y_entreno, X_testeo, y_testeo):
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

#from raw data
dataTraining,X_entreno,y_entreno = trainData(X_train,y_train)
dataTest, X_testeo, y_testeo = testData(X_test,y_test)
bayesClassifier(X_entreno,y_entreno,X_testeo,y_testeo)

#from PCA
