# Load libraries
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import label_binarize
from sklearn.cluster import KMeans
from random import randint
import statistics


#############################################################################




def perform_stack(df, X, y):
    kmeans = KMeans(n_clusters=5)
    nn = KNeighborsClassifier(n_neighbors=3)
    X['kmeanspd'] = kmeans.fit_predict(X)
    total_acc=[]
    total_fscore = []
    for i in range(10):
        acc = []
        f_scores = []
        for index, (train_ind, test_ind) in enumerate(KFold(X.shape[0], n_folds=10, shuffle=True, random_state=None)):
            X_train, X_test = X.ix[train_ind,:], X.ix[test_ind, :]
            y_train, y_test = y[train_ind], y[test_ind]

            nn.fit(X_train, y_train)
            preds = nn.predict(X_test)
            acc.append(accuracy_score(y_test, preds))
            f_scores.append(f1_score(y_test, preds, average='weighted'))
        total_acc.append(np.array(acc).mean())
        total_fscore.append(np.array(f_scores).mean())
    print("Stacking:")
    print("Mean accuracy: " + str(np.array(total_acc).mean()))
    print("Mean f_score: " + str(np.array(total_fscore).mean()))


def BreastCancerFunc(model, LongDataset):
    # Load dataset
    name = ['Sample code number' ,'Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion' ,'Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class'] 

    dataset = pandas.read_csv("breast-cancer-wisconsin/wdbc.data.txt",header=None)
    print(dataset.shape)
    dataset[1] = np.where(dataset[1] == 'B', 0, 1)
    array = dataset.values
    data = array[:,2:31]
    label = array[:,1]
    count =0 
    f_score=[]
    accuracy=[]
    a_auc=[]
    for name,model in models:
        print ("-----------------------"+ name + "---------------------------------")
        meanfinder = []
        ffinder = []
        aucfinder = []        
        for x in range(1,11):
            kf = cross_validation.KFold(n = 569, n_folds=10,shuffle = True, random_state = None)
            acc = 0
            i = 0
            f1 = 0
            auc = 0
            for train_index, test_index in kf:
                X_train, X_test = data[train_index], data[test_index]
                y_train, y_test = label[train_index], label[test_index]
                model.fit(X_train,y_train)
                prediction = model.predict(X_test)
                acc = acc + accuracy_score(y_test, prediction)
                auc = auc + roc_auc_score(y_test, prediction)
                f1 = f1 + f1_score(y_test, prediction)                
                i = i+1
            meanfinder.append(acc/10)
            ffinder.append(f1/10)        
            aucfinder.append(auc/10)
            print("Iteration number " + str(x))
            print("Accuracy using "+ name + " " + str(acc / 10))
            print("F measure using "+ name + " " + str(f1 / 10))   
            print("Area under curve using "+ name + " " + str(auc / 10))
            print ()
   
        j=sum(ffinder) / float(len(ffinder))  
        k=sum(meanfinder) / float(len(meanfinder))  
        l=sum(aucfinder) / float(len(aucfinder))  
        
        print ("--------------------------------------------------")
        
        print("Final Accuracy using " +name + " " + str(k) )
        #BreastCancer[count]['Accuracy']= k
               
        print("Final F measure using " +name + " " + str(j) )                
        
        #BreastCancer[count]['F-Measure']= j
        print("Final Area Under Curve using " +name + " " + str(l) )                
        
        BreastCancer[count]['AUC'] = l        
        print ("--------------------------------------------------")
        count=count + 1
        accuracy.append(k) 
        f_score.append(j)
        a_auc.append(l)
    #print (accuracy,f_score,a_auc)
    print("Standard deviation of BreastCancer at accuracy measure: "+str(statistics.stdev(accuracy)))
    print("Standard deviation of BreastCancer at f_score measure: "+str(statistics.stdev(f_score)))
    print("Standard deviation of BreastCancer at auc measure: "+str(statistics.stdev(a_auc)))
    #perform_stack(dataset,data,label)


    
#############################################################################    


def AbaloneFunc(model, LongDataset):
    # Load dataset
    dataset = pandas.read_csv("abalone/abalone.data.txt", names=['Sex', 'Length', 'Diameter', 'Height', 'Whole Weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings'])
    print(dataset.shape)
    #dataset[1] = np.where(dataset[1] == 'B', 0, 1)
    dataset.Sex = dataset.Sex.map({'M':0, 'F':1, 'I':2})
    array = dataset.values
    
    #print (dataset)
    
    data = array[:,1:8]
    label = array[:,0]
    count =0 
    f_score=[]
    accuracy=[]
    a_auc=[]
    for name,model in models:
        print ("-----------------------"+ name + "---------------------------------")
        meanfinder = []
        ffinder = []
        aucfinder = []        
        for x in range(1,11):
            kf = cross_validation.KFold(n = 4177, n_folds=10,shuffle = True, random_state = None)
            acc = 0
            i = 0
            f1 = 0
            auc = 0
            for train_index, test_index in kf:
                slabel = np.where(label == randint(0,2), 0, 1)
                X_train, X_test = data[train_index], data[test_index]
                y_train, y_test = slabel[train_index], slabel[test_index]
                model.fit(X_train,y_train)
                prediction = model.predict(X_test)
                acc = acc + accuracy_score(y_test, prediction)
                auc = auc + roc_auc_score(y_test, prediction)
                f1 = f1 + f1_score(y_test, prediction)                
                i = i+1
            meanfinder.append(acc/10)
            ffinder.append(f1/10)        
            aucfinder.append(auc/10)
            print("Iteration number " + str(x))
            print("Accuracy using "+ name + " " + str(acc / 10))
            print("F measure using "+ name + " " + str(f1 / 10))   
            print("Area under curve using "+ name + " " + str(auc / 10))
            print ()
   
        j=sum(ffinder) / float(len(ffinder))  
        k=sum(meanfinder) / float(len(meanfinder))  
        l=sum(aucfinder) / float(len(aucfinder))  
        
        
        print ("--------------------------------------------------")
        
        print("Final Accuracy using " +name + " " + str(k) )
       # Abalone[count]['Accuracy']= k
        print("Final F measure using " +name + " " + str(j) )                
        #Abalone[count]['F-Measure']= j
        print("Final Area Under Curve using " +name + " " + str(l) )                
        #Abalone[count]['AUC'] = l        
        print ("--------------------------------------------------")
        count=count + 1
        accuracy.append(k) 
        f_score.append(j)
        a_auc.append(l)
    #print (accuracy,f_score,a_auc)
    print("Standard deviation of Abalone at accuracy measure: "+str(statistics.stdev(accuracy)))
    print("Standard deviation of Abalone at f_score measure: "+str(statistics.stdev(f_score)))
    print("Standard deviation of Abalone at auc measure: "+str(statistics.stdev(a_auc)))
        
    #print (LongDataset)
    
    
    
#############################################################################    


def Arrythemia(model, LongDataset):
    # Load dataset
    dataset = pandas.read_csv('arrhythmia/arrhythmia.data.txt', header=None)
   # print(dataset.shape)
    #print(dataset)
    #dataset[1] = np.where(dataset[1] == 'B', 0, 1)
    dataset = dataset.replace('?', 0)
    
    array = dataset.values
    
    data = array[:,2:278]
    label = array[:,1].astype(int)
    
    f_score=[]
    accuracy=[]
    a_auc=[]
    count =0 
    for name,model in models:
        print ("-----------------------"+ name + "---------------------------------")
        meanfinder = []
        ffinder = []
        aucfinder = []        
        for x in range(1,11):
            kf = cross_validation.KFold(n = 451, n_folds=10,shuffle = True, random_state = None)
            acc = 0
            i = 0
            f1 = 0
            auc = 0
            for train_index, test_index in kf:
                X_train, X_test = data[train_index], data[test_index]
                y_train, y_test = label[train_index], label[test_index]
                model.fit(X_train,y_train)
                prediction = model.predict(X_test)
                acc = acc + accuracy_score(y_test, prediction)
                auc = auc + roc_auc_score(y_test, prediction)
                f1 = f1 + f1_score(y_test, prediction)                
                i = i+1
            meanfinder.append(acc/10)
            ffinder.append(f1/10)        
            aucfinder.append(auc/10)
            
            j=sum(ffinder) / float(len(ffinder))  
            k=sum(meanfinder) / float(len(meanfinder))  
            l=sum(aucfinder) / float(len(aucfinder))  
            
            
            print ("--------------------------------------------------")
        
            
            
            print("Iteration number " + str(x))
            print("Final Accuracy using " +name + " " + str(k) )
            #Aarrhythmia[count]['Accuracy']= k
            print("Final F measure using " +name + " " + str(j) )                
            #Aarrhythmia[count]['F-Measure']= j
            print("Final Area Under Curve using " +name + " " + str(l) )                
            #Aarrhythmia[count]['AUC'] = l    
            print ()
   
       
        print ("--------------------------------------------------")
        count=count + 1
        
        accuracy.append(k) 
        f_score.append(j)
        a_auc.append(l)
    #print (accuracy,f_score,a_auc)
    print("Standard deviation of Arrythemia at accuracy measure: "+str(statistics.stdev(accuracy)))
    print("Standard deviation of Arrythemia at f_score measure: "+str(statistics.stdev(f_score)))
    print("Standard deviation of Arrythemia at auc measure: "+str(statistics.stdev(a_auc)))
    #print (LongDataset)
    #perform_stack(dataset,data,label)



def hepatitasFunc(models,LongDataset):
    name = [ 'Class','AGE','SEX','STEROID','ANTIVIRALS','FATIGUE','MALAISE','ANOREXIA','LIVER BIG','LIVER FIRM','SPLEEN PALPABLE','SPIDERS','ASCITES','VARICES','BILIRUBIN','ALK PHOSPHATE','SGOT','ALBUMIN','PROTIME','HISTOLOGY']
    dataset = pandas.read_csv("hepatitas/hepatitis.data.txt",names=name)
    dataset = dataset.replace('?',1)
    #print(dataset.shape)
    #print(dataset)
    
    dataset['Class'] = np.where(dataset['Class'] == 1, 0, 1)
    print (dataset)
    
    array = dataset.values
    data = array[:,1:19]
    label = array[:,0].astype(int)
    count =0 
    
    f_score=[]
    accuracy=[]
    a_auc=[]
    for name,model in models:
        print ("-----------------------"+ name + "---------------------------------")
        meanfinder = []
        ffinder = []
        aucfinder = []        
        for x in range(1,11):
            kf = cross_validation.KFold(n = 155, n_folds=10,shuffle = True, random_state = None)
            acc = 0
            i = 0
            f1 = 0
            auc = 0
            for train_index, test_index in kf:
                X_train, X_test = data[train_index], data[test_index]
                y_train, y_test = label[train_index], label[test_index]
                model.fit(X_train,y_train)
                prediction = model.predict(X_test)
                acc = acc + accuracy_score(y_test, prediction)
                
                #auc = auc + roc_auc_score(y_test, prediction,average=None)
                f1 = f1 + f1_score(y_test, prediction)                
                i = i+1
            meanfinder.append(acc/10)
            ffinder.append(f1/10)        
            aucfinder.append(auc/10)
            print("Iteration number " + str(x))
            print("Accuracy using "+ name + " " + str(acc / 10))
            print("F measure using "+ name + " " + str(f1 / 10))   
           # print("Area under curve using "+ name + " " + str(auc / 10))
            print ()
   
        j=sum(ffinder) / float(len(ffinder))  
        k=sum(meanfinder) / float(len(meanfinder))  
        #l=sum(aucfinder) / float(len(aucfinder))  
        
        print ("--------------------------------------------------")
        
        print("Final Accuracy using " +name + " " + str(k) )
        Hepatitas[count]['Accuracy']= k
        print("Final F measure using " +name + " " + str(j) )                
        Hepatitas[count]['F-Measure']= j
        #print("Final Area Under Curve using " +name + " " + str(l) )                
        #Hepatitas[count]['AUC'] = l        
        print ("--------------------------------------------------")
        count=count + 1
        accuracy.append(k) 
        f_score.append(j)
        #a_auc.append(l)
    #print (accuracy,f_score,a_auc)
    print("Standard deviation of hepatitis at accuracy measure: "+str(statistics.stdev(accuracy)))
    print("Standard deviation of hepatitis at f_score measure: "+str(statistics.stdev(f_score)))
    #print("Standard deviation of BreastCancer at auc measure: "+str(statistics.stdev(a_auc)))
    #print (LongDataset)




def nurseryFunc(models,LongDataset):
    # Load dataset
    dataset = pandas.read_csv("nursery/nursery.data.txt",names=['parents','has_nurs','form','children','housing','finance','social','health','Class'])
    dataset.Class = dataset.Class.map({'not_recom':0,'recommend':1,'very_recom':2,'priority':3,'spec_prior':4})
    dataset.parents = dataset.parents.map({'usual':0,'pretentious':1,'great_pret':2})
    dataset.has_nurs = dataset.has_nurs.map({'proper':0,'less_proper':1,'improper':2,'critical':3,'very_crit':4})
    dataset.form = dataset.form.map({'complete':0,'completed':1,'incomplete':2,'foster':3})
    dataset.children = dataset.children.map({'1':0,'2':1,'3':2,'more':3,'spec_prior':4})
    dataset.housing = dataset.housing.map({'convenient':0,'less_conv':1,'critical':2})
    dataset.finance = dataset.finance.map({'convenient':0,'inconv':1})
    dataset.social = dataset.social.map({'nonprob':0,'slightly_prob':1,'problematic':2})
    dataset.health = dataset.health.map({'recommended':0,'priority':1,'not_recom':2})
    #print(dataset.shape)
    
#    array = dataset.values
    data = dataset.ix[:,'parents':'health']
    #label = dataset['lclass']
    #dataset['Class'] = np.where(dataset['Class'] == 0, 0, 1)
    #slabel = label_binarize(label, classes=[0, 1, 2, 3,4])
    label = dataset['Class']
    #print(dataset)
    
    count =0 
    
    f_score=[]
    accuracy=[]
    a_auc=[]
    for name,model in models:
        print ("-----------------------"+ name + "---------------------------------")
        meanfinder = []
        ffinder = []
        aucfinder = []        
        for x in range(1,11):
            kf = cross_validation.KFold(n = 12960, n_folds=10,shuffle = True, random_state = None)
            acc = 0
            i = 0
            f1 = 0
            auc = 0
            
            for train_index, test_index in kf:
                slabel = np.where(label == randint(0,5), 0, 1)
                X_train, X_test = data.ix[train_index,:], data.ix[test_index,:]
                y_train, y_test = slabel[train_index], slabel[test_index]
                model.fit(X_train,y_train)
                prediction = model.predict(X_test)
                acc = acc + accuracy_score(y_test, prediction)
                #auc = auc + roc_auc_score(y_test, prediction)
                f1 = f1 + f1_score(y_test, prediction)                
                i = i+1
            meanfinder.append(acc/10)
            ffinder.append(f1/10)        
            aucfinder.append(auc/10)
            print("Iteration number " + str(x))
            print("Accuracy using "+ name + " " + str(acc / 10))
            print("F measure using "+ name + " " + str(f1 / 10))   
            #print("Area under curve using "+ name + " " + str(auc / 10))
            print ()
   
        j=sum(ffinder) / float(len(ffinder))  
        k=sum(meanfinder) / float(len(meanfinder))  
        #l=sum(aucfinder) / float(len(aucfinder))  
        
        print ("--------------------------------------------------")
        
        print("Final Accuracy using " +name + " " + str(k) )
        Nursery[count]['Accuracy']= k
        print("Final F measure using " +name + " " + str(j) )                
        Nursery[count]['F-Measure']= j
        #print("Final Area Under Curve using " +name + " " + str(l) )                
        #BreastCancer[count]['AUC'] = l        
        print ("--------------------------------------------------")
        count=count + 1
        accuracy.append(k) 
        f_score.append(j)
        #a_auc.append(l)
    #print (accuracy,f_score,a_auc)
    print("Standard deviation of Nursery at accuracy measure: "+str(statistics.stdev(accuracy)))
    print("Standard deviation of Nursery at f_score measure: "+str(statistics.stdev(f_score)))
    #print("Standard deviation of BreastCancer at auc measure: "+str(statistics.stdev(a_auc)))
    #print (LongDataset)
    #perform_stack(dataset,data,label)
    
    
    
def glassDataset(models,LongDataset):
    dataset = pandas.read_csv("Glass/glass.data.txt",names=['id','ri','na','mg','al','si','ka','ca','ba','fe','Class'])
    print(dataset.shape)
    #print(dataset)
    
    dataset['Class'] = np.where(dataset['Class'] == 1, 0, 1)
    array = dataset.values
    data = array[:,0:9]
    label = dataset['Class']
    count =0 
    #print(dataset)
    
    
    f_score=[]
    accuracy=[]
    a_auc=[]
    for name,model in models:
        print ("-----------------------"+ name + "---------------------------------")
        meanfinder = []
        ffinder = []
        aucfinder = []        
        for x in range(1,11):
            kf = cross_validation.KFold(n = 214, n_folds=10,shuffle = True, random_state = None)
            acc = 0
            i = 0
            f1 = 0
            auc = 0
            for train_index, test_index in kf:
                X_train, X_test = data[train_index], data[test_index]
                y_train, y_test = label[train_index], label[test_index]
                model.fit(X_train,y_train)
                prediction = model.predict(X_test)
                acc = acc + accuracy_score(y_test, prediction)
                auc = auc + roc_auc_score(y_test, prediction)
                f1 = f1 + f1_score(y_test, prediction)                
                i = i+1
            meanfinder.append(acc/10)
            ffinder.append(f1/10)        
            aucfinder.append(auc/10)
            print("Iteration number " + str(x))
            print("Accuracy using "+ name + " " + str(acc / 10))
            print("F measure using "+ name + " " + str(f1 / 10))   
            print("Area under curve using "+ name + " " + str(auc / 10))
            print ()
   
        j=sum(ffinder) / float(len(ffinder))  
        k=sum(meanfinder) / float(len(meanfinder))  
        l=sum(aucfinder) / float(len(aucfinder))  
        
        print ("--------------------------------------------------")
        
        print("Final Accuracy using " +name + " " + str(k) )
        #Glass[count]['Accuracy']= k
        print("Final F measure using " +name + " " + str(j) )                
        #Glass[count]['F-Measure']= j
        print("Final Area Under Curve using " +name + " " + str(l) )                
        #Glass[count]['AUC'] = l        
        print ("--------------------------------------------------")
        count=count + 1
        accuracy.append(k) 
        f_score.append(j)
        a_auc.append(l)
    #print (accuracy,f_score,a_auc)
    print("Standard deviation of Glass at accuracy measure: "+str(statistics.stdev(accuracy)))
    print("Standard deviation of Glass at f_score measure: "+str(statistics.stdev(f_score)))
    print("Standard deviation of Glass at auc measure: "+str(statistics.stdev(a_auc)))
    #print (LongDataset)
    
    
def flagDataset(models,LongDataset):
    name = ['name','landmass','zone','area','population'	,'language','religion','bars','stripes','colours','red','green','blue','gold','white','black','orange','mainhue' ,'circles','crosses','saltires','quarters','sunstars','crescent','triangle','icon' ,'animate','text','topleft','botright']
    dataset = pandas.read_csv("Flag/flag.data.txt",names=name)
    dataset = dataset.replace('NaN',0)
    dataset = dataset.replace('green',0)
    dataset = dataset.replace('red',1)
    dataset = dataset.replace('gold',2)
    dataset = dataset.replace('white',3)
    dataset = dataset.replace('blue',4)
    dataset = dataset.replace('orange',5)
    dataset = dataset.replace('brown',6)
    dataset = dataset.replace('black',7)
    
    print(dataset.shape)
   # print(dataset)
    
    #dataset['Class'] = np.where(dataset['Class'] == 1, 0, 1)
    array = dataset.values
    data = array[:,1:28]
    label = dataset['botright']
    count =0 
    #print(dataset)
    f_score=[]
    accuracy=[]
    a_auc=[]
    
    for name,model in models:
        print ("-----------------------"+ name + "---------------------------------")
        meanfinder = []
        ffinder = []
        aucfinder = []        
        for x in range(1,11):
            kf = cross_validation.KFold(n = 194, n_folds=10,shuffle = True, random_state = None)
            acc = 0
            i = 0
            f1 = 0
            auc = 0
            for train_index, test_index in kf:
                slabel = np.where(label == randint(1,6), 0, 1)
                X_train, X_test = data[train_index], data[test_index]
                y_train, y_test = slabel[train_index], slabel[test_index]
                model.fit(X_train,y_train)
                prediction = model.predict(X_test)
                acc = acc + accuracy_score(y_test, prediction)
                #auc = auc + roc_auc_score(y_test, prediction)
                f1 = f1 + f1_score(y_test, prediction)                
                i = i+1
            meanfinder.append(acc/10)
            ffinder.append(f1/10)        
            aucfinder.append(auc/10)
            print("Iteration number " + str(x))
            print("Accuracy using "+ name + " " + str(acc / 10))
            print("F measure using "+ name + " " + str(f1 / 10))   
            #print("Area under curve using "+ name + " " + str(auc / 10))
            print ()
   
        j=sum(ffinder) / float(len(ffinder))  
        k=sum(meanfinder) / float(len(meanfinder))  
        #l=sum(aucfinder) / float(len(aucfinder))  
        
        print ("--------------------------------------------------")
        
        print("Final Accuracy using " +name + " " + str(k) )
        Flag[count]['Accuracy']= k
        print("Final F measure using " +name + " " + str(j) )                
        Flag[count]['F-Measure']= j
        #print("Final Area Under Curve using " +name + " " + str(l) )                
        #BreastCancer[count]['AUC'] = l        
        print ("--------------------------------------------------")
        count=count + 1
        accuracy.append(k) 
        f_score.append(j)
        #a_auc.append(l)
    #print (accuracy,f_score,a_auc)
    print("Standard deviation of Flag at accuracy measure: "+str(statistics.stdev(accuracy)))
    print("Standard deviation of Flag at f_score measure: "+str(statistics.stdev(f_score)))
    #print("Standard deviation of BreastCancer at auc measure: "+str(statistics.stdev(a_auc)))
    #print (LongDataset)
   
    
def CMCDataset(models,LongDataset):
    dataset = pandas.read_csv("CMC/cmc.data.txt",names=['Wife age','Wife Education','Husband Education','No of Children','Wife Religion','Wifes now working','Husbands occupation','Standard-of-living','Media exposure','Class'])
    print(dataset.shape)
    #print(dataset)
    
    dataset['Class'] = np.where(dataset['Class'] == 1, 0, 1)
    array = dataset.values
    data = array[:,0:8]
    label = dataset['Class']
    count =0 
    #print(dataset)
    
    
    f_score=[]
    accuracy=[]
    a_auc=[]
    
    for name,model in models:
        print ("-----------------------"+ name + "---------------------------------")
        meanfinder = []
        ffinder = []
        aucfinder = []        
        for x in range(1,11):
            kf = cross_validation.KFold(n = 1473, n_folds=10,shuffle = True, random_state = None)
            acc = 0
            i = 0
            f1 = 0
            auc = 0
            for train_index, test_index in kf:
                X_train, X_test = data[train_index], data[test_index]
                y_train, y_test = label[train_index], label[test_index]
                model.fit(X_train,y_train)
                prediction = model.predict(X_test)
                acc = acc + accuracy_score(y_test, prediction)
                auc = auc + roc_auc_score(y_test, prediction)
                f1 = f1 + f1_score(y_test, prediction)                
                i = i+1
            meanfinder.append(acc/10)
            ffinder.append(f1/10)        
            aucfinder.append(auc/10)
            print("Iteration number " + str(x))
            print("Accuracy using "+ name + " " + str(acc / 10))
            print("F measure using "+ name + " " + str(f1 / 10))   
            print("Area under curve using "+ name + " " + str(auc / 10))
            print ()
   
        j=sum(ffinder) / float(len(ffinder))  
        k=sum(meanfinder) / float(len(meanfinder))  
        l=sum(aucfinder) / float(len(aucfinder))  
        
        print ("--------------------------------------------------")
        
        print("Final Accuracy using " +name + " " + str(k) )
        CMC[count]['Accuracy']= k
        print("Final F measure using " +name + " " + str(j) )                
        CMC[count]['F-Measure']= j
        print("Final Area Under Curve using " +name + " " + str(l) )                
        CMC[count]['AUC'] = l        
        print ("--------------------------------------------------")
        count=count + 1
        accuracy.append(k) 
        f_score.append(j)
        a_auc.append(l)
    #print (accuracy,f_score,a_auc)
    print("Standard deviation of CMC at accuracy measure: "+str(statistics.stdev(accuracy)))
    print("Standard deviation of CMC at f_score measure: "+str(statistics.stdev(f_score)))
    print("Standard deviation of CMC at auc measure: "+str(statistics.stdev(a_auc)))
    #print (LongDataset)
    
def balanceScaleDataset(models,LongDataset):
    dataset = pandas.read_csv("balance-scale/balance-scale.data.txt",names=['Class','Left Weight','Left Distance','Right Weight','Right Distance'])
    print(dataset.shape)
    #print(dataset)
    
    dataset['Class'] = np.where(dataset['Class'] == 'B', 0, 1)
    array = dataset.values
    data = array[:,1:4]
    label = dataset['Class']
    count =0 
    #print(dataset)
    
    f_score=[]
    accuracy=[]
    a_auc=[]
    
    for name,model in models:
        print ("-----------------------"+ name + "---------------------------------")
        meanfinder = []
        ffinder = []
        aucfinder = []        
        for x in range(1,11):
            kf = cross_validation.KFold(n = 625, n_folds=10,shuffle = True, random_state = None)
            acc = 0
            i = 0
            f1 = 0
            auc = 0
            for train_index, test_index in kf:
                X_train, X_test = data[train_index], data[test_index]
                y_train, y_test = label[train_index], label[test_index]
                model.fit(X_train,y_train)
                prediction = model.predict(X_test)
                acc = acc + accuracy_score(y_test, prediction)
                #auc = auc + roc_auc_score(y_test, prediction)
                f1 = f1 + f1_score(y_test, prediction)                
                i = i+1
            meanfinder.append(acc/10)
            ffinder.append(f1/10)        
            aucfinder.append(auc/10)
            print("Iteration number " + str(x))
            print("Accuracy using "+ name + " " + str(acc / 10))
            print("F measure using "+ name + " " + str(f1 / 10))   
            #print("Area under curve using "+ name + " " + str(auc / 10))
            print ()
   
        j=sum(ffinder) / float(len(ffinder))  
        k=sum(meanfinder) / float(len(meanfinder))  
        l=sum(aucfinder) / float(len(aucfinder))  
        
        print ("--------------------------------------------------")
        
        print("Final Accuracy using " +name + " " + str(k) )
        Balancescale[count]['Accuracy']= k
        print("Final F measure using " +name + " " + str(j) )                
        Balancescale[count]['F-Measure']= j
        #print("Final Area Under Curve using " +name + " " + str(l) )                
        #BreastCancer[count]['AUC'] = l        
        print ("--------------------------------------------------")
        count=count + 1
        accuracy.append(k) 
        f_score.append(j)
        #a_auc.append(l)
    #print (accuracy,f_score,a_auc)
    print("Standard deviation of BalanceScale at accuracy measure: "+str(statistics.stdev(accuracy)))
    print("Standard deviation of BalanceScale at f_score measure: "+str(statistics.stdev(f_score)))
    #print("Standard deviation of BreastCancer at auc measure: "+str(statistics.stdev(a_auc)))
    #print (LongDataset)

'''   
def housingDataset(models,LongDataset):
   # dataset = pandas.read_csv("housing/housing.data.txt")
    #lambda x: pandas.Series([i for i in reversed(dataset.split(' '))])
    name = ['CRIM','ZN','INDUS','CHAS','NOX' ,'RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
    dataset = pandas.read_csv("housing/housing.data.txt",header=None,sep="|" )
    print(dataset.shape)
    print(dataset)
  
    dataset['Class'] = np.where(dataset['Class'] == 'B', 0, 1)
    array = dataset.values
    data = array[:,1:4]
    label = dataset['Class']
    count =0 
    print(dataset)
    for name,model in models:
        print ("-----------------------"+ name + "---------------------------------")
        meanfinder = []
        ffinder = []
        aucfinder = []        
        for x in range(1,11):
            kf = cross_validation.KFold(n = 625, n_folds=10,shuffle = True, random_state = None)
            acc = 0
            i = 0
            f1 = 0
            auc = 0
            for train_index, test_index in kf:
                X_train, X_test = data[train_index], data[test_index]
                y_train, y_test = label[train_index], label[test_index]
                model.fit(X_train,y_train)
                prediction = model.predict(X_test)
                acc = acc + accuracy_score(y_test, prediction)
                auc = auc + roc_auc_score(y_test, prediction)
                f1 = f1 + f1_score(y_test, prediction)                
                i = i+1
            meanfinder.append(acc/10)
            ffinder.append(f1/10)        
            aucfinder.append(auc/10)
            print("Iteration number " + str(x))
            print("Accuracy using "+ name + " " + str(acc / 10))
            print("F measure using "+ name + " " + str(f1 / 10))   
            print("Area under curve using "+ name + " " + str(auc / 10))
            print ()
   
        j=sum(ffinder) / float(len(ffinder))  
        k=sum(meanfinder) / float(len(meanfinder))  
        l=sum(aucfinder) / float(len(aucfinder))  
        
        print ("--------------------------------------------------")
        
        print("Final Accuracy using " +name + " " + str(k) )
        BreastCancer[count]['Accuracy']= k
        print("Final F measure using " +name + " " + str(j) )                
        BreastCancer[count]['F-Measure']= j
        print("Final Area Under Curve using " +name + " " + str(l) )                
        BreastCancer[count]['AUC'] = l        
        print ("--------------------------------------------------")
        count=count + 1
    print (LongDataset)    
'''
models = []
LongDataset = []
models.append(('Bagging', BaggingClassifier(DecisionTreeClassifier())))
models.append(('RF', RandomForestClassifier()))

models.append(('SVM_Linear', SVC(kernel='linear')))
models.append(('SVM_RBF', SVC(kernel='rbf')))
models.append(('KNN', KNeighborsClassifier(n_neighbors=3)))
models.append(('AdaBoost', AdaBoostClassifier(DecisionTreeClassifier())))
models.append(('DecisionTree', DecisionTreeClassifier()))
models.append(('NaiveBayes', GaussianNB()))


Bagging= {'Accuracy':0, 'F-Measure':0,'AUC':0}
RF= {'Accuracy':0, 'F-Measure':0,'AUC':0}
SVM_Linear= {'Accuracy':0, 'F-Measure':0,'AUC':0}
SVM_RBF= {'Accuracy':0, 'F-Measure':0,'AUC':0}
KNN= {'Accuracy':0, 'F-Measure':0,'AUC':0}
AdaBoost= {'Accuracy':0, 'F-Measure':0,'AUC':0}
DecisionTree= {'Accuracy':0, 'F-Measure':0,'AUC':0}
NaiveBayes= {'Accuracy':0, 'F-Measure':0,'AUC':0}

BreastCancer= [Bagging,RF,SVM_Linear,SVM_RBF,KNN,AdaBoost,DecisionTree,NaiveBayes]
Abalone= [Bagging,RF,SVM_Linear,SVM_RBF,KNN,AdaBoost,DecisionTree,NaiveBayes]
Aarrhythmia= [Bagging,RF,SVM_Linear,SVM_RBF,KNN,AdaBoost,DecisionTree,NaiveBayes]
Balancescale= [Bagging,RF,SVM_Linear,SVM_RBF,KNN,AdaBoost,DecisionTree,NaiveBayes]
CMC= [Bagging,RF,SVM_Linear,SVM_RBF,KNN,AdaBoost,DecisionTree,NaiveBayes]
Flag= [Bagging,RF,SVM_Linear,SVM_RBF,KNN,AdaBoost,DecisionTree,NaiveBayes]
Glass= [Bagging,RF,SVM_Linear,SVM_RBF,KNN,AdaBoost,DecisionTree,NaiveBayes]
Hepatitas= [Bagging,RF,SVM_Linear,SVM_RBF,KNN,AdaBoost,DecisionTree,NaiveBayes]

Nursery= [Bagging,RF,SVM_Linear,SVM_RBF,KNN,AdaBoost,DecisionTree,NaiveBayes]

LongDataset.append(BreastCancer) #
LongDataset.append(Abalone) #
LongDataset.append(Aarrhythmia) #
LongDataset.append(Balancescale) #
LongDataset.append(CMC) #
LongDataset.append(Flag)
LongDataset.append(Glass) #
LongDataset.append(Hepatitas) #

LongDataset.append(Nursery) #


#BreastCancerFunc(models,LongDataset)

#AbaloneFunc(models,LongDataset)

#Arrythemia(models,LongDataset)

balanceScaleDataset(models,LongDataset)

CMCDataset(models,LongDataset)

flagDataset(models,LongDataset)
glassDataset(models,LongDataset)

hepatitasFunc(models,LongDataset)


nurseryFunc(models,LongDataset)









#housingDataset(models,LongDataset)