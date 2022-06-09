# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 21:28:21 2022

@author: brian
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LEARNING_TIME = 1
LEARNING_STEP = 5

LAYER_WIDTH = [2,6,1]



def sigmoid(z):
    return 1/(1+np.exp(-z))
sigmoid_v = np.vectorize(sigmoid)


class model(list):        
    def __init__(self,init,result,layer_width,learning_step,lamda=50):
        self.layer_width = layer_width
        self.alpha = learning_step
        self.init = init
        self.result = result
        self.append(layer(init,self.layer_width[0]))
        self.lamda = lamda
        
        while len(self)<len(self.layer_width):
            L=self[-1]
            self.append(layer(L.forward(),self.layer_width[L.l+1],l=L.l+1))
        L=self[-1]

        self.append(layer(L.forward(),nn=0,l=L.l+1))
        self.cost_iteration = []
        self.accuracy_iteration = []
        self.acc=[]
        self.f1=[]
        self.epsilon_iteration=[]
    
    def forward(self,start_l=0,end_l=None):
        if (end_l==None or end_l>=len(self)):end_l=len(self)-1
        "Use the data in layer start_l to update until layer end_l included."
        for l in range(start_l,end_l):
            self[l+1].data = self[l].forward()
            self[l+1].biased = np.concatenate((np.ones((self[l+1].data.shape[0],1)),self[l+1].data),axis=1)
        
    def backward(self,start_l=0,end_l=None,d_next=None):
        "Update the start_l to end_l (not included) theta"
        if end_l!=None and end_l>=len(self):
            end_l=None
        if end_l != None and d_next == None:
            print("Error, d_next is not provided.")
            return None
        
        if end_l==None:end_l=len(self)-1
        for l in range(end_l,start_l-1,-1):
            if l == end_l:
                d_next=self.result
            if l!=end_l:
                self[l].derivative(d_next)
            d_next = self[l].delta(d_next)
    
    def update(self,start_l=0,end_l=None):
        "Update the layer from start_l to end_l (not included)"
        if (end_l==None or end_l>=len(self)):end_l=len(self)-1
        for l in range(start_l,end_l-1):
            self[l].theta = self[l].theta-self.alpha*self[l].D-self.alpha*self.lamda*self[l].theta/self[l].data.shape[0]
            
    def cost(self):
        return ((np.matmul(np.log(self[-1].data).T,self.result)+np.matmul(np.log(1-self[-1].data).T,(1-self.result)))/-self[-1].data.shape[0])[0][0]
    
    def epsilon(self):
        centroid_A,centroid_B = np.random.choice(self[-1].data.ravel(),size=2,replace=False)
        while True:
            temp_A=[]
            temp_B=[]
            
            for m in self[-1].data:
                if abs(m[0]-centroid_A) <= abs(m[0]-centroid_B):
                    temp_A.append(m[0])
                else:
                    temp_B.append(m[0])
            if temp_A==[]:
                print("Empty A: "+str(centroid_A)+" "+str(centroid_B))
                return -1,-1
            if temp_B==[]:
                print("Empty B: "+str(centroid_A)+" "+str(centroid_B))
                return -1,-1
            c_A = np.mean(temp_A)
            c_B = np.mean(temp_B)
            if c_A==centroid_A and c_B==centroid_B:    
                return centroid_A,centroid_B
            else:
                centroid_A = c_A
                centroid_B = c_B
                
                
    def training(self,start_l=0,end_l=None,iteration=100):
        for i in range(iteration):
            self.forward(start_l,end_l)
            self.backward(start_l,end_l)
            self.update(start_l,end_l)
            self.cost_iteration.append(self.cost())
            tp,tn,fp,fn=0,0,0,0
            centroid_A,centroid_B = self.epsilon()
            epsilon = 0.5*centroid_A+0.5*centroid_B
            self.epsilon_iteration.append(epsilon)
            for m in range(self[-1].data.shape[0]):
                if self[-1].data[m]>=epsilon and self.result[m]==1:tp+=1
                if self[-1].data[m]<epsilon and self.result[m]==1:fn+=1
                if self[-1].data[m]>=epsilon and self.result[m]==0:fp+=1
                if self[-1].data[m]<epsilon and self.result[m]==0:tn+=1
            self.acc.append([tp,tn,fp,fn])
            if [tp,tn,fp,fn] == [0,0,0,0]:
                print(i)
                print(centroid_A,centroid_B)
                break
            self.f1.append(2*tp/(tp+fn+tp+fp))
        self.acc = np.array(self.acc)
        
    
    def predict(self,variable):
        self.pre_list = []
        init = layer(variable,nn=self.layer_width[0])
        init.theta = self[0].theta
        init.biased = np.concatenate((np.ones((init.data.shape[0],1)),init.data),axis=1)
        self.pre_list.append(init)
        prediction=[]
        
        l=1
        for w in self.layer_width:
            self.pre_list.append(layer(self.pre_list[-1].forward(),w,l))
            self.pre_list[l].theta=self[l].theta
            self.pre_list[l].biased=np.concatenate((np.ones((self.pre_list[l].data.shape[0],1)),self.pre_list[l].data),axis=1)
            l+=1
        for p in self.pre_list[-1].data:
            if p>=self.epsilon_iteration[-1]:
                prediction.append(1)
            else:
                prediction.append(0)
        return prediction
    
class layer:    
    def __init__(self,curr,nn=0,l=0):
        self.data = curr
        self.nn = nn # Number of features in next level
        self.l = l
        self.D = None
        if self.nn==0:
            self.theta = None
            self.biased = None
        else:
            self.theta = np.random.random((self.nn,curr.shape[1]+1))
            self.biased = np.concatenate((np.ones((self.data.shape[0],1)),self.data),axis=1)
            self.D = np.zeros_like(self.theta)
        
    def forward(self):
        if self.nn==0:
            self.next = None
        else:
            self.next = sigmoid_v(np.matmul(self.biased,self.theta.transpose()))
        return self.next
    
    def delta(self,d_next=None):
        if self.nn==0:
            self.d = self.data-d_next
            return self.d
        else:
            self.d = np.multiply(np.multiply(np.matmul(d_next,self.theta),self.biased),1-self.biased)[:,1:]
            return self.d
    def derivative(self,d_next=None):
        if self.nn==0:
            print("It is the output layer.\nNo derivative of the cost function.")
            self.D = None
        else:
            self.D = self.D + np.matmul(d_next.transpose(),self.biased)/self.data.shape[0]
        return self.D
    
    
        
    
# =============================================================================
# Data cleaning
# =============================================================================

training_excel = pd.read_csv("train.csv")

training_x = training_excel.drop(columns=["PassengerId","Name","Survived","Ticket","Cabin","Fare"])
training_x.rename(columns={"Pclass":"isFirstClass","Embarked":"isSouthampton"},inplace=True)
training_x.insert(1,"isSecondClass",0)
training_x.insert(2,"isThirdClass",0)
training_x.insert(training_x.shape[1],"isCherbourg",0)
training_x.insert(training_x.shape[1],"isQueenstown",0)

training_x.loc[training_x["isFirstClass"] == 2,"isSecondClass"] = 1
training_x.loc[training_x["isFirstClass"] == 3,"isThirdClass"] = 1
training_x.loc[training_x["isFirstClass"] != 1,"isFirstClass"] = 0
training_x.loc[training_x["Sex"] == "male","Sex"] = 0
training_x.loc[training_x["Sex"] == "female","Sex"] = 1
training_x.loc[training_x["isSouthampton"] == "S","isSouthampton"] = 1
training_x.loc[training_x["isSouthampton"] == "C","isCherbourg"] = 1
training_x.loc[training_x["isSouthampton"] == "Q","isQueenstown"] = 1
training_x.loc[training_x["isSouthampton"] != 1,"isSouthampton"] = 0

training_x = training_x.astype({"isFirstClass":np.int64,"isSouthampton":np.int64})

training_y = training_excel["Survived"]

# =============================================================================
# Data cleaning
# =============================================================================

# =============================================================================
# Main
# =============================================================================

# training_x["Age"] = (training_x["Age"]-training_x["Age"].min())/(training_x["Age"].max()-training_x["Age"].min())# Age 0-1<->max-min-normalization

training_x["Age"] = (training_x["Age"] - training_x["Age"].mean())/training_x["Age"].std(ddof=0)
training_x["Age"] = training_x["Age"].fillna(0)
#Age standardlize

X = training_x.to_numpy()
Y = training_y.to_numpy()
Y = Y.reshape(len(Y),1)

TRAIN_IT=891

M = model(X[:TRAIN_IT],Y[:TRAIN_IT],LAYER_WIDTH,LEARNING_STEP)
M.training(iteration=400)

test_excel = pd.read_csv("test.csv")

test_x = test_excel.drop(columns=["PassengerId","Name","Ticket","Cabin","Fare"])
test_x.rename(columns={"Pclass":"isFirstClass","Embarked":"isSouthampton"},inplace=True)
test_x.insert(1,"isSecondClass",0)
test_x.insert(2,"isThirdClass",0)
test_x.insert(test_x.shape[1],"isCherbourg",0)
test_x.insert(test_x.shape[1],"isQueenstown",0)

test_x.loc[test_x["isFirstClass"] == 2,"isSecondClass"] = 1
test_x.loc[test_x["isFirstClass"] == 3,"isThirdClass"] = 1
test_x.loc[test_x["isFirstClass"] != 1,"isFirstClass"] = 0
test_x.loc[test_x["Sex"] == "male","Sex"] = 0
test_x.loc[test_x["Sex"] == "female","Sex"] = 1
test_x.loc[test_x["isSouthampton"] == "S","isSouthampton"] = 1
test_x.loc[test_x["isSouthampton"] == "C","isCherbourg"] = 1
test_x.loc[test_x["isSouthampton"] == "Q","isQueenstown"] = 1
test_x.loc[test_x["isSouthampton"] != 1,"isSouthampton"] = 0

test_x = test_x.astype({"isFirstClass":np.int64,"isSouthampton":np.int64})
test_x["Age"] = (test_x["Age"] - test_x["Age"].mean())/test_x["Age"].std(ddof=0)
test_x["Age"] = test_x["Age"].fillna(0)
test_X = test_x.to_numpy()

prediction = M.predict(test_X)
plt.hist(prediction)
# tp,tn,fp,fn = 0,0,0,0
# for i in range(891-TRAIN_IT):
#     if prediction[i]==1 and Y[i+TRAIN_IT][0]==1:tp+=1
#     if prediction[i]==0 and Y[i+TRAIN_IT][0]==0:tn+=1
#     if prediction[i]==1 and Y[i+TRAIN_IT][0]==0:fp+=1
#     if prediction[i]==0 and Y[i+TRAIN_IT][0]==1:fn+=1

# plt.scatter(M[-1].data,Y)
# plt.show()
# plt.plot(M.cost_iteration)