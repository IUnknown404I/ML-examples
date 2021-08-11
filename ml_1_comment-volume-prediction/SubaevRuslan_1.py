import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler, scale
from sklearn.model_selection import train_test_split

import copy
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from pandas import ExcelWriter

# !!!  I used strict pointers there, so u have to change it before running
# !!!               There are 2 cases: 15 and 91 lines

syspathTraining = str(os.getcwd()).replace('SubaevRuslan_1.ipynb','') + "/facebook-comment-volume-prediction-master/dataset/Training/"
fileTraining = ['Features_Variant_1.csv', 
                'Features_Variant_2.csv', 
                'Features_Variant_3.csv', 
                'Features_Variant_4.csv', 
                'Features_Variant_5.csv']
scaner = StandardScaler()

fullTraining = pd.concat(pd.read_csv(syspathTraining + file, index_col=False, header=None) for file in fileTraining)
y = np.array(fullTraining[53]).reshape(-1,1)

x = fullTraining.iloc[:,:-1]
x = scale(x, axis=0)

print(x.shape)
print(y.shape,end='\n\n')

def MSE(xb,y,theta):
    return np.sum(np.square(xb.dot(theta)-y))/len(y)

# Adding ones to the X matrix
xb = np.c_[np.ones((len(x),1)),x]
# xbStandartized = scaner.fit_transform(copy.deepcopy(xb))
# print(np.mean(xbStandartized[:,:]), np.std(xbStandartized[:,:]))

# Num of any samples
m = len(y) 
cost = []
learningRate = 0.0018
numIterations = 1200

# Random initialization with standard normal distribution -- randn(.,.)
theta = np.random.randn(54,1) 

# Start gradient descent Uuuf
for i in range(numIterations):
    gradient = 2/m * xb.T.dot(xb.dot(theta) - y) # dimension: (54,1)
    theta = theta - learningRate * gradient
    cost.append(MSE(xb,y,theta))
print(theta.shape, end='\n\n')

import matplotlib.pyplot as plt

# Out the MSE jumping over learning rate and iterations
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 6))

ax[0].plot(range(0,250), cost[0:250])
ax[1].plot(range(500,1199), cost[500:1199])
plt.subplots_adjust(wspace=0.5)

ax[0].set_xlabel('Num of iterations', fontsize = 14)
ax[1].set_xlabel('Num of iterations', fontsize = 14)
ax[0].set_ylabel('MSE', fontsize = 14)
ax[1].set_ylabel('MSE', fontsize = 14)

X_train_list = []
y_train_list = []


"""    Now fill in every list of 5 vars
      plus one more == summury of all-rs   """

for i in range(5):
    delta = pd.read_csv(syspathTraining + fileTraining[i], index_col=False, header=None).iloc[:,:-1]
    delta = scale(delta,axis=0)
    X_train_list.append(delta)

    delta = pd.read_csv(syspathTraining + fileTraining[i], index_col=False, header=None).iloc[:,-1:]
    y_train_list.append(delta)

X_train_list.append(x)
y_train_list.append(fullTraining.iloc[:,-1:])


"""     And now for the test-ones     """

sysPathTesting = str(os.getcwd()).replace('SubaevRuslan_1.ipynb','') + "/facebook-comment-volume-prediction-master/dataset/Testing/TestSet/"
fileTesting = ['Test_Case_1.csv', 'Test_Case_2.csv', 'Test_Case_3.csv',
              'Test_Case_4.csv', 'Test_Case_5.csv',  
              'Test_Case_6.csv', 'Test_Case_7.csv', 'Test_Case_8.csv', 
              'Test_Case_9.csv', 'Test_Case_10.csv']
X_test_list = []
y_test_list = []
fullTesting = pd.concat(pd.read_csv(sysPathTesting + path, index_col=False, header=None) for path in fileTesting)

for i in range(10):
    delta = pd.read_csv(sysPathTesting + fileTesting[i], index_col=False, header=None).iloc[:,:-1]
    delta = scale(delta, axis=0)
    X_test_list.append(delta)

    delta = pd.read_csv(sysPathTesting + fileTesting[i], index_col=False, header=None).iloc[:,-1:]
    y_test_list.append(delta)

xT = fullTesting.iloc[:,:-1]
xT = scale(xT, axis=0)
X_test_list.append(xT)
y_test_list.append(fullTesting.iloc[:,-1:])


'''
                    ##           Def for Training-ones            ##
'''
final=[]
final_any=[]
rmse_any=[]
r2_any=[]
s=0
t=1

for i in range(6):
    for every in X_train_list[i]:
        for each in every:
            s+=each*theta[t]
            t+=1

        if(s<0): final.append(0)
        else: final.append(float(s).__round__())
        s=0  
        t=1

    final = np.array(final)
    rmse = sqrt(mean_squared_error(y_train_list[i], final))
    rmse_any.append(rmse)
    final_any.append(final)
    final=[]

    if i == 5: 
        print("Variant overall: RMSE", int(rmse*1000)/1000)
    else:
        r2 = r2_score(np.array(y_train_list[i]), np.array(final_any[i]))
        r2_any.append(r2)
        print("Variant {0}: ".format(i + 1),' RMSE: ', int(rmse*1000)/1000,'\n\t','    R**2: ',int(r2*10000)/10000)

r2 = r2_score(np.array(y_train_list[5]), np.array(final_any[5]))
r2_any.append(r2)
print('\t\t R**2: ',int(r2*10000)/10000, end='\n\n')

check = pd.DataFrame({'Actual': np.array(y_train_list[0]).tolist(), 'Predicted': np.array(final_any[0]).tolist()})


'''
                    ##           Def for Test-ones            ##
'''
final_Test=[]
final_any_Test=[]
rmse_any_Test=[]
r2_any_Test=[]
s=0
t=1

for i in range(11):
    for every in X_test_list[i]:
        for each in every:
            s+=each*theta[t]
            t+=1
        
        if(s<0): final_Test.append(0)
        else: final_Test.append(float(s).__round__())
        s=0  
        t=1
    
    final_Test = np.array(final_Test)
    rmse = sqrt(mean_squared_error(y_test_list[i], final_Test))
    rmse_any_Test.append(rmse)
    final_any_Test.append(final_Test)
    final_Test=[]
    
    if i == 10:
        print("Variant overall: RMSE", int(rmse*1000)/1000)
    else:
        r2 = r2_score(np.array(y_test_list[i]), np.array(final_any_Test[i]))
        r2_any_Test.append(r2)
        if i!=9:
            print("Variant {0}: ".format(i + 1),' RMSE: ', int(rmse*1000)/1000,'\n\t','    R**2: ',int(r2*10000)/10000)
        else:
            print("Variant {0}: ".format(i + 1),'RMSE: ', int(rmse*1000)/1000,'\n\t','    R**2: ',int(r2*10000)/10000)

r2 = r2_score(np.array(y_test_list[10]), np.array(final_any_Test[10]))
r2_any_Test.append(r2)
print('\t\t R**2: ',int(r2*10000)/10000, end='\n\n')

check = pd.DataFrame({'Actual': np.array(y_test_list[5]).tolist(), 'Predicted': np.array(final_any_Test[5]).tolist()})
print(check)


'''
                    ##           Output            ##
'''
indexes = []
for i in range(54):
    indexes.append('f-'+str(i))
main = []
main.append(r2_any[:5]), main.append(rmse_any[:5]), main.append(rmse_any_Test[:5])
main = np.array(main)

e = []
delta = 0

# E calc
for each in main:
    for every in each:
        delta+=every
    e.append(delta/5)
    delta = 0

# STD calc
std = []
for every in main:
    std.append(np.var(every))

dfOutMain = pd.DataFrame(main.reshape(3,5), index=['R2', 'RMSE-train', 'RMSE-test'], columns=['T1','T2','T3','T4','T5'])
dfOutPar = pd.DataFrame({
    'E': e,
    'STD': std,
}, index=['R2', 'RMSE-train', 'RMSE-test'])
dfOutCost = pd.DataFrame(theta, index=indexes, columns=['Features'])

out1 = pd.concat([dfOutMain,dfOutPar],sort=False,axis=1)
out = pd.concat([out1,dfOutCost],sort=False,axis=1)

exWriter = ExcelWriter('SubaevRuslan.xlsx')  # pylint: disable=abstract-class-instantiated
out.to_excel(exWriter, sheet_name='Subaev_1st')
exWriter.save()

# # Coefficient of determination
# d1 = np.array(y_train_list[5]) - np.array(pd.DataFrame(final))
# d2 = np.array(y_train_list[5]) - np.array(pd.DataFrame(final)).mean()
# r2 = 1 - d1.dot(d1)/d2.dot(d2)

# print("Coefficient of determination: ", r2)

# def rSquare(estimations, measureds):
#     """ 
#             Compute the coefficient of determination of random data. 
#     This metric gives the level of confidence about the model used to model data
#     """

#     SEE =  (( np.array(measureds) - np.array(estimations) )**2 ).sum()
#     mMean = (np.array(measureds)).sum() / float(len(measureds))
#     dErr = ((mMean - measureds)).sum()

#     return 1 - (SEE / dErr)

# rSquare(np.array(y_train_list[5]),np.array(pd.DataFrame(final))