
# %%time
import pandas as pd 

df1=pd.DataFrame()
df2=pd.DataFrame()
df3=pd.DataFrame()
df4=pd.DataFrame()
df5=pd.DataFrame()

fileName = ['Features_Variant_1.csv', 
            'Features_Variant_2.csv', 
            'Features_Variant_3.csv', 
            'Features_Variant_4.csv', 
            'Features_Variant_5.csv']
dataFrames=[ df1, df2, df3, df4, df5 ]

df = pd.concat(pd.read_csv("dataset//training//Features_Variant_" + str(num) + ".csv", index_col=False, header=None) for num in range(1,6,1))
print(df)



# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# from sklearn.tree import DecisionTreeClassifier, plot_tree

# # Load data
# iris = load_iris()

# plt.figure(figsize=((20,13)))
# clf = DecisionTreeClassifier()
# clf = clf.fit(iris.data, iris.target)
# plot_tree(clf, 
#           filled=True, 
#           feature_names=iris.feature_names, 
#           class_names=iris.target_names, 
#           rounded=True)
# plt.show()