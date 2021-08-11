import pandas as pd 

# data = pd.read_csv('D:\Python_all\ml_test\\nba.csv',sep='\t')
data = pd.DataFrame({
    'country': ['Kazakhstan', 'Russia', 'Belarus', 'Ukraine'],
    'population': [17.04, 143.5, 9.5, 45.5],
    'square': [2724902, 17125191, 207600, 603628]
})
print(data)