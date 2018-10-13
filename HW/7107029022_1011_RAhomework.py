
# coding: utf-8

# In[1]:


#匯入模組
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import operator
from scipy import stats


# In[2]:


# 讀取數據
# Index : 0 - Extreme Weak 1 - Weak 2 - Normal 3 - Overweight 4 - Obesity 5 - Extreme Obesity
data = pd.read_csv('data/500_Person_Gender_Height_Weight_Index.csv')


# # ID3

# In[3]:


from pprint import pprint
dataset = pd.read_csv('data/500_Person_Gender_Height_Weight_Index.csv')


# In[4]:


# 計算entropy
def entropy(target_col):
    elements,counts = np.unique(target_col,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy


# In[5]:


# 計算InfoGain
def InfoGain(data,split_attribute_name,target_name="Index"):
    total_entropy = entropy(data[target_name])
    vals,counts= np.unique(data[split_attribute_name],return_counts=True)
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain


# In[6]:


# ID3演算法
def ID3(data,originaldata,features,target_attribute_name="Index",parent_node_class = None):

    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]    
    elif len(data)==0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name],return_counts=True)[1])]    
    elif len(features) ==0:
        return parent_node_class
    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return_counts=True)[1])]
        item_values = [InfoGain(data,feature,target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        tree = {best_feature:{}}
        features = [i for i in features if i != best_feature]
        
        for value in np.unique(data[best_feature]):
            value = value
            sub_data = data.where(data[best_feature] == value).dropna() 
            subtree = ID3(sub_data,dataset,features,target_attribute_name,parent_node_class)
            tree[best_feature][value] = subtree
            
        return(tree)


# In[7]:


# 預測
def predict(query,tree,default = 1): 
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]] 
            except:
                return default
            result = tree[key][query[key]]
            if isinstance(result,dict):
                return predict(query,result)
            else:
                return result


# In[8]:


# 資料集測試和訓練
def train_test_split(dataset):
    training_data = dataset.iloc[:400].reset_index(drop=True)
    testing_data = dataset.iloc[400:].reset_index(drop=True)
    return training_data,testing_data
training_data = train_test_split(dataset)[0]
testing_data = train_test_split(dataset)[1] 

def test(data,tree):
    queries = data.iloc[:,:-1].to_dict(orient = "records")
    predicted = pd.DataFrame(columns=["predicted"]) 
    
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(queries[i],tree,1.0) 
    print('The prediction accuracy is: ',(np.sum(predicted["predicted"] == data["Index"])/len(data))*100,'%')


# In[9]:


# ID3結果
print("ID3結果")
tree = ID3(training_data,training_data,training_data.columns[:-1])
pprint(tree)
print('\n')
test(testing_data,tree)


# # CART

# In[10]:


# CART
from random import seed
from random import randrange
from csv import reader
 
# Load a CSV file
def load_csv(filename):
    file = open(filename, "r")
    lines = reader(file)
    dataset = list(lines)
    return dataset
 
# string轉成float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# In[11]:


# 資料集分成 k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split
 
# 計算平均準確率
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# In[12]:


# 使用cross validation split評估
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores
 
# 根據屬性分類數據集
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# In[13]:


# 計算Gini index
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini
 
# 選擇最佳分類點
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset)) # class_values的值為: [0, 1]
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1): # index的值為: [0, 1, 2, 3]
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index':b_index, 'value':b_value, 'groups':b_groups} # 返回字典數據類型

def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# In[14]:


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)

# 決策樹
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root


# In[15]:


# 對決策樹預測
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

# 分類
def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return(predictions)


# In[16]:


# CART結果
seed(1)
filename = 'data/500_Person_Gender_Height_Weight_Index.csv'
dataset = load_csv(filename)

n_folds = 5
max_depth = 5
min_size = 10
scores = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size)
print("CART結果")
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))


# # McNemar

# In[17]:


m0 = ((data["Gender"] == "Male") & (data["Index"] == 0)).sum()
m1 = ((data["Gender"] == "Male") & (data["Index"] == 1)).sum()
m2 = ((data["Gender"] == "Male") & (data["Index"] == 2)).sum()
m3 = ((data["Gender"] == "Male") & (data["Index"] == 3)).sum()
m4 = ((data["Gender"] == "Male") & (data["Index"] == 4)).sum()
m5 = ((data["Gender"] == "Male") & (data["Index"] == 5)).sum()
print(m0,m1,m2,m3,m4,m5)


# In[18]:


f0 = ((data["Gender"] == "Female") & (data["Index"] == 0)).sum()
f1 = ((data["Gender"] == "Female") & (data["Index"] == 1)).sum()
f2 = ((data["Gender"] == "Female") & (data["Index"] == 2)).sum()
f3 = ((data["Gender"] == "Female") & (data["Index"] == 3)).sum()
f4 = ((data["Gender"] == "Female") & (data["Index"] == 4)).sum()
f5 = ((data["Gender"] == "Female") & (data["Index"] == 5)).sum()
print(f0,f1,f2,f3,f4,f5)


# In[19]:


gender = np.array((["男"]*6)+(["男"]*15)+(["男"]*28)+(["男"]*32)+(["男"]*59)+(["男"]*105)+(["女"]*7)+(["女"]*7)+(["女"]*41)+(["女"]*36)+(["女"]*71)+(["女"]*93))
index = np.array((["0"]*6)+(["1"]*15)+(["2"]*28)+(["3"]*32)+(["4"]*59)+(["5"]*105)+(["0"]*7)+(["1"]*7)+(["2"]*41)+(["3"]*36)+(["4"]*71)+(["5"]*93))


# In[20]:


table = pd.DataFrame({"Gender":gender,"Index":index})
table_tab = pd.crosstab(table.Gender,table.Index,margins=True)
table_tab.columns = ["ExtremeWeak","Weak","Normal","Overweight","Obesity","ExtremeObesity","Total"]
table_tab.index = ["Female","Male","Total"]
observed = table_tab.iloc[0:3,0:7]
print(observed)


# In[21]:


expected = np.outer(table_tab["Total"][0:2],table_tab.loc["Total"][0:6]) / 500
expected = pd.DataFrame(expected)
expected.columns = ["ExtremeWeak","Weak","Normal","Overweight","Obesity","ExtremeObesity"]
expected.index = ["Female","Male"]
print(expected)


# In[22]:


rows = 2
columns = 6
df = (rows-1)*(columns-1)
print("自由度:",df,'\n')

chi_square_stat = (((observed-expected)**2)/expected).sum()
chi_square_stat2 = (((observed-expected)**2)/expected).sum().sum()
print("卡方檢定統計量:",'\n',chi_square_stat,'\n')
print("卡方檢定統計量:",chi_square_stat2)


# In[23]:


chi_squared, p_value,degree_of_freedom, matrix=stats.chi2_contingency(observed=observed)
print(chi_squared, p_value)

obs = stats.chi2.ppf(q=0.95, df=df)
print("臨界值:",obs)

from statsmodels.stats.contingency_tables import mcnemar
table = np.array([[2, 4],[0, 4]])
result = mcnemar(table, exact=True)

print("statistic = ",result.statistic)
print("p-value = ",result.pvalue)

