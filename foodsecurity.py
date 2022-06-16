import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF 
from sklearn import svm
train=pd.read_csv('D:/Programming projects/PythonProject/code/MachineLearning/competition/foodsecurity/train.csv',sep='\t')
test=pd.read_csv('D:/Programming projects/PythonProject/code/MachineLearning/competition/foodsecurity/test_new.csv')

import re
def extractChinese(s):
    pattern="[\u4e00-\u9fa5]+"#中文正则表达式
    regex = re.compile(pattern) #生成正则对象 
    results = regex.findall(s) #匹配
    return "". join(results)
# 预处理数据
label = train['label']
train_data = []
for i in range(len(train['comment'])):
    train_data.append(' '.join(extractChinese(train['comment'][i])))
test_data = []
for i in range(len(test['comment'])):
    test_data.append(' '.join(extractChinese(test['comment'][i])))


tfidf = TFIDF(min_df=1, # 最小支持长度
           max_features=150000, #取特征数量
           strip_accents='unicode',
           analyzer='word',
           token_pattern=r'\w{1,}',
           ngram_range=(1, 3),  
           use_idf=1,
           smooth_idf=1,
           sublinear_tf=1,
           stop_words=None,
  
           ) 

# 合并训练和测试集以便进行TFIDF向量化操作
data_all = train_data + test_data
len_train = len(train_data)

tfidf.fit(data_all)

data_all = tfidf.transform(data_all)

# 恢复成训练集和测试集部分
train_x = data_all[:len_train]
test_x = data_all[len_train:]
print ('TF-IDF处理结束.')

clf=svm.LinearSVC(loss='squared_hinge', dual=True, tol=0.0001,
                  C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, 
                  class_weight='balanced', verbose=0, random_state=None, max_iter=1000)
svm=clf.fit(train_x,label)
svm_pre=svm.predict(test_x)
svm = pd.DataFrame(data=svm_pre, columns=['comment'])
svm['id'] = test.id
svm = svm[['id', 'comment']]
svm.to_csv('svm.csv',index=False)


r=test
r['id']=svm_pre
r0=r[r.id==0]

key_word=['蚊子','老鼠','苍蝇','酸臭']

# key_word2=['蚊子','剩','不新鲜','没熟','老鼠','烂','骚味','苍蝇','虫','臭','想吐','太硬']
for i in key_word:
  print (r0[r0['comment'].str.contains(i)])


key_word2=['剩','不新鲜','没熟','烂']
for i in key_word2:
  print (r0[r0['comment'].str.contains(i)])


key_word3=['骚味','苍蝇','虫','臭','想吐','太硬']
for i in key_word3:
  print (r0[r0['comment'].str.contains(i)])

a=[17,1753,229,1767,238,1953 ,581, 697,722,1369, 
  1985,1342,753,1963,74,1296,531,1296,613,662 ,322,1413]

for i in a:
  svm.loc[i,('comment')]=[1]
svm.to_csv('new_svm.csv', index=False)
print ('结束3.')
