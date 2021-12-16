#!/usr/bin/env python
# coding: utf-8

# # Importing Necessary Python Libraries

# In[1]:


# Dataframe building and Analysis library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle
import random

# Word Stemming Library to make root words in a String
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

# Words to Vectors Library
from sklearn.feature_extraction.text import CountVectorizer

# Similarity and Distance Metrics Library
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from scipy.stats import kendalltau
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.spatial import distance
from Levenshtein import distance as levenshtein_distance


# # Data Extraction and Exploratory Analysis

# In[13]:


# Dataset in two files (test and train) directly downloaded from Blob Storage in Azure Cloud
test = pd.read_csv(r'https://researchproject.blob.core.windows.net/project/drugsComTest_raw.csv', na_values=['(NA)']).fillna(0)
train = pd.read_csv(r'https://researchproject.blob.core.windows.net/project/drugsComTrain_raw.csv', na_values=['(NA)']).fillna(0)


# In[14]:


df = pd.concat([test, train], ignore_index=True, sort=False)
df.head()


# In[15]:


#Dropping Unneccesary Columns
df = df.drop(columns=['date','Unnamed: 7', 'Unnamed: 8','Unnamed: 9','Unnamed: 10','Unnamed: 11','Unnamed: 12'])
df.info()


# In[16]:


# Converting 'Rating' and 'UsefulCount' into float type for generating new column of 'most_reviewed' drugs
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['usefulCount'] = pd.to_numeric(df['usefulCount'], errors='coerce')
df['most_reviewed'] = df['rating'] * df['usefulCount']


# In[17]:


df = df[['uniqueID','drugName','condition','review','most_reviewed']]
df.head()


# # Plotting the graph for Top 10 Most Reviewed Drugs

# In[18]:


#Sorting Top 10 Drugs in Dataset
df = df.sort_values(by='most_reviewed', ascending=False, ignore_index=True)
df = df.drop_duplicates(subset = ['condition'], ignore_index=True)
df = df.drop_duplicates(subset = ['drugName'], ignore_index=True)
df_graph = df[['drugName','most_reviewed']]
df_graph = df_graph.head(10)
df_graph


# In[19]:


#MatplotLb Barh graph
df_graph.plot(kind='barh', x='drugName', y='most_reviewed')
plt.show()


# In[20]:


#Removing unwanted characters and splitting in words the string value of 'Review' column
df['review'] = df['review'].map(lambda x: re.sub(r'["&#039;"]','', x))
df['review'] = df['review'].apply(lambda x: x.split())
df['review'].values


# In[21]:


# Now Concatnating Drugs and Condition values with Reviews to build metadata of drugs
df['condition_list'] = df['condition'].apply(lambda x: x.split())
df['condition_list'] = df['condition'].apply(lambda x: "".join(x))
df['drugName_list'] = df['drugName'].apply(lambda x: x.split())
df['drugName_list'] = df['drugName'].apply(lambda x: "".join(x))


# In[22]:


df['condition_list'] = df['condition'].apply(lambda x: x.split())
df['drugName_list'] = df['drugName'].apply(lambda x: x.split())


# In[23]:


df['tags'] = df['drugName_list'] + df['condition_list'] + df['review']


# In[24]:


df.head()


# In[125]:


#Creating New Dataframe with relevant columns for further analysis
new_df = df[['uniqueID','drugName','condition','tags']]
new_df


# In[126]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())


# In[127]:


new_df['tags'][0]


# # Stemming using NLTK Library and comparing results 

# In[128]:


# PorterStemmer is applied to check root words
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[129]:


new_df_ps = new_df[['uniqueID','drugName','condition','tags']]
new_df_ps['tags'] = new_df_ps['tags'].apply(stem)
new_df_ps['tags'][0]


# In[130]:


# WordNetLemmatizer is also deployed to check root words and meaning consistency
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

def lemma(text):
    y = []
    for i in text.split():
        y.append(wordnet_lemmatizer.lemmatize(i, pos="v"))
        
    return " ".join(y)


# In[131]:


new_df_lemma = new_df[['uniqueID','drugName','condition','tags']]
new_df_lemma['tags'] = new_df_lemma['tags'].apply(lemma)
new_df_lemma['tags'][0]


# #  Implementing Count Vectorizer to make words in vector form

# In[132]:


# Max Features is set to 500 a d stop words are removed from string
from sklearn.feature_extraction.text import CountVectorizer
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    return text
cv = CountVectorizer(max_features=500, stop_words='english', analyzer='word', preprocessor=preprocess_text)


# In[133]:


# Applying Fit Transform to Lemma processed vector as it give more relevant meaning
vectors = cv.fit_transform(new_df_lemma['tags']).toarray()
vectors[0]


# In[134]:


# Getting Top 500 most used words from bag of words of 'Tags' column values
cv.get_feature_names_out()


# # Evaluating Model with Various Similarity and Distances Metrices

# In[110]:


# Pearson Correlation with first drug and random drugs
l=[]
for i in random.sample(range(1, 600), 10):
    corr, _ = pearsonr(vectors[0],vectors[i])
    l.append(corr)
    print('Pearsons correlation: %.3f' % corr)
avg=np.mean(l)
print('Pearsons correlation average: %.3f' % avg)


# In[111]:


# Spearman Correlation with first drug and random drugs
l=[]
for i in random.sample(range(1, 600), 10):
    corr, _ = spearmanr(vectors[0],vectors[i])
    l.append(corr)
    print('Spearmans correlation: %.3f' % corr)
avg=np.mean(l)
print('Spearmans correlation average: %.3f' % avg)


# In[112]:


# Kendall Tau's correlation with first drug and random drugs
l=[]
for i in random.sample(range(1, 600), 10):
    corr, _ = kendalltau(vectors[0],vectors[i])
    l.append(corr)
    print('Kendalls tau: %.3f' % corr)
avg=np.mean(l)
print('Kendalls tau average: %.3f' % avg)


# In[113]:


# Cosine Similarity with first drug and random drugs
l=[]
for i in random.sample(range(1, 600), 10):
    cos_sim = cosine_similarity(vectors[0].reshape(1,-1),vectors[i].reshape(1,-1))
    l.append(cos_sim)
    print('Cosine similarity: %.3f' % cos_sim)
avg=np.mean(l)
print('Cosine similarity average: %.3f' % avg)


# In[114]:


# Jaccard's similarity among two first drug and random drugs
l=[]
for i in random.sample(range(1, 600), 10):
    jacc = jaccard_score(vectors[0],vectors[i], average='macro')
    l.append(jacc)
    print('Jaccard similarity: %.3f' % jacc)
avg=np.mean(l)
print('Jaccard similarity average: %.3f' % avg)


# In[115]:


#Euclidean Distance between two first drug and random drugs
for i in random.sample(range(1, 600), 10):
    dst = distance.euclidean(vectors[0],vectors[i])
    print('Euclidean distance: %.3f' % dst)


# In[116]:


#Manhattan Distance between two first drug and random drugs
for i in random.sample(range(1, 600), 10):
    dst = distance.cityblock(vectors[0],vectors[i])
    print('Manhattan distance: %.3f' % dst)


# # Building Recommender system

# In[68]:


# Choosing Cosine Similarity for recommender system building
similarity = cosine_similarity(vectors)
similarity[0]


# In[118]:


# Sorting similarity in descending orders to make more similar drugs on top.
sorted((list(enumerate(similarity[2]))), reverse=True, key=lambda x:x[1])[0:11]


# In[138]:


#Defing a function to recommend drug based on condition selected
def recommend(condition):
    drug_index = new_df_lemma[new_df_lemma['condition'] == condition].index[0]
    distances = similarity[drug_index]
    drug_list = sorted((list(enumerate(distances))), reverse=True, key=lambda x:x[1])[0:5]
    for i in drug_list:
        print(new_df_lemma.iloc[i[0]].drugName)


# In[139]:


# Testing the Recommender system
recommend('Varicose Veins')


# In[141]:


# PKL files are imported for deploying the application into Heroku Cloud App
pickle.dump(new_df_lemma.to_dict(), open('drugs_dict.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




