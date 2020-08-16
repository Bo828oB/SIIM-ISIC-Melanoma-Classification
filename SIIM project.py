#!/usr/bin/env python
# coding: utf-8

# # <center>SIIM-ISIC Melanoma Classification</center>
# ### <center>Identify melanoma in lesion images</center>

# #### Problems of Classification in this competence
# 
# * Images and tabular data
# * imbalanced classification ( from EDA )
# 

# #### Basic steps to solve the problems
# 
# * EDA 
# * Dealing with missing value
# * Transforming table data
# * Asseccing images and image augmentation
# * Predicting the melanoma with specific metrics and learning rate using pre-trained Inception V3 Dense

# In[ ]:


import pandas as pd
import numpy as np
import csv


import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns



import os
import random


get_ipython().system('pip install pandasql')
from pandasql import sqldf
pysqldf = lambda q: sqldf(q,globals())


from PIL import Image
import cv2


from tqdm import tqdm, tqdm_notebook
import gc





import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model


from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split


from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
import keras.backend as K

import warnings

warnings.filterwarnings('ignore') 


# ### EDA

# In[ ]:


# import images
# Directory
directory = '/Users/XXXX/Downloads'  # hide ^*^
# Import the 2 csv s
train_df = pd.read_csv(directory + '/train.csv')
test_df = pd.read_csv(directory + '/test.csv')


# In[ ]:


# Create the paths
path_train = directory + '/jpeg/train/' + train_df['image_name'] + '.jpg'
path_test = directory + '/jpeg/test/' + test_df['image_name'] + '.jpg'

# Append to the original dataframes with image paths
train_df['path_jpeg'] = path_train
test_df['path_jpeg'] = path_test


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


print("The train data has {} records and for each record has {} columns  ".format(train_df.shape[0],train_df.shape[1]))
print("The test data has {} records and for each record has {} columns  ".format(test_df.shape[0],test_df.shape[1]))


# In[ ]:


print(train_df.info())
print(test_df.info())


# In[ ]:


## now we have know some columns has missing value and know exact missing number and unique values for categorical data
# check the column name and unique value, missing value
ColMiss = []
ColMissN = []
for column in list(train_df.columns):
    NumUni = len(list(train_df[column].unique()))
    NumNan = train_df[column].isna().sum()
    if NumNan !=0:
        ColMiss.append(column)
        ColMissN.append(NumNan)
    print('{} has {} unique value and {} NaN value in {} train data \n'.format(column,NumUni,NumNan,train_df.shape[0]))
    

#test missing value
ColMiss = []
ColMissN = []
for column in list(test_df.columns):
    NumUni = len(list(test_df[column].unique()))
    NumNan = test_df[column].isna().sum()
    if NumNan !=0:
        ColMiss.append(column)
        ColMissN.append(NumNan)
    print('{} has {} unique value and {} NaN value in {} test data \n'.format(column,NumUni,NumNan,test_df.shape[0]))


# In[ ]:


train_cl = train_df[['sex', 'age_approx','anatom_site_general_challenge', 'diagnosis', 'benign_malignant','target']]


# In[ ]:


# get information from univariate distribution in train data (for prectice sql)

df_sex = pysqldf("SELECT sex, count(*)*100.0/(SELECT count(*) FROM train_cl) as count_sex FROM train_cl GROUP BY sex;")
df_age = pysqldf("SELECT age_approx, count(*)*100.0/(SELECT count(*) FROM train_cl)as count_age FROM train_cl GROUP BY age_approx;")
df_site = pysqldf("SELECT anatom_site_general_challenge, count(*)*100.0/(SELECT count(*) FROM train_cl) as count_site FROM train_cl GROUP BY anatom_site_general_challenge;")
df_diagnosis = pysqldf("SELECT diagnosis, count(*)*100.0/(SELECT count(*) FROM train_cl) as count_diagnosis FROM train_cl GROUP BY diagnosis;")
df_benign = pysqldf("SELECT benign_malignant, count(*)*100.0/(SELECT count(*) FROM train_cl) as count_benign FROM train_cl GROUP BY benign_malignant;")
df_target = pysqldf("SELECT target, count(*)*100.0/(SELECT count(*) FROM train_cl) as count_target FROM train_cl GROUP BY target;")


# In[ ]:


# get information from univariate distribution in train data
test_cl = test_df[['sex', 'age_approx','anatom_site_general_challenge']]
df_sex_t = pysqldf("SELECT sex, count(*)*100.0/(SELECT count(*) FROM test_cl) as count_sex FROM test_cl GROUP BY sex;")
df_age_t = pysqldf("SELECT age_approx, count(*)*100.0/(SELECT count(*) FROM test_cl) as count_age FROM test_cl GROUP BY age_approx;")
df_site_t = pysqldf("SELECT anatom_site_general_challenge, count(*)*100.0/(SELECT count(*) FROM test_cl) as count_site FROM test_cl GROUP BY anatom_site_general_challenge;")
    


# In[ ]:


### train_df['age_approx'].value_counts()### is also for univariate distribution 


# In[ ]:


print(list(df_age['age_approx']))
print(list(df_sex['sex']))
print(list(df_site['anatom_site_general_challenge']))
print(list(df_site_t['anatom_site_general_challenge']))


# In[ ]:


#because None or nan can't be plot, so first replace it with a str 'NaN'
df_age['age_approx'] = ['NaN', 0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0]
df_sex['sex'] = ['NaN', 'female', 'male']
df_site['anatom_site_general_challenge'] = ['NaN', 'head/neck', 'lower extremity', 'oral/genital', 'palms/soles', 'torso', 'upper extremity']
df_site_t['anatom_site_general_challenge'] = ['NaN', 'head/neck', 'lower extremity', 'oral/genital', 'palms/soles', 'torso', 'upper extremity']


# In[ ]:


#plot about age distribution

f, axes = plt.subplots(2,1,figsize=(20,9))

ax = sns.barplot(x='age_approx',y = 'count_age',data=df_age,ax=axes[0])

categories = ax.get_xticks()
ax.set_title('The distribution of age include NaN in train data',fontsize=14, fontweight='bold')
for cate in categories:
    
    y = df_age['count_age'][cate]+0.5

    ax.text(
        cate, 
        y, 
        f"{round(df_age['count_age'][cate],2)}%", 
        ha='center', 
        va='center', 
        size=10,
        color='black')
      
ax_1 = sns.barplot(x='age_approx',y='count_age',data=df_age_t,ax = axes[1])
categories_1 = ax_1.get_xticks()
ax_1.set_title('The distribution of age include NaN in test data',fontsize=14, fontweight='bold')
for cate_1 in categories_1:
    
    y = df_age_t['count_age'][cate_1]+0.2

    ax_1.text(
        cate_1, 
        y, 
        f"{round(df_age_t['count_age'][cate_1],2)}%", 
        ha='center', 
        va='center', 
        size=10,
        color='black')


# In[ ]:


pysqldf("SELECT AVG(CAST(age_approx AS int)) as avg_age,MIN(CAST(age_approx AS int)) as min_age, MAX(CAST(age_approx AS int)) as max_age FROM train_cl WHERE age_approx is not 'NaN';")

# train data 


# In[ ]:


pysqldf("SELECT AVG(CAST(age_approx AS int)) as avg_age,MIN(CAST(age_approx AS int)) as min_age, MAX(CAST(age_approx AS int)) as max_age FROM test_cl WHERE age_approx is not 'NaN';")
# test data


# 
# ---
# 
# 
# 1. So we can see from the plot, the Missing value is a very small part of whole train data, and the age distribution is approximate normal distribution.
# 
# 
# ---
# 
# 
# 2. The test data is also approximate distribution distribution, test data has a bit higher average age and min age
# 
# 
# ---
# 
# 

# In[ ]:


#sex



f, axes = plt.subplots(2,1,figsize=(10,8))

ax = sns.barplot(x='sex',y = 'count_sex',data=df_sex,ax=axes[0])

categories = ax.get_xticks()
ax.set_title('The distribution of sex include NaN in train data',fontsize=14, fontweight='bold')
for cate in categories:
    
    y = df_sex['count_sex'][cate]+0.5

    ax.text(
        cate, 
        y, 
        f"{round(df_sex['count_sex'][cate],2)}%", 
        ha='center', 
        va='center', 
        size=10,
        color='black')
      
ax_1 = sns.barplot(x='sex',y='count_sex',data=df_sex_t,ax = axes[1])
categories_1 = ax_1.get_xticks()
ax_1.set_title('The distribution of sex include NaN in test data',fontsize=14, fontweight='bold')
for cate_1 in categories_1:
    
    y = df_sex_t['count_sex'][cate_1]+0.2

    ax_1.text(
        cate_1, 
        y, 
        f"{round(df_sex_t['count_sex'][cate_1],2)}%", 
        ha='center', 
        va='center', 
        size=10,
        color='black')


# 
# 
# 
# ---
# 
# 
# 1.   In the train data , the difference of female and male category is very small
# 
# ---
# 
# 
# 2.   Test data, no missing value and the percentage of male patients records is more than  10% than female patients
# 
# 
# 
# 
# ---
# 
# 
# 
# 

# In[ ]:


#plot about body position distribution

f, axes = plt.subplots(2,1,figsize=(15,9))

ax = sns.barplot(x='anatom_site_general_challenge',y = 'count_site',data=df_site,ax=axes[0])

categories = ax.get_xticks()
ax.set_title('The distribution of position categories include NaN in train data',fontsize=14, fontweight='bold')
for cate in categories:
    
    y = df_site['count_site'][cate]+0.5

    ax.text(
        cate, 
        y, 
        f"{round(df_site['count_site'][cate],2)}%", 
        ha='center', 
        va='center', 
        size=10,
        color='black')
      
ax_1 = sns.barplot(x='anatom_site_general_challenge',y='count_site',data=df_site_t,ax = axes[1])
categories_1 = ax_1.get_xticks()
ax_1.set_title('The distribution of position categories include NaN in test data',fontsize=14, fontweight='bold')
for cate_1 in categories_1:
    
    y = df_site_t['count_site'][cate_1]+0.2

    ax_1.text(
        cate_1, 
        y, 
        f"{round(df_site_t['count_site'][cate_1],2)}%", 
        ha='center', 
        va='center', 
        size=10,
        color='black')


# The most body part is torso which is the biggest surface of our body.
# 
# the percentage of body site is correspond to the surface of body part for the first impression

# In[ ]:


#diagnose
plt.figure(figsize=(20,9))

ax = sns.barplot(x='diagnosis',y = 'count_diagnosis',data=df_diagnosis)

categories = ax.get_xticks()
ax.set_title('The distribution of dianosis in train data',fontsize=14, fontweight='bold')
for cate in categories:
    
    y = df_diagnosis['count_diagnosis'][cate]+0.5

    ax.text(
        cate, 
        y, 
        f"{round(df_diagnosis['count_diagnosis'][cate],2)}%", 
        ha='center', 
        va='center', 
        size=10,
        color='black')


# In[ ]:


print(df_benign)
print(df_target)


# ---
# 
# Total unbalanced target data, only 2% of patients is malignant.
# For the model design, we need to consider the solution for unbalanced data
# 
# ---

# In[ ]:


plt.figure(figsize=(20,8))
ax = sns.countplot(x="target", hue="sex", data=train_df)

ax.set_title('The relation between target and sex')
ax.set_ylabel("Count")
ax.legend( loc='upper right')


# In[ ]:


# Plot age and target 
plt.figure(figsize=(15,10))


ax = sns.kdeplot(train_df[train_df['target'] == 0]['age_approx'],color='#1A1A1D',label='Benign')
ax = sns.kdeplot(train_df[train_df['target'] == 1]['age_approx'],color='#C3073F',label='Malignant')
ax.set_title('Age Distribution by result classification')


# ---
# For malignant patients, they are average elder.
# 
# ---

# In[ ]:


#age and target boxplot and distribution
plt.figure(figsize=(15,8))
ax = sns.boxplot(x="target", y="age_approx", hue="sex",
                 data=train_df, linewidth=2.5)
ax.set_title('The relation between age,sex and target')
ax.set_ylabel('age')


# In[ ]:


plt.figure(figsize=(20,8))
ax=sns.lineplot(x='age_approx',y='target',hue = 'sex',data=train_df)
ax.set_title('The percentage of target among age and sex')
ax.set_xlabel('age')


# 1. the percentage of male patients with malignant increase dramatic  after 80 age. 
# 
# 2. patients after 70 years old are more likely to get the disease of malignant.  
# 
# 3. the average age for both male and female benign patient is lower than malignan patient.
# 

# In[ ]:


# Deal with missing value and one-hot transformation


# In[ ]:


#deal with tabular data 

train_df['anatom_site_general_challenge'].fillna('unknown', inplace=True)
test_df['anatom_site_general_challenge'].fillna('unknown', inplace=True)   
train_df['sex'].fillna('unknown', inplace=True)

train_df['age_approx'].fillna(-1, inplace=True)


# In[ ]:



#deal with categorical features in train data

encoder=OneHotEncoder(sparse=False)

train_df_encoded_sex = pd.DataFrame(encoder.fit_transform(train_df[['sex']]))

train_df_encoded_sex.columns = encoder.get_feature_names(['sex'])


train_df= pd.concat([train_df, train_df_encoded_sex ], axis=1)
train_df_encoded_site = pd.DataFrame(encoder.fit_transform(train_df[['anatom_site_general_challenge']]))

train_df_encoded_site.columns = encoder.get_feature_names(['anatom_site_general_challenge'])

train_df= pd.concat([train_df, train_df_encoded_site ], axis=1)

#deal with categorical features in test data
encoder=OneHotEncoder(sparse=False)

test_df_encoded_sex = pd.DataFrame(encoder.fit_transform(test_df[['sex']]))

test_df_encoded_sex.columns = encoder.get_feature_names(['sex'])


test_df= pd.concat([test_df, test_df_encoded_sex ], axis=1)

test_df_encoded_site = pd.DataFrame(encoder.fit_transform(test_df[['anatom_site_general_challenge']]))

test_df_encoded_site.columns = encoder.get_feature_names(['anatom_site_general_challenge'])

test_df= pd.concat([test_df, test_df_encoded_site ], axis=1)


# In[ ]:


train_df.pop('sex_unknown') # delete sex-unknown because test doesn't have sex missing value


# In[ ]:


def get_tab(df):
    """select tabular data and normalize"""
    tab = df[['age_approx','sex_female', 'sex_male',
       'anatom_site_general_challenge_head/neck',
       'anatom_site_general_challenge_lower extremity',
       'anatom_site_general_challenge_oral/genital',
       'anatom_site_general_challenge_palms/soles',
       'anatom_site_general_challenge_torso',
       'anatom_site_general_challenge_unknown',
       'anatom_site_general_challenge_upper extremity']].values
    tab = tf.keras.utils.normalize(tab, axis=0, order=2)
    tab = tf.convert_to_tensor(tab)
    return tab


# In[ ]:


#deal with img


# In[ ]:


#read and resize image
def resize_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    new_img = cv2.resize(img,(img_size,img_size),interpolation = cv2.INTER_AREA)
    return new_img


# In[ ]:


# image augmentation
tf.compat.v1.random.set_random_seed(1234)
train_augmentation = tf.keras.Sequential([
    
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.RandomContrast(0.1, seed=tf.compat.v1.random.set_random_seed(1234), name=None)
 
])


# In[ ]:


#deal with img
def resize_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    new_img = cv2.resize(img,(img_size,img_size),interpolation = cv2.INTER_AREA)
    return new_img


# In[ ]:


tf.compat.v1.random.set_random_seed(1234)
train_augmentation = tf.keras.Sequential([
    
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.RandomContrast(0.1, seed=tf.compat.v1.random.set_random_seed(1234), name=None)
 
])

test_augmentation = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.Rescaling(1./255)])


# In[ ]:


# transform imag 

img_size = 256
batch_size = 16 #16 images per batch
path=directory+'/jpeg/train/'

train_img_ids = train_df['image_name'].values
n_batches = len(train_img_ids)//batch_size + 1

features = {}
for b in tqdm_notebook(range(n_batches)):
    start = b*batch_size
    end = (b+1)*batch_size
    batch_ids = train_img_ids[start:end]
    batch_images = np.zeros((len(batch_ids),img_size,img_size,3))
    for i,img_id in enumerate(batch_ids):
        try:
            batch_images[i] = resize_image(os.path.join(path,img_id+'.jpg'))
        except:
            pass
    batch_augs = train_augmentation(batch_images)
    #batch_augs = model_aug(batch_augs,256)
    features[b] = batch_augs
train_img = tf.concat(list(features.values()), axis=0)


# In[ ]:


# transform test images
img_size = 256
batch_size = 16 #16 images per batch
path=directory+'/jpeg/test/'

test_img_ids = test_df['image_name'].values
n_batches = len(test_img_ids)//batch_size + 1

features_test = {}
for b in tqdm_notebook(range(n_batches)):
    start = b*batch_size
    end = (b+1)*batch_size
    batch_ids = test_img_ids[start:end]
    batch_images = np.zeros((len(batch_ids),img_size,img_size,3))
    for i,img_id in enumerate(batch_ids):
        try:
            batch_images[i] = resize_image(os.path.join(path,img_id+'.jpg'))
        except:
            pass
    batch_augs = train_augmentation(batch_images)
    #batch_augs = model_aug(batch_augs,256)
    features_test[b] = batch_augs
test_img = tf.concat(list(features_test.values()), axis=0)


# In[ ]:


# get tab train and test
train_tab = get_tab(train_df)
test_tab = get_tab(test_df)


# In[ ]:


train_labels = tf.convert_to_tensor(train_df.target.values)


# ### To solve imbalanced classification, I choose upsamping the target ==1 data and keep the percentage of 1:0 in train and validation data , after then random the order
# 

# In[ ]:




train_1_all = train_df[train_df['target']==1]
train_0_all = train_df[train_df['target']==0]


# In[ ]:


from sklearn.model_selection import train_test_split
train_0, validation_0 = train_test_split(train_0_all, test_size=0.2, random_state=42)
train_1, validation_1 = train_test_split(train_1_all, test_size=0.2, random_state=42)


# In[ ]:


train_img_1 = tf.gather(train_img,(pos_index+neg_index))
train_tab_1 = tf.gather(train_tab,(pos_index+neg_index))
train_labels_1 = tf.gather(train_labels,(pos_index+neg_index))


# In[ ]:


len(train_img_1),len(train_tab_1),len(train_labels_1)


# In[ ]:


#upsampleing train_0 and validation_1 
neg = len(train_0)
pos = len(train_1)
neg_index = list(train_0.index)
pos_index = list(train_1.index)
val_index_0 = list(validation_0.index)
val_index_1 = list(validation_1.index)
val_index = val_index_0+val_index_1


# In[ ]:


val_img = tf.gather(train_img,val_index)
val_tab = tf.gather(train_tab,val_index)
val_labels = tf.gather(train_labels,val_index)


# In[ ]:


ids = np.arange(neg)
choices = np.random.choice(ids,pos)
res_1_tab = tf.gather(train_tab,choices)
res_1_img = tf.gather(train_img,choices)
res_1_labels = tf.gather(train_labels,choices)

res_1_all_tab = tf.concat([train_tab_1,res_1_tab],axis=0)
res_1_all_img = tf.concat([train_img_1,res_1_img],axis=0)
res_1_all_labels = tf.concat([train_labels_1,res_1_labels],axis=0)
order = np.arange(res_1_all_tab.shape[0])
np.random.shuffle(order)
res_1_all_tab = tf.gather(res_1_all_tab,order)
res_1_all_img = tf.gather(res_1_all_img,order)
res_1_all_labels = tf.gather(res_1_all_labels,order)


# In[ ]:


#random validation data 
order = np.arange(val_labels.shape[0])
np.random.shuffle(order)
val_img = tf.gather(val_img,order)
val_tab = tf.gather(val_tab,order)
val_labels = tf.gather(val_labels,order)


# In[ ]:


print(res_1_all_labels.shape[0])
print(res_1_all_img.shape[0])
print(res_1_all_tab.shape[0])


# In[ ]:


res_1_all_tab.shape


# In[ ]:


#Inception_v3 weights,use one of this
WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
WEIGHTS_M = 'https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'


# In[ ]:


weights_path = tf.keras.utils.get_file(
                'inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_M,
                cache_subdir='models',
                md5_hash='9a0d58056eeedaa3f26cb7ebd46da564')


# In[ ]:


from tensorflow.keras.applications.inception_v3 import InceptionV3

#local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
img_size = 256
pre_trained_model = InceptionV3(input_shape = (img_size, img_size, 3), 
                                include_top = True, 
                                weights = None)

pre_trained_model.load_weights(weights_path)

for layer in pre_trained_model.layers:
    layer.trainable = False
last_layer = pre_trained_model.get_layer('mixed10')
last_output = last_layer.output
img_x = tf.keras.layers.GlobalAveragePooling2D()(last_output)
model_aug = Model( pre_trained_model.input, img_x)


# In[ ]:


def aug_pre(img_input,img_size):
    img_input = tf.keras.layers.Input(shape=(img_size,img_size,3))
    
    img_x = tf.keras.applications.EfficientNetB5(
        include_top=False, # only false the model can specify the input shape
        weights="imagenet",
        input_shape=(img_size,img_size,3))(img_input)
    img_x = tf.keras.layers.GlobalAveragePooling2D()(img_x)
    return img_x


# In[ ]:


# model , early stop and store the best val-auc model


# In[ ]:


METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.AUC(curve='ROC',name='auc')
]


# In[ ]:


Earlystopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc', 
    verbose=1,
    patience=5,
    mode='max',
    restore_best_weights=True)
directory = "/Users/sunbo/Downloads/"

checkpoint_path = directory+"weights/fold_6.hdf5t"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
Checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 monitor='val_auc',
                                                 save_best_only=True,
                                                 mode='max',
                                                 
                                                 verbose=1)


my_callbacks = [Checkpoint, Earlystopping]


# In[ ]:


#train model
tab_size = 10
img_size = 256


def make_model(lr):
    """"""
    
    img_input = tf.keras.layers.Input(shape=(img_size,img_size,3)) #image input with shape(256,256,3)
    img_x = pre_trained_model()(img_input) #apply Inception V3 model
    img_x = tf.keras.layers.GlobalAveragePooling2D()(img_x)
    tab_input = tf.keras.layers.Input(shape = tab_size)  #input tabular 
    tab_x = tf.keras.layers.Dense(16,activation = 'relu')(tab_input)
    concat = tf.keras.layers.concatenate([img_input,tab_x])  # concate image and tabular data 
    concat = tf.keras.layers.Dense(128, activation = 'relu')(concat)
    concat = tf.keras.layers.BatchNormalization()(concat)
    concat = tf.keras.layers.Dropout(0.2)(concat)
    concat = tf.keras.layers.Dense(32, activation = 'relu')(concat)
    concat = tf.keras.layers.Dropout(0.2)(concat)
    output = tf.keras.layers.Dense(1, activation = 'sigmoid')(concat)
    model = tf.keras.models.Model(inputs=[img_input,tab_input],outputs=[output])
    
    
    #model compile
    opt = tf.keras.optimizers.Adam(learning_rate=lr) # lr is changeable
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05) 
    model.compile(optimizer=opt,loss=loss,metrics=[METRICS]) 
    
    return model


# In[ ]:


#plot the loss and auc 
mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
def plot_metrics(history,i):
    metrics =  ['loss', 'auc']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,1,n+1)
        plt.plot(history.epoch,  history.history[metric],'o--', color=colors[(n+i)],label='Train_{}'.format(i))
        plt.plot(history.epoch, history.history['val_'+metric], 'o-',color=colors[(n+i)],label='Val_{}'.format(i))
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0,1])
        else:
            plt.ylim([0,1])

        plt.legend()


# In[ ]:


# fit model with for lr and get the best model per learning rate
lr = [1e-5,1e-4,1e-3,1e-2]


for i,lr in enumerate(lr):
    Earlystopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc', 
    verbose=1,
    patience=5,
    mode='max',
    restore_best_weights=True)
    

    checkpoint_path = directory+"weights/upsample_lr_{}.hdf5t".format(str(lr))
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    Checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 monitor='val_auc',
                                                 save_best_only=True,
                                                 mode='max',
                                                 
                                                 verbose=1)


    my_callbacks = [Checkpoint, Earlystopping]


    model = make_model(lr)
    #model.save_weights(checkpoint_path)
    
    
    
    history = model.fit([res_1_all_img,res_1_all_tab],res_1_all_labels,batch_size = 32,epochs=30,validation_data=([val_img,val_tab],val_labels),callbacks = [my_callbacks],verbose =1)
    
    
    model.save_weights(checkpoint_path)
    plot_metrics(history,i)


# In[ ]:


# best learning rate 
1e-3


# In[ ]:


Earlystopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc', 
    verbose=1,
    patience=5,
    mode='max',
    restore_best_weights=True)
directory = "/Users/sunbo/Downloads/"

checkpoint_path = directory+"weights/upsample_lr_{}.hdf5t_1".format(str(1e-3))
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
Checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 monitor='val_auc',
                                                 save_best_only=True,
                                                 mode='max',
                                                 
                                                 verbose=1)


my_callbacks = [Checkpoint, Earlystopping]


model = make_model(1e-3)
    #model.save_weights(checkpoint_path)
    
    
    
history = model.fit([res_1_all_img,res_tab_1_x],res_1_all_labels,batch_size = 32,epochs=40,validation_data=([val_img,val_tab_1],val_labels),callbacks = [my_callbacks],verbose =1)
    
    
model.save_weights(checkpoint_path)
plot_metrics(history,1)


# In[ ]:


# check the model and val_auc
results = model.evaluate([val_img,val_tab],val_labels, batch_size=32) 
print("test loss, test acc:", results)


# In[ ]:


predictions = model.predict([test_img,test_tab])


# In[ ]:


predictions


# In[ ]:


test_df.head()


# In[ ]:


# sbumission csv


# In[ ]:


test_df_summission = test_df[['image_name']]


# In[ ]:


test_df_summission['target'] = predictions


# In[ ]:


path_sub = directory+'submissions_1.csv'


# In[ ]:


test_df_summission.to_csv(path_sub,index=None)

