
# coding: utf-8

# In[14]:


from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pandas as pd
num_col = ['acousticness', 'danceability', 'energy',
       'instrumentalness', 'liveness', 'speechiness', 'tempo', 'valence']    

org_data = pd.read_csv('Merged.csv', encoding='latin-1')
nf = pd.read_csv('Merged.csv', encoding='latin-1', usecols=num_col)
matrix_nf = nf.as_matrix()
scaler = StandardScaler()
scaler.fit(matrix_nf)
nf = scaler.transform(matrix_nf)
nf = pd.DataFrame(nf)

nf_genre = pd.read_csv('Merged.csv', encoding='latin-1', usecols=['genre_top'])
music_cat = pd.get_dummies(nf_genre['genre_top'])
nf = pd.concat([nf,music_cat], axis = 1)

def input_fn():
  return tf.constant(nf.as_matrix(), tf.float32, nf.shape), None

kmeans = tf.contrib.learn.KMeansClustering(num_clusters=10,relative_tolerance=0.01)
kmeans.fit(input_fn=input_fn)

clusters = kmeans.clusters()
print(clusters)
print (kmeans.score(input_fn=input_fn, steps = 100))
assignments = list(kmeans.predict_cluster_idx(input_fn=input_fn))

assign = pd.DataFrame(assignments)
output_frame = pd.concat([org_data, assign], axis=1)
output_frame = output_frame.rename(columns={0: 'Cluster'})


# In[2]:


import os
import random
os.getcwd()
os.chdir("C:\\Users\\chapp\\Documents\\GitHub\\fma\\List")


# In[4]:


AUDIO_DIR='C:/Users/chapp/Courses/Python/Project'


# In[6]:


filenames = next(os.walk("C:\\Users\\chapp\\Documents\\GitHub\\fma\\List"))[2]


# In[8]:


filenames = [x.strip('.mp3') for x in filenames]


# In[9]:


os.chdir("C:\\Users\\chapp\\Documents\\GitHub\\fma")


# In[11]:


filenames


# In[15]:


output_frame.head()


# In[18]:


output_frame['track_id'] = output_frame['track_id'].astype(str)
output_frame['track_id'] = output_frame['track_id'].str.zfill(6)
fin_output_frame = output_frame[output_frame['track_id'].isin(filenames)]


# In[123]:


genre_theme = input("Enter Theme:")


# In[140]:


cluster_input=fin_output_frame[fin_output_frame['Cluster']==int(genre_theme)].track_id.tolist()
import random,pygame
track = random.choice(cluster_input)
sep = '/'
tid_str = track
filename = AUDIO_DIR + sep + tid_str[:3] + sep + tid_str + '.mp3'
print (filename)

pygame.mixer.init() 
pygame.mixer.music.load(filename)
pygame.mixer.music.play()
pygame.mixer.music.fadeout(20000)


# In[141]:


pygame.mixer.music.stop()


# In[116]:


track


# In[ ]:




