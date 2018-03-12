import numpy as np
import tensorflow as tf
import pandas as pd

import copy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


music = pd.read_csv('Merged.csv', header = 0, encoding = 'latin1')
genre_s = music['genre_top'].unique()
num_s = np.arange(1, genre_s.size + 1)
genre_map = pd.DataFrame([genre_s, num_s]).T
genre_map.columns = ['type', 'label']
music_label = music.merge(genre_map, left_on = 'genre_top', right_on = 'type', how = 'outer')

col_names = ['acousticness', 'danceability', 'energy','instrumentalness', \
             'liveness', 'speechiness', 'tempo', 'valence','label']

music_dnn = music_label[col_names]

training_data = music_dnn[music_dnn['label'] != 6].drop('label',axis = 1)
training_target = music_dnn[music_dnn['label'] != 6]['label']
targetting_data = music_dnn[music_dnn['label'] == 6].drop('label',axis = 1)
targetting_target = music_dnn[music_dnn['label'] == 6]['label']

def input_fn_train():
    x = tf.constant(training_data.as_matrix(), tf.float32, training_data.shape)
    y = tf.constant(training_target.as_matrix(), tf.int32, training_target.shape)
    return x, y

def input_fn_target():
    x = tf.constant(targetting_data.as_matrix(), tf.float32, targetting_data.shape)
    y = tf.constant(targetting_target.as_matrix(), tf.int32, targetting_target.shape)
    return x, y

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=13)]

classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=[10, 20, 10],
                                              n_classes=14 )

classifier.fit(input_fn=input_fn_train, steps=2000)

result = list(classifier.predict(input_fn=input_fn_target))

print (len(result))


targetting_data['label'] = result

training_data = music_dnn[music_dnn['label'] != 6]
fin_frame = training_data.append(targetting_data)


fin_frame.head()


col_names = ['acousticness', 'danceability', 'energy',\
             'instrumentalness', 'liveness', 'speechiness', \
             'tempo', 'valence']
music_numerical = fin_frame[col_names]
music_matrix = music_numerical.as_matrix()
scaler = StandardScaler()
scaler.fit(music_matrix)
new_matrix = scaler.transform(music_matrix)
music_std = pd.DataFrame(new_matrix, columns = col_names)

music_std = music_std[col_names]

music_std.head()


music_std['label'] = fin_frame['label']


music_std.head()


genre_map.columns = ['genre_top', 'label']
fin_output_frame = pd.merge(music_std,genre_map)

fin_output_frame.head()


music_cat = pd.get_dummies(fin_output_frame['genre_top'])
nf = pd.concat([music_std,music_cat], axis = 1)

nf.head()

norm_data = copy.copy(nf)
del norm_data['label']

norm_data.head()

def input_fn():
  return tf.constant(norm_data.as_matrix(), tf.float32, norm_data.shape), None

tf.logging.set_verbosity(tf.logging.ERROR)
kmeans = tf.contrib.learn.KMeansClustering(num_clusters=10, 
relative_tolerance=0.0001)
kmeans.fit(input_fn=input_fn)

clusters = kmeans.clusters()
print(clusters)
assignments = list(kmeans.predict_cluster_idx(input_fn=input_fn))

kmeans.score(input_fn=input_fn, steps = 50)


assign = pd.DataFrame(assignments)
output_frame = pd.concat([fin_frame, assign], axis=1)
output_frame = output_frame.rename(columns={0: 'Cluster'})


output_frame.head()
output_frame_merge = pd.merge(output_frame,genre_map)
output_cat = pd.get_dummies(output_frame_merge['genre_top'])
output_frame_fin = pd.concat([output_frame_merge,output_cat], axis = 1)


#rack_id = pd.read_csv('Merged.csv', encoding='latin-1', usecols=['track_id'])
#output_frame = pd.concat([track_id, output_frame], axis=1)
output_frame_fin.to_csv("Fin_Output_Cluster.csv")

