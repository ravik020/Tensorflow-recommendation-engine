
import numpy as np
import pandas as pd
import tensorflow as tf

music = pd.read_csv('Merged.csv', header = 0, encoding = 'latin1')
genre_s = music['genre_top'].unique()
num_s = np.arange(1, genre_s.size + 1)
genre_map = pd.DataFrame([genre_s, num_s]).T
genre_map.columns = ['type', 'label']
music_label = music.merge(genre_map, left_on = 'genre_top', right_on = 'type', how = 'outer')

col_names = ['acousticness', 'danceability', 'energy', \
                       'instrumentalness', 'liveness', 'speechiness', \
                       'tempo', 'valence', 'number', 'artist_discovery', \
                       'artist_familiarity', 'artist_hotttnesss', 'song_currency',\
                       'song_hotttnesss', 'label']

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
