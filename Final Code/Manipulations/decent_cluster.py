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


track_id = pd.read_csv('Merged.csv', encoding='latin-1', usecols=['track_id'])
output_frame = pd.concat([track_id, output_frame], axis=1)
output_frame.to_csv("Fin_Output_Cluster_3.csv")

output_frame_new = pd.concat([org_data, assign], axis=1)
output_frame_new = pd.concat([output_frame_new, music_cat], axis=1)
track_id = pd.read_csv('Merged.csv', encoding='latin-1', usecols=['track_id'])
output_frame_new = pd.concat([track_id, output_frame_new], axis=1)
output_frame_new.to_csv("Fin_Output_Cluster_new.csv")