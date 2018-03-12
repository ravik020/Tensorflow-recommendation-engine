from pandas import Series
import pandas as pd
import numpy as np

#prepare tracks
tracks = pd.read_csv('tracks.csv', header= None)
tl = np.array(tracks.iloc[:2,:])
tl = tl.astype(str)
tl = np.core.defchararray.replace(tl, 'nan', '')
plus = lambda x, y: x + ' ' + y
new = np.vectorize(plus)(tl[0], tl[1])
new[0] = 'track_id'
tracks.columns = new
tracks = tracks[3:]


#prepare honest
honest = pd.read_csv('echonest.csv', header= None)
tl = np.array(honest.iloc[:3,:])
tl = tl.astype(str)
tl = np.core.defchararray.replace(tl, 'nan', '')
plus = lambda x, y, z: x + ' ' + y + ' ' + z
new = np.vectorize(plus)(tl[0], tl[1], tl[2])
new[0] = 'track_id'
honest.columns = new
honest = honest[4:]
