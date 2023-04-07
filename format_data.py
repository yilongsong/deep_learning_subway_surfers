

# cleardataset.py

# yilong song
# Apr 6, 2023


'''Script for clearly data collected in the five not downsampled image folders
'''

import os

dir = 'dataset'

for folder in os.listdir(dir):
    if folder == '.DS_Store' or folder == 'raw':
        continue
    for file in os.listdir(dir+'/'+folder):
        os.remove(dir+'/'+folder+'/'+file)