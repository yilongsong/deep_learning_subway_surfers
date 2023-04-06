

# cleardataset.py

# yilong song
# Apr 6, 2023

import os

dir = 'dataset'

for folder in os.listdir(dir):
    if folder == '.DS_Store':
        continue
    for file in os.listdir(dir+'/'+folder):
        os.remove(dir+'/'+folder+'/'+file)