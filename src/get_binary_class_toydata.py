import numpy as np
import pandas as pd
import os

pj_root = os.path.join(os.getcwd(), '..')
data_dir = os.path.join(pj_root, 'data')
src_dir = os.getcwd()

negative = np.random.normal(0,2,(2,1750))
negative = pd.DataFrame(negative.T, columns=['feature_x','feature_y'])
negative['response'] = 0

positive = np.random.normal(1,0.5,(2,250))
positive = pd.DataFrame(positive.T, columns=['feature_x','feature_y'])
positive['response'] = 1

binary_toydata = pd.concat([negative, positive]).reset_index(drop=True)

output = os.path.join(data_dir, 'binary_toydata.csv')
binary_toydata.to_csv(output, index=False)