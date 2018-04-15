import numpy as np
import pandas as pd
import os

pj_root = os.path.join(os.getcwd(), '..')
data_dir = os.path.join(pj_root, 'data')
src_dir = os.getcwd()

def create_toy_data(arrays_x, arrays_y, class_label):
    data = np.array([arrays_x,arrays_y]).T
    data = pd.DataFrame(data, columns=['feature_x','feature_y'])
    data['response'] = class_label
    return data

classes_data = []
arrays_x = np.random.normal(0,2,(1,3750))[0]
arrays_y = np.random.normal(0,2,(1,3750))[0]
classes_data.append(create_toy_data(arrays_x,arrays_y, 0))

arrays_x = np.random.normal(1,0.5,(1,250))[0]
arrays_y = np.random.normal(1,0.5,(1,250))[0]
classes_data.append(create_toy_data(arrays_x,arrays_y, 1))

arrays_x = np.random.normal(-1,0.5,(1,250))[0]
arrays_y = np.random.normal(-1,0.5,(1,250))[0]
classes_data.append(create_toy_data(arrays_x,arrays_y, 2))

arrays_x = np.random.normal(-1,0.5,(1,250))[0]
arrays_y = np.random.normal(1,0.5,(1,250))[0]
classes_data.append(create_toy_data(arrays_x,arrays_y, 3))

arrays_x = np.random.normal(2,0.5,(1,250))[0]
arrays_y = np.random.normal(-1,0.5,(1,250))[0]
classes_data.append(create_toy_data(arrays_x,arrays_y, 4))

multi_toydata = pd.concat(classes_data)

output = os.path.join(data_dir, 'multi_toydata.csv')
multi_toydata.to_csv(output, index=False)