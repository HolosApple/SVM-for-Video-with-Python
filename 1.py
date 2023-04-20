import json
import numpy as np
import pathlib
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.svm import SVC

import matplotlib.pyplot as plt

# globals
#vid_details_root = 'F:\\WORK\\DATASETS\\jk_data_only_22\\video_details\\'
#squat_root = 'F:\\WORK\\DATASETS\\jk_data_only_22\\openposedata\\'

vid_details_root = 'C:\\TEACHING\\DATASETS\\jk_data_only_22\\video_details\\'
squat_root = 'C:\\TEACHING\\DATASETS\\jk_data_only_22\\openposedata\\'

class VidObj:
    def __init__(self, class_labels, in_and_outs):
        self.class_labels = class_labels
        self.in_and_outs = in_and_outs
        self.num_squats = len(class_labels)

def read_json_meta(root_path, path):
    with open(root_path + path, "r") as read_file:
        data = json.load(read_file)
        in_and_outs = ((data['squats'])[0])['in_and_out']
        class_labels = ((data['squats'])[0])['class_label']
    return VidObj(class_labels, in_and_outs)

def interp_row(row_0, numcols):
    t_old = np.linspace(0, len(row_0) - 1, num=len(row_0))
    f = interpolate.interp1d(t_old, row_0)
    t_new = np.linspace(0, len(row_0) - 1, num=numcols)
    ynew = f(t_new)
    #plt.plot(t_old, row_0, 'o', t_new, ynew, 'x')
    #plt.show()

    return ynew

def read_squat_edit(directory, start_frame, end_frame):
    pose_data = np.zeros((18 * 2, end_frame - start_frame))
    all_files = sorted(pathlib.Path(directory).glob("*.json"))
    for i, file_path in enumerate(all_files[start_frame:end_frame]):
        with open(file_path, "r") as read_file:
            data = json.load(read_file)
            pose_keypoints = data['people'][0]['pose_keypoints']

            row = 0
            for ind in range(0,54,3):
                pose_data[row, i] = pose_keypoints[ind];
                pose_data[row + 1, i] = pose_keypoints[ind + 1]
                row = row + 2

    return pose_data


all_labels = []
all_squats = []

all_files_meta = sorted(pathlib.Path(vid_details_root).glob("*.json"))
for i, file_path in enumerate(all_files_meta):
    curr_vid_root = file_path.name[0:len(file_path.name) - 5]
    vo = read_json_meta(vid_details_root,  curr_vid_root + '.json')
    video_folder = squat_root + curr_vid_root

    label_counter = 0
    for start_frame_ind in range(0, vo.num_squats * 2, 2):
        pose_data = read_squat_edit(video_folder, vo.in_and_outs[start_frame_ind], vo.in_and_outs[start_frame_ind + 1])
        all_squats.append(pose_data)
        all_labels.append(vo.class_labels[label_counter])
        label_counter = label_counter + 1

all_squats_same_cols = []
n = 50
# Look up how to use pickle to save the 2 lists
for i in range(0, len(all_squats)):
    squat_1 = all_squats[i]
    squat_1_new = np.zeros((18 * 2, n))

    for row in range(0, 18 * 2):
        row_curr = squat_1[row, :]
        row_interp = interp_row(row_curr, n)
        squat_1_new[row, :] = row_interp

    all_squats_same_cols.append(squat_1_new)

# Sanity check
#squat_0 = all_squats[5]
#row_0 = squat_0[9, :]
#squat_interp = all_squats_same_cols[5]
#row_interp = squat_interp[9, :]
#t_new = np.linspace(0, len(row_0) - 1, num=n)
#t_old = np.linspace(0, len(row_0) - 1, num=len(row_0))
#plt.plot(t_old, row_0, 'o', t_new, row_interp, 'x')
#plt.show()
#  End Sanity check

# Next stage is to make X - where is column is a whole squat which 18 * 2 * 50 tall
X = np.zeros((18 * 2 * n, len(all_squats_same_cols)));
for i in range(0, len(all_squats_same_cols)):
    squat_curr = all_squats_same_cols[i]
    X[:, i] = squat_curr.flatten()
    print(2)

# Do PCA on X such that X still has num_squats (296?) columns but much less rows
# Train SVM and get validation data with X and a new vector y that you make from the all_labels
#Use PCA to reduce the dimensionality of the squat data to 2 dimensions:
pca = PCA(n_components=25)
X_pca = pca.fit_transform(X.T) # Note X was transposed so that we now have 296 squats with each represented by 25 features

#Train an SVM classifier on the PCA-transformed data:
svm = SVC()
svm.fit(X_pca, all_labels)


print(1)





