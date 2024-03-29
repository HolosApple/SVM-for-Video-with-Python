import json
import numpy as np
import pathlib
from scipy import interpolate
import matplotlib.pyplot as plt
import pickle
#import os
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# # globals
vid_details_root = 'F:\\WORK\\DATASETS\\jk_data_only_22\\video_details\\'
squat_root = 'F:\\WORK\\DATASETS\\jk_data_only_22\\openposedata\\'
# vid_details_root = 'video_details/'
# squat_root = 'openposedata/'

class VidObj:
    def __init__(self, class_labels, in_and_outs):
        self.class_labels = class_labels
        self.in_and_outs = in_and_outs
        self.num_squats = len(class_labels)


def read_json_meta(root_path, path):
    with open(root_path + path, "r") as read_file:
        data = json.load(read_file)
        # data = eval(read_file.read())
        # print(root_path + path, type(data))
        in_and_outs = ((data['squats'])[0])['in_and_out']
        class_labels = ((data['squats'])[0])['class_label']
    return VidObj(class_labels, in_and_outs)


def interp_row(row_0, numcols):
    t_old = np.linspace(0, len(row_0) - 1, num=len(row_0))
    f = interpolate.interp1d(t_old, row_0)
    t_new = np.linspace(0, len(row_0) - 1, num=numcols)
    ynew = f(t_new)
    plt.plot(t_old, row_0, 'o', t_new, ynew, 'x')
    plt.show()

    return ynew


def read_squat_edit(directory, start_frame, end_frame):
    pose_data = np.zeros((18 * 2, end_frame - start_frame))
    all_files = sorted(pathlib.Path(directory).glob("*.json"))
    for i, file_path in enumerate(all_files[start_frame:end_frame]):
        with open(file_path, "r") as read_file:
            data = json.load(read_file)
            pose_keypoints = data['people'][0]['pose_keypoints']

            row = 0
            for ind in range(0, 54, 3):
                pose_data[row, i] = pose_keypoints[ind];
                pose_data[row + 1, i] = pose_keypoints[ind + 1]
                row = row + 2

    return pose_data


def read_squat(directory, start_frame, end_frame):
    pose_data = np.zeros((18, end_frame - start_frame))
    all_files = sorted(pathlib.Path(directory).glob("*.json"))
    for i, file_path in enumerate(all_files[start_frame:end_frame]):
        with open(file_path, "r") as read_file:
            data = json.load(read_file)
            if i == 0:
                squat_index = 0
            if squat_index >= len(vo.class_labels):
                break

            expected_label = vo.class_labels[squat_index]
            actual_label = "correct"
            if i < vo.in_and_outs[squat_index * 2] - start_frame or i >= vo.in_and_outs[
                squat_index * 2 + 1] - start_frame:
                actual_label = "incorrect_not_low"
            else:
                keypoints = np.array(data['people'][0]['pose_keypoints_2d']).reshape(-1, 3).T
                if keypoints[1, 0] - keypoints[1, 8] < -100:
                    actual_label = "incorrect_lean_fwd"
                elif keypoints[1, 15] - keypoints[1, 16] < 0:
                    actual_label = "incorrect_chin_tuck"
                elif keypoints[0, 0] - keypoints[0, 4] < -100:
                    actual_label = "incorrect_feet_close"

            if actual_label == expected_label:
                print(f"Frame {i} is Correct Squat")
            else:
                print(f"Frame {i} is {actual_label}")

            pose_data[:, i] = np.array(data['people'][0]['pose_keypoints_2d']).reshape(-1, 3).T[:, :-1].flatten()

    return pose_data

    # if a squat matrix has less than 50 columns, the function pads the matrix with zeros to make it 50 columns wide.
    # If a squat matrix has more than 50 columns, the function truncates it to have only the first 50 columns.
    def preprocessed_squats(squats):
        preprocessed_squats = []
        for squat in squats:
            squat_matrix = np.loadtxt(squat, delimiter=",")
            # pad or truncate the matrix to have 50 columns
            num_cols = squat_matrix.shape[1]
            if num_cols < 50:
                padding = np.zeros((squat_matrix.shape[0], 50 - num_cols))
                squat_matrix = np.hstack((squat_matrix, padding))
            elif num_cols > 50:
                squat_matrix = squat_matrix[:, :50]
            preprocessed_squats.append(squat_matrix)
    return preprocessed_squats


all_labels = []
all_squats = []

all_files_meta = sorted(pathlib.Path(vid_details_root).glob("*.json"))
for i, file_path in enumerate(all_files_meta):
    curr_vid_root = file_path.name[0:len(file_path.name) - 5]
    vo = read_json_meta(vid_details_root, curr_vid_root + '.json')
    video_folder = squat_root + curr_vid_root

    label_counter = 0
    for start_frame_ind in range(0, vo.num_squats * 2, 2):
        pose_data = read_squat_edit(video_folder, vo.in_and_outs[start_frame_ind], vo.in_and_outs[start_frame_ind + 1])
        all_squats.append(pose_data)
        all_labels.append(vo.class_labels[label_counter])
        label_counter = label_counter + 1


# Do PCA on X such that X still has num_squats (296?) columns but much less rows
# Train SVM and get validation data with X and a new vector y that you make from the all_labels
print(len(all_squats),len(all_labels)) #296 squats,
print(np.unique(all_labels)) #classes

min_feature_dim = 99 #cut off the feature frame
for sample in all_squats:
    if sample.shape[1]<min_feature_dim:
        min_feature_dim = sample.shape[1]
print(min_feature_dim)

X = []
for i in range(len(all_labels)):
    X.append(all_squats[i][:,:min_feature_dim].flatten()) #keep the first min_feature_dim frame data
X = np.array(X)
print(X.shape)

X_train,X_val,y_train,y_val = train_test_split(X,all_labels,random_state=0)
print(X_train.shape,X_val.shape)

pca = PCA(n_components=0.95)
X_train_trans = pca.fit_transform(X_train)
X_val_trans = pca.transform(X_val)
print(X_train_trans.shape)

svm = SVC()
svm.fit(X_train_trans, y_train)
pred_train = svm.predict(X_train_trans)
pred_val = svm.predict(X_val_trans)

print("train")
print(classification_report(y_train,pred_train,zero_division=1))
print("test")
print(classification_report(y_val,pred_val,zero_division=1))