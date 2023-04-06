import json
import numpy as np
import pathlib

# globals
vid_details_root = 'F:\\WORK\\DATASETS\\jk_data_only_22\\video_details\\'
squat_root = 'F:\\WORK\\DATASETS\\jk_data_only_22\\openposedata\\'

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

def read_squat(directory, start_frame, end_frame):
    pose_data = np.zeros((18, end_frame-start_frame))
    all_files = sorted(pathlib.Path(directory).glob("*.json"))
    for i, file_path in enumerate(all_files[start_frame:end_frame]):
        with open(file_path, "r") as read_file:
            data = json.load(read_file)
            if i == 0:
                squat_index = 0
            elif data['people']:
                keypoints = np.array(data['people'][0]['pose_keypoints_2d']).reshape(-1, 3).T
                hips_y = (keypoints[1, 11] + keypoints[1, 12]) / 2
                if hips_y < prev_hips_y:
                    squat_index += 1
            prev_hips_y = hips_y
            
            if squat_index >= len(vo.class_labels):
                break
                
            expected_label = vo.class_labels[squat_index]
            actual_label = "correct"
            if i < vo.in_and_outs[squat_index*2] - start_frame or i >= vo.in_and_outs[squat_index*2+1] - start_frame:
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

vo = read_json_meta(vid_details_root, 'J_01.json')
video_folder = squat_root + 'J_01'

pose_data = read_squat(video_folder, vo.in_and_outs[0], vo.in_and_outs[-1])

print(pose_data)
