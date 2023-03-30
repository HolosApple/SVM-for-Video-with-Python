import cv2
import json
import numpy as np
import os
import pandas as pd
import pathlib

# globals
vid_details_root = 'F:\\WORK\\DATASETS\\jk_data_only_22\\video_details\\'
squat_root = 'F:\\WORK\\DATASETS\\jk_data_only_22\\openposedata\\'


# Define a class for video object
class VidObj:
    def __init__(self, class_labels, in_and_outs):
        self.class_labels = class_labels
        self.in_and_outs = in_and_outs
        self.num_squats = len(class_labels)

# Read the JSON metadata and return a VidObj object
def read_json_meta(root_path, path):
    with open(root_path + path, "r") as read_file:
        data = json.load(read_file)
        in_and_outs = ((data['squats'])[0])['in_and_out']
        class_labels = ((data['squats'])[0])['class_label']
        return VidObj(class_labels, in_and_outs)

# Read the JSON files containing pose keypoints and return pose keypoints data for the frame
def extract_pose_keypoints(pose_keypoints_file):
    with open(pose_keypoints_file, 'r') as f:
        pose_data = json.load(f)
    pose_keypoints = pose_data['people'][0]['pose_keypoints']
    return pose_keypoints

# Define a function to extract pose keypoints from a JSON file and calculate squat correctness
def read_squat(video_folder, class_label, in_time, out_time):
    all_files = os.listdir(video_folder)
    frames = []
    for i in range(in_time, out_time):
        filename = all_files[i]
        f = os.path.join(video_folder, filename)
        # checking if it is a file
        if os.path.isfile(f):
            pose_keypoints = extract_pose_keypoints(f)
            # Check if the squat movement is correct or not
            if class_label == 'correct':
                # Extract the coordinates of the left ankle, right ankle, left hip, and right hip keypoints
                left_ankle = (pose_keypoints[39], pose_keypoints[40])
                right_ankle = (pose_keypoints[42], pose_keypoints[43])
                left_hip = (pose_keypoints[33], pose_keypoints[34])
                right_hip = (pose_keypoints[36], pose_keypoints[37])
                
                # Calculate the distance between the left ankle and left hip keypoints
                left_distance = ((left_ankle[0]-left_hip[0])**2 + (left_ankle[1]-left_hip[1])**2)**0.5
                
                # Calculate the distance between the right ankle and right hip keypoints
                right_distance = ((right_ankle[0]-right_hip[0])**2 + (right_ankle[1]-right_hip[1])**2)**0.5

            # Check if the distance between the left ankle and left hip keypoints is less than the distance between the right ankle and right hip keypoints
            if left_distance < right_distance:
                print('Squat movement is correct')
            else:
                print('Squat movement is incorrect')
        else:
            # If the squat is incorrect, we can calculate the angle between the hip and knee joints
            # If the angle is less than a threshold value, we can say that the squat movement is incorrect
            hip_x, hip_y = pose_data['people'][0]['pose_keypoints'][24:26]
            knee_x, knee_y = pose_data['people'][0]['pose_keypoints'][27:29]
            ankle_x, ankle_y = pose_data['people'][0]['pose_keypoints'][30:32]

            hip_knee_angle = np.degrees(np.arctan2(knee_y - hip_y, knee_x - hip_x) - np.arctan2(ankle_y - knee_y, ankle_x - knee_x))

            if hip_knee_angle < 90:
                # The angle is less than 90 degrees, which indicates that the squat movement is incorrect
                print('Incorrect squat movement: Incorrect angle between hip and knee joints')
            else:
                # The angle is greater than or equal to 90 degrees, which indicates that the squat movement is correct
                print('Correct squat movement: Angle between hip and knee joints is greater than or equal to 90 degrees')

def process_videos(video_folder, vid_details_folder, out_folder):
    # Read the video details CSV file
    vid_details_file = os.path.join(vid_details_folder, 'video_details.csv')
    vid_details = pd.read_csv(vid_details_file)

    # Iterate over the rows of the video details CSV file and process each video
    for index, row in vid_details.iterrows():
        # Get the video ID, start time, and end time
        video_id = row['video_id']
        start_time = row['start_time']
        end_time = row['end_time']

        # Read the JSON metadata file for the video
        meta_file = os.path.join(vid_details_folder, video_id + '.json')
        vid_obj = read_json_meta(vid_details_folder, video_id + '.json')

        # Iterate over the squats in the video and process each one
        for i in range(vid_obj.num_squats):
            class_label = vid_obj.class_labels[i]
            in_time = vid_obj.in_and_outs[i][0]
            out_time = vid_obj.in_and_outs[i][1]

            # Check if the squat is within the specified start and end times
            if out_time < start_time or in_time > end_time:
                continue

            # Get the directory containing the pose keypoints JSON files for the squat
            squat_dir = os.path.join(video_folder, video_id, str(i))

            # Read the pose keypoints JSON files for the squat and process them
            read_squat(squat_dir, class_label, in_time, out_time)
