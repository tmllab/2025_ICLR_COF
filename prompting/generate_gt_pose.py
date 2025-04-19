import json
import numpy as np
import matplotlib.pyplot as plt
import os
"""
This is the script for generating keypoint ground-truth images
"""

MISSING_VALUE = 0

OPENPOSE_SKELETON = np.array([
                    [1, 2],
                    [1, 5],
                    [2, 3],
                    [3, 4],
                    [5, 6],
                    [6, 7],
                    [1, 8],
                    [8, 9],
                    [9, 10],
                    [1, 11],
                    [11, 12],
                    [12, 13],
                    [1, 0],
                    [0, 14],
                    [14, 16],
                    [0, 15],
                    [15, 17]])

OPENPOSE_KEYPOINTS = ['nose', 'neck', 'right_shoulder', 
                  'right_elbow', 'right_wrist', 'left_shoulder',
                  'left_elbow', 'left_wrist', 'right_hip',
                  'right_knee', 'right_ankle', 'left_hip',
                  'left_knee', 'left_ankle', 'right_eye', 
                  'left_eye', 'right_ear', 'left_ear']

OPENPOSE_KEYPOINTS_COLOUR = ['#ff0000', '#ff5500', '#ffaa00',
                         '#ffff00', '#aaff00', '#55ff00',
                         '#00ff00', '#00ff55', '#00ffaa',
                         '#00ffff', '#00aaff', '#0055ff',
                         '#0000ff', '#5500ff', '#aa00ff', 
                         '#ff00ff', '#ff00aa', '#ff0055'] 


OPENPOSE_COLOUR_MAP = ['#990000', '#993300', '#996600',
              '#999900', '#669900', '#339900',
              '#009900', '#009933', '#009966',
              '#009999', '#006699', '#003399',
              '#000099', '#330099', '#660099',
              '#990099', '#990066']


def convert_coco_to_openpose_cords(coco_keypoints_list):
    # coco keypoints: [x1,y1,v1,...,xk,yk,vk]       (k=17)
    #     ['Nose', Leye', 'Reye', 'Lear', 'Rear', 'Lsho', 'Rsho', 'Lelb',
    #      'Relb', 'Lwri', 'Rwri', 'Lhip', 'Rhip', 'Lkne', 'Rkne', 'Lank', 'Rank']
    # openpose keypoints: [y1,...,yk], [x1,...xk]   (k=18, with Neck)
    #     ['Nose', *'Neck'*, 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri','Rhip',
    #      'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear']
    indices = [0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 1, 2, 3, 4]
    y_cords = []
    x_cords = []
    v_cords = []
    for i in indices:
        xi, yi, vi = coco_keypoints_list[i*3:(i+1)*3]
        if vi == 0: # not labeled
            y_cords.append(MISSING_VALUE)
            x_cords.append(MISSING_VALUE)
            v_cords.append(MISSING_VALUE)
        elif vi == 1:   # labeled but not visible
            y_cords.append(yi)
            x_cords.append(xi)
            v_cords.append(1)
        elif vi == 2:   # labeled and visible
            y_cords.append(yi)
            x_cords.append(xi)
            v_cords.append(2)
        else:
            raise ValueError("vi value: {}".format(vi))
    # Get 'Neck' keypoint by interpolating between 'Lsho' and 'Rsho' keypoints
    l_shoulder_index = 5
    r_shoulder_index = 6
    l_shoulder_keypoint = coco_keypoints_list[l_shoulder_index*3:(l_shoulder_index+1)*3]
    r_shoulder_keypoint = coco_keypoints_list[r_shoulder_index*3:(r_shoulder_index+1)*3]
    if l_shoulder_keypoint[2] > 0 and r_shoulder_keypoint[2] > 0:
        neck_keypoint_y = int((l_shoulder_keypoint[1]+r_shoulder_keypoint[1])/2.)
        neck_keypoint_x = int((l_shoulder_keypoint[0]+r_shoulder_keypoint[0])/2.)
    else:
        neck_keypoint_y = neck_keypoint_x = MISSING_VALUE
    open_pose_neck_index = 1
    y_cords.insert(open_pose_neck_index, neck_keypoint_y)
    x_cords.insert(open_pose_neck_index, neck_keypoint_x)
    v_cords.insert(open_pose_neck_index, 2)

    return np.concatenate([np.expand_dims(x_cords, -1),
                           np.expand_dims(y_cords, -1),
                           np.expand_dims(v_cords, -1)], axis=1)


def visualize_keypoints(input_directory, annotations_file, output_directory):
    # Load annotations
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)

    # Create output directory if it does not exist
    os.makedirs(output_directory, exist_ok=True)

    # Process each image
    for img_details in annotations['images']:
        image_id = img_details['id']
        print("image_id", image_id)
        anns = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]
        if not anns:
            continue

        img_path = os.path.join(input_directory, img_details['file_name'])
        if not os.path.exists(img_path):
            continue

        X = np.zeros((img_details['height'], img_details['width'], 3), dtype=np.uint8)

        valid_objects = 0

        for ann in anns:
            keypoints_tmp = convert_coco_to_openpose_cords(ann['keypoints']).flatten()

            x = keypoints_tmp[0::3]
            y = keypoints_tmp[1::3]
            v = keypoints_tmp[2::3]

            keypoints = []
            for i in range(len(x)):
                keypoint = [x[i], y[i], v[i]]
                keypoints.append(keypoint)

            keypoints = np.array(keypoints)
            # Plot predicted keypoints on bounding box image
            point_x = []
            point_y = []
            valid = np.zeros(len(x))

            for i in range(len(x)):
                if keypoints[i][0] != 0 and keypoints[i][1] != 0:
                    valid[i] = 1
                point_x.append(keypoints[i, 0])
                point_y.append(keypoints[i, 1])

            already_draw_dot = []

            for i in range(len(OPENPOSE_SKELETON)):
                # joint a to joint b
                a = OPENPOSE_SKELETON[i, 0]
                b = OPENPOSE_SKELETON[i, 1]
                # if both are valid keypoints
                if valid[a] and valid[b]:
                    # linewidth = 5, linestyle = "--",
                    plt.plot([keypoints[a, 0], keypoints[b, 0]], [keypoints[a, 1], keypoints[b, 1]],
                                color=OPENPOSE_COLOUR_MAP[i])
                    if a not in already_draw_dot:
                        plt.scatter(point_x[a], point_y[a], color=OPENPOSE_KEYPOINTS_COLOUR[a], s=5)
                        already_draw_dot.append(a)
                    if b not in already_draw_dot:
                        plt.scatter(point_x[b], point_y[b], color=OPENPOSE_KEYPOINTS_COLOUR[b], s=5)
                        already_draw_dot.append(b)
            valid_objects += sum(valid)

        if valid_objects > 0:
            plt.axis('off')
            plt.margins(0, 0)
            plt.imshow(X)
            plt.savefig(os.path.join(output_directory, f"{img_details['file_name'].split('.')[0]}_keypoints.png")
                        , bbox_inches='tight', pad_inches=-0.1)
            plt.close()


# Usage example
input_directory = '../dataset/coco/train2017/'
annotations_file = '../dataset/annotations/person_keypoints_train2017.json'  # json of images
output_directory = 'coco_pose/gt_prompts'  # Output images

input_directory_test = '../dataset/coco/val2017/'
annotations_file_test = '../dataset/annotations/person_keypoints_val2017.json'  # json of images
output_directory_test = 'coco_pose/gt_prompts_test'  # Output images

visualize_keypoints(input_directory, annotations_file, output_directory)
visualize_keypoints(input_directory_test, annotations_file_test, output_directory_test)
