import json
import numpy as np
import cv2
import glob
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
import os
import heapq
import torch
import torch.multiprocessing
from torchvision import transforms
from torch.utils.data import DataLoader
from U_2_Net.model.u2net import U2NET, U2NETP
from U_2_Net.data_loader import RescaleT, ToTensorLab
from generate_gt_seg import CocoDetection, convert_coco_poly_to_mask

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


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn


def saliency_prob(d):
    return d/torch.sum(d)


def save_vis_saliency_map(image_name,pred,d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')
    im.close()


def get_prob_map(image_name, prob):

    prob = prob.squeeze()
    prob_np = prob.cpu().data.numpy()

    im = Image.fromarray(prob_np)
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    return np.array(imo)


def visualize_keypoints(img, anns, save_path, j):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    width, height, img_name = img.shape[1], img.shape[0], str(anns[0]["image_id"].item()).zfill(12)
    X = np.zeros((height, width, 3), dtype=np.uint8)

    valid_objects = 0
    for ann in anns:
        keypoints_tmp = convert_coco_to_openpose_cords(np.array(ann['keypoints']).flatten().tolist()).flatten()

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

        plt.savefig(os.path.join(save_path, img_name + '_' + str(j) + ".jpg")
                    , bbox_inches='tight', pad_inches=-0.1)
        plt.close()


def convert_lists_to_ndarrays(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, list):
            dictionary[key] = np.array(value)
        elif isinstance(value, dict):
            dictionary[key] = convert_lists_to_ndarrays(value)
    return dictionary


def convert_tensor_list_to_float_list(nested_list):
    float_list = []
    for sublist in nested_list:
        float_sublist = [float(tensor.item()) for tensor in sublist]
        float_list.append(float_sublist)
    return float_list


def convert_tensor_dict_to_normal(dict_with_tensors):
    normal_dict = {}
    for key, value in dict_with_tensors.items():
        if isinstance(value, list):
            normal_list = []
            for tensor in value:
                normal_list.append(tensor.item())
            normal_dict[key] = normal_list
        else:
            normal_dict[key] = value.item()
    return normal_dict


def indexes_of_largest_k(lst, k):
    if k <= 0:
        return []
    # Use a max heap in terms of values but store them as negative to use Python's min-heap
    max_heap = [(-lst[i], i) for i in range(len(lst))]
    heapq.heapify(max_heap)

    # Extract the largest k elements based on their negative value (which corresponds to the actual largest values)
    largest_k_indexes = heapq.nsmallest(k, max_heap)
    # Sort indexes based on their value in descending order
    largest_k_indexes_sorted = sorted(largest_k_indexes, key=lambda x: x[0], reverse=True)

    # Extract and return the indexes
    return [idx for (_, idx) in largest_k_indexes_sorted]


def elements_count_in_divided_sublists(lst, n):
    length = len(lst)
    base_size, extras = divmod(length, n)
    counts = [base_size + 1 if i < extras else base_size for i in range(n)]
    return counts


def main():

    # --------- 1. get image & model path ---------
    saliency_model_name = 'u2netp'  # u2netp

    # Save gt image path
    save_path = "coco_pose/cof_prompts/"
    # GT Annotation path
    anno_file_path = "../dataset/annotations/person_keypoints_train2017.json"
    # GT image path
    img_file_path = '../dataset/coco/train2017/'

    saliency_res_dir = os.path.join(os.getcwd(), 'coco_pose','saliency_results', 'visualizations' + os.sep)
    saliency_model_dir = os.path.join(os.getcwd(), 'U_2_Net', 'saved_models',
                                      saliency_model_name, saliency_model_name + '.pth')

    img_name_list = glob.glob(img_file_path + os.sep + '*')
    print(img_name_list)

    # --------- 2. prepare data ---------
    datasets = CocoDetection(img_file_path, anno_file_path, transform=transforms.Compose([RescaleT(320),
                                                                                          ToTensorLab(flag=0)]), pose_num=True)
    dataloader = DataLoader(datasets, batch_size=1, shuffle=False, num_workers=1)

    # --------- 3. load saliency model ---------
    if (saliency_model_name == 'u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)
    elif (saliency_model_name == 'u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3, 1)
    else:
        raise Exception('Undefined model')

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(saliency_model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(saliency_model_dir, map_location='cpu'))
    net.eval()

    map_dict = {}

    for i, (imgs, img_np, target, id) in enumerate(dataloader):

        print("inferencing:", img_name_list[i].split(os.sep)[-1])

        imgs = imgs.type(torch.FloatTensor)

        if torch.cuda.is_available():
            imgs = imgs.cuda()

        d1, d2, d3, d4, d5, d6, d7 = net(imgs)

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)
        prob = saliency_prob(pred)

        # save saliency visualisation
        if not os.path.exists(saliency_res_dir):
            os.makedirs(saliency_res_dir, exist_ok=True)
        # save_vis_saliency_map(img_name_list[i], pred, saliency_res_dir)

        del d1, d2, d3, d4, d5, d6, d7

        prob_map = get_prob_map(img_name_list[i], prob)

        id = str(id.item())

        map_dict[id] = prob_map

        print('Processing... Image ID: ' + id)
        img_prob_map = np.array(map_dict[id])

        if len(target) == 0:
            continue

        width, height, filename = img_prob_map.shape[1], img_prob_map.shape[0], str(target[0]["image_id"]).zfill(12)
        sorted_anns = sorted(target, key=(lambda x: x['area']), reverse=True)

        saliency_scores = []

        for ann in sorted_anns:
            # Polygons to mask
            if type(ann['segmentation']) == list:
                ann_segmentation = convert_tensor_list_to_float_list(ann['segmentation'])
            else:
                ann_segmentation = convert_tensor_dict_to_normal(ann['segmentation'])
            mask = convert_coco_poly_to_mask(ann_segmentation, height, width)
            mask_np = mask.squeeze(dim=0).detach().cpu().numpy()
            object_saliency_score = np.sum(img_prob_map * mask_np)
            saliency_scores.append(object_saliency_score)
            del mask

        # --------- 6. Prompt Generation ---------

        n = 2

        num_ann = elements_count_in_divided_sublists(saliency_scores, n)

        for i in range(n):
            ann_ids = indexes_of_largest_k(saliency_scores, sum(num_ann[:i + 1]))
            selected_ann = [sorted_anns[i] for i in ann_ids]

            visualize_keypoints(img_prob_map,selected_ann,save_path,i)


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()
