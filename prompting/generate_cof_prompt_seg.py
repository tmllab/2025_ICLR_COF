import os
import glob
import json
import torch
import heapq
import numpy as np
from PIL import Image
from skimage import io
import torch.multiprocessing
from torchvision import transforms
from torch.utils.data import DataLoader
from U_2_Net.model.u2net import U2NET, U2NETP
from U_2_Net.data_loader import RescaleT, ToTensorLab
from generate_gt_seg import CocoDetection, convert_coco_poly_to_mask


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


# Add colorful segmentation mask on the image.
def convert_anns_to_mask_seg(img, anns):
    # Process on cpu
    device = "cuda"
    img = Image.fromarray(img).convert('RGB')
    width, height, filename = img.size[0], img.size[1], str(anns[0]["image_id"].item()).zfill(12)
    if len(anns) == 0:
        return

    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    mask_img = torch.zeros((height, width, 3), device=device)
    one_img = torch.ones((height, width, 3), device=device)

    for ann in sorted_anns:
        # Polygons to mask
        if type(ann['segmentation']) == list:
            ann_segmentation = convert_tensor_list_to_float_list(ann['segmentation'])
        else:
            ann_segmentation = convert_tensor_dict_to_normal(ann['segmentation'])

        mask = convert_coco_poly_to_mask(ann_segmentation, height, width)
        mask = torch.tensor(mask, device=device)
        mask = torch.repeat_interleave(mask.squeeze(dim=0).unsqueeze(dim=2), repeats=3, dim=2)
        color_mask = torch.rand(3, device=device) * one_img
        mask_img += color_mask * mask

        del mask, color_mask
    mask_img_npy = mask_img.cpu().numpy()
    mask_img_npy = (255 * mask_img_npy).astype(np.uint8)

    del mask_img, one_img

    return mask_img_npy, filename


def save_dict_to_json(dictionary, file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    dictionary = convert_ndarrays_to_lists(dictionary)
    with open(file_path, 'w') as json_file:
        json.dump(dictionary, json_file, indent=4)


def convert_ndarrays_to_lists(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, np.ndarray):
            dictionary[key] = value.tolist()
        elif isinstance(value, dict):
            dictionary[key] = convert_ndarrays_to_lists(value)
    return dictionary


def read_dict_from_json(file_path):
    with open(file_path, 'r') as json_file:
        dictionary = json.load(json_file)
    # Convert lists to NumPy ndarrays
    dictionary = convert_lists_to_ndarrays(dictionary)
    return dictionary


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
    save_path = "coco_seg/cof_prompts/"
    # GT Annotation path
    anno_file_path = "../dataset/annotations/instances_train2017.json"
    # GT image path
    img_file_path = '../dataset/coco/train2017/'

    saliency_res_dir = os.path.join(os.getcwd(), 'coco_seg','saliency_results', 'visualizations' + os.sep)
    saliency_model_dir = os.path.join(os.getcwd(), 'U_2_Net', 'saved_models',
                                      saliency_model_name, saliency_model_name + '.pth')

    img_name_list = glob.glob(img_file_path + os.sep + '*')

    # --------- 2. prepare data ---------
    datasets = CocoDetection(img_file_path, anno_file_path, transform=transforms.Compose([RescaleT(320),
                                                                                          ToTensorLab(flag=0)]))
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

    # --------- 4. get saliency probability map ---------
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
        # You can visualize the attention map.
        # save_vis_saliency_map(img_name_list[i], pred, saliency_res_dir)

        del d1, d2, d3, d4, d5, d6, d7

        prob_map = get_prob_map(img_name_list[i], prob)

        id = str(id.item())

        map_dict[id] = prob_map

        img_np = img_np.squeeze().cpu().detach().numpy()
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

    # --------- 5. Prompt Generation ---------

        n = 2

        num_ann = elements_count_in_divided_sublists(saliency_scores,n)

        for i in range(n):
            ann_ids = indexes_of_largest_k(saliency_scores, sum(num_ann[:i+1]))
            selected_ann = [sorted_anns[i] for i in ann_ids]

            color_masked_img, img_name = convert_anns_to_mask_seg(img_np, selected_ann)
            im = Image.fromarray(color_masked_img)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            im.save(os.path.join(save_path, img_name + '_' + str(i) + ".jpg"))


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()
