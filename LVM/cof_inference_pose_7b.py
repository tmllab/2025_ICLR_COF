# import packages
import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from inference import LocalInferenceModel
from utils import encode_transform


WHITE = 1  # Define foreground
BLACK = 0  # Define background


def cal_metric(pred_arr, gt_arr):
    intersection = np.logical_and(pred_arr, gt_arr).sum()
    union = np.logical_or(pred_arr, gt_arr).sum()
    iou = intersection / union

    # Cal foreground acc
    foreground_acc = intersection / sum(gt_arr.flatten() == 1)

    # Cal image pixel-level acc, (correct_fore + correct_back) / all_pixel
    pixel_acc = (sum(np.logical_and((pred_arr.flatten() == 1), (gt_arr.flatten() == 1))) + sum(
        np.logical_and((pred_arr.flatten() == 0), (gt_arr.flatten() == 0)))) / (256 * 256)
    black_rate = sum(pred_arr.flatten() == 0) / (256 * 256)
    return iou, foreground_acc, pixel_acc, black_rate


class Test_Input(Dataset):
    def __init__(self, root, transform=None, validation = ''):
        self.root = root
        self.transform = transform
        imgs_path_tmp = glob.glob(os.path.join(root, '*.jpg'))

        # Filter out bad samples
        if validation:
            valid_imgs_path = []
            for img_path in imgs_path_tmp:
                img_name = os.path.basename(img_path)
                p_name, p_ext = os.path.splitext(img_name)

                img_prompts_name = p_name + '_' + str(0) + p_ext
                prompts_img_path = os.path.join(validation, img_prompts_name)

                if os.path.exists(prompts_img_path):
                    valid_imgs_path.append(img_path)

            self.imgs_path = valid_imgs_path
        else:
            self.imgs_path = imgs_path_tmp

    def __getitem__(self, index):
        path = self.imgs_path[index]
        img_name = os.path.basename(path)

        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, img_name

    def __len__(self):
        return len(self.imgs_path)


def jaccard_similarity(l1, l2):
    set1 = set(l1)
    set2 = set(l2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


def codebook_similarity(codebooks, input_ids):
    similarities = []

    for codebook in codebooks:
        sim = jaccard_similarity(list(codebook), list(input_ids.reshape(-1)))
        similarities.append(sim)
    return similarities


def generate_images(input_images, n_new_frames, n_candidates, temperature=1.0, top_p=0.9):
    assert len(input_images) > 0
    input_images = [
        np.array(img.convert('RGB').resize((256, 256)), dtype=np.float32) / 255.0
        for img in input_images
    ]
    input_images = np.stack(input_images, axis=0)
    output_images = model([input_images], n_new_frames, n_candidates, temperature, top_p)[0]

    generated_images = []
    for candidate in output_images:
        concatenated_image = []
        for i, img in enumerate(candidate):
            concatenated_image.append(img)
            # if i < len(candidate) - 1:
            #     concatenated_image.append(checkerboard)
        generated_images.append(
            Image.fromarray(
                (np.concatenate(concatenated_image, axis=1) * 255).astype(np.uint8)
            )
        )

    return generated_images


if __name__ == '__main__':
    # --------- 1. Prepare parameters ---------
    prompt_path = '../prompting/coco_pose/cof_prompts/'  # path to prompt

    validation_sample_path = '../dataset/coco/train2017/'  # path of origin referenced/validation images
    test_sample_path = '../dataset/coco/val2017/'  # path of origin test images
    save_path = '../Results/COF/Pose_7B' # Save results
    lvm_path = "weights/lvm"  # path to converted hf model

    # Model hyper-parameters
    gen_length = 1
    n_candidates = 1  # Number of generated predition results, for time-effiency, we only generate once as result.
    temperature = 1.0
    top_p = 1.0

    diversity = True

    val_batch_size = 4
    k = 1
    n_cof = 2
    # The reconstructed image size is 256 * 256
    recons_img_size = 256
    print("Load model fail")

    model = LocalInferenceModel(
        checkpoint=lvm_path,
        torch_device=torch.device("cuda"),
        dtype='float16',
        context_frames=16,
        use_lock=False,
    )
    vq_model = model.tokenizer


    # --------- 2. Load Codebooks ---------
    codebooks = []
    img_ids = []

    val_data = Test_Input(validation_sample_path, transform=encode_transform, validation=prompt_path)
    val_data_loader = DataLoader(val_data, batch_size=val_batch_size, shuffle=False)

    print("Fetching Codebook")

    for i, (img, img_name) in enumerate(val_data_loader):
        print("processing image:", img_name)

        if torch.cuda.is_available():
            img = img.cuda()
        quantized_states, indices = vq_model.encode(img)
        # if i == 0:
        #     codebooks.extend(input_ids)
        #     codebooks = np.array(codebooks)
        # else:
        #     codebooks = np.vstack((codebooks,input_ids))
        n_tmp = len(list(img_name))
        input_ids = indices.reshape(n_tmp, -1).detach().cpu().numpy()
        codebooks.extend(input_ids)
        img_ids.extend(list(img_name))

    codebooks = np.array(codebooks)
    print("Codebook Shape: ")
    print(codebooks.shape)

    # --------- 3. Prepare input ---------
    test_data = Test_Input(test_sample_path, transform=encode_transform)
    data_loader = DataLoader(test_data, batch_size=1)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    print('Start Inference')
    processing_id = 1
    for i, (img, img_name) in enumerate(data_loader):
        if torch.cuda.is_available():
            img = img.cuda()
        Test_imgIds = int(img_name[0].split('.')[0])
        print("Test_imgIds", Test_imgIds)
        quantized_states, indices = vq_model.encode(img)
        input_ids = indices.reshape(1, -1)
        test_input_ids = indices.reshape(1, -1)
        input_ids_np = input_ids.detach().cpu().numpy()
        # --------- 4. Find closest Prompt ---------
        closest_prompts = np.argsort(codebook_similarity(codebooks, input_ids_np))[::-1][:100].tolist()

        # # Diversity of Annotation
        prompt_ids_tmp = [img_ids[i] for i in closest_prompts]

        if diversity:
            diversity_scores = []
            for prompt_id in prompt_ids_tmp:
                p_name, p_ext = os.path.splitext(prompt_id)
                prompts_id = p_name + '_' + str(n_cof - 1) + p_ext
                prompts_img_path = os.path.join(prompt_path, prompts_id)

                annotation = Image.open(prompts_img_path).convert('RGB')
                annotation = encode_transform(annotation)
                annotation = annotation[0:3, :, :].unsqueeze(0)
                annotation = annotation.cuda()
                quantized_states, indices = vq_model.encode(annotation)
                annotation_ids = indices.reshape(1, -1).detach().cpu().numpy()
                diversity = len(np.unique(annotation_ids))

                diversity_scores.append(diversity)

            # Select most diverse example
            top_k = np.argsort(diversity_scores)[::-1][:k]

            prompt_ids = [prompt_ids_tmp[i] for i in top_k]
        else:
            prompt_ids = prompt_ids_tmp[:k]

        full_prompts_img_path = []

        for prompt_id in prompt_ids:
            p_name, p_ext = os.path.splitext(prompt_id)

            for j in range(n_cof):
                # Append img
                cur_img_path = os.path.join(validation_sample_path, prompt_id)
                full_prompts_img_path.append(cur_img_path)
                # Append prompts
                prompts_id = p_name + '_' + str(j) + p_ext
                prompts_img_path = os.path.join(prompt_path, prompts_id)
                full_prompts_img_path.append(prompts_img_path)

        seq_prompt = []
        for img_path in full_prompts_img_path:

            image = Image.open(img_path)
            seq_prompt.append(image)
        test_img = Image.open(test_sample_path + img_name[0])
        seq_prompt.append(test_img)

        # generate
        with torch.no_grad():
            generated_img_rec = generate_images(
                seq_prompt,
                gen_length,
                n_candidates=n_candidates,
                temperature=temperature,
                top_p=top_p,)[0]

        # visulization
        save_name = 'result_' + img_name[0]
        generated_img_rec.save(os.path.join(save_path, save_name))

        processing_id += 1
