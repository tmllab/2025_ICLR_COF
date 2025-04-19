"""
This is the script for inferencing and evaluating the COF prompting method.
Please modify the file, if you want to evaluate the 300M model
Both 1b and 300m models are from the paper 'Data-efficient large vision models through sequential autoregression' 
"""
# import packages
import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, GenerationConfig

from tools.model_hf.muse import VQGANModel
from tools.utils import convert_decode_to_pil, encode_transform, codebook_similarity

WHITE = 1  # Define foreground
BLACK = 0  # Define background


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


if __name__ == '__main__':
    # --------- 1. Prepare parameters ---------
    prompt_path = '../prompting/coco_pose/cof_prompts/'  # path to prompts, the prompts are generated from saliency score map.
    reference_sample_path = '../dataset/coco/train2017/'  # path of origin referenced images
    test_sample_path = '../dataset/coco/val2017/'  # path of origin test images
    save_path = '../Results/COF/Pose_1B'  # Save results
    lvm_path = 'weights/llama_1b_hf'  # path to converted hf model
    vqgan_path = 'weights/vqgan-f16-8192-laion'  # path to vqgan model

    val_batch_size = 32
    k = 1
    n_cof = 2
    diversity = True
    # The reconstructed image size is 256 * 256
    recons_img_size = 256

    model = AutoModel.from_pretrained(lvm_path, trust_remote_code=True).cuda().eval()
    vq_model = VQGANModel.from_pretrained(vqgan_path).cuda().eval()

    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        num_beams=1,
        do_sample=True,
        max_new_tokens=256,
    )

    # --------- 2. Load Codebooks ---------
    codebooks = []
    img_ids = []

    ref_data = Test_Input(reference_sample_path, transform=encode_transform, validation=prompt_path)
    ref_data_loader = DataLoader(ref_data, batch_size=val_batch_size, shuffle=False)

    print("Fetching Codebook")

    for i, (img, img_name) in enumerate(ref_data_loader):
        print("processing image:", img_name)

        if torch.cuda.is_available():
            img = img.cuda()
        quantized_states, indices = vq_model.encode(img)
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
        print("processing_id:{0}, img_name:{1}".format(processing_id, img_name))
        if torch.cuda.is_available():
            img = img.cuda()
        Test_imgIds = int(img_name[0].split('.')[0])
        print("Test_imgIds", Test_imgIds)
        quantized_states, indices = vq_model.encode(img)
        input_ids = indices.reshape(1, -1)
        test_input_ids = indices.reshape(1, -1)
        input_ids_np = input_ids.detach().cpu().numpy()
        # --------- 4. Find closest Prompt --------- (QR)
        closest_prompts = np.argsort(codebook_similarity(codebooks,input_ids_np))[::-1][:k].tolist()

        prompt_ids_tmp = [img_ids[i] for i in closest_prompts]

        # --------- 5. Prompts with diversity selection --------- (AD)
        if diversity:
            diversity_scores = []
            for prompt_id in prompt_ids_tmp:
                p_name, p_ext = os.path.splitext(prompt_id)
                prompts_id = p_name + '_' + str(n_cof-1) + p_ext
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
                cur_img_path = os.path.join(reference_sample_path, prompt_id)
                full_prompts_img_path.append(cur_img_path)
                # Append prompts
                prompts_id = p_name+'_'+str(j)+p_ext
                prompts_img_path = os.path.join(prompt_path, prompts_id)
                full_prompts_img_path.append(prompts_img_path)

        seq_prompt, names = [], []
        for img_path in full_prompts_img_path:

            image = Image.open(img_path).convert('RGB')
            image = encode_transform(image)
            image = image[0:3, :, :].unsqueeze(0)
            seq_prompt.append(image)

        seq_ids = []
        for images in seq_prompt:
            images = images.cuda()

            # tokenize
            quantized_states, indices = vq_model.encode(images)
            prompt_ids = indices.reshape(1, -1)
            seq_ids.append(prompt_ids)

        seq_ids = torch.cat(seq_ids, dim=1)

        input_ids = torch.cat([seq_ids, input_ids], dim=1)

        # generate
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids,
                                        generation_config=generation_config,
                                        max_new_tokens=256,
                                        return_dict_in_generate=True,
                                        output_scores=True)

        # visulization
        generated_tokens = vq_model.quantize.get_codebook_entry_for_lvm(outputs.sequences[:, -256:])
        generated_img = vq_model.decode(
            generated_tokens.view(1, generated_tokens.shape[1] // 16, 16, -1).permute(0, 3, 1, 2))
        generated_img_rec = convert_decode_to_pil(generated_img)[0]
        save_name = 'result_' + img_name[0]
        generated_img_rec.save(os.path.join(save_path, save_name))

        print("Proceeding to {}th test sample, img_name:{}".format(str(processing_id), str(Test_imgIds)))
        processing_id += 1
        torch.cuda.empty_cache()

