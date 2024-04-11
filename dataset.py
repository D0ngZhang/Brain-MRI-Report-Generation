import os
from torch.utils.data import Dataset as dataset_torch
import numpy as np
import random
import torch
import torchvision.transforms as transforms

import torchio as tio

from torch.utils.data import DataLoader
from transformers import BertTokenizer

mean = np.array([0.5, ])
std = np.array([0.5, ])


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(1):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)

def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


def _make_dataset(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            # if is_image_file(fname):
            path = os.path.join(root, fname)
            item = path
            images.append(item)
    return images


def _make_image_namelist(imgs_dir, text_dir):
    images = []
    namelist = []

    reports = []


    for root, _, fnames in sorted(os.walk(imgs_dir)):
        for fname in fnames:
            if fname.endswith('_T1.nii'):
                item_name = fname
                img_path = os.path.join(root, fname)
                text_path = os.path.join(text_dir, item_name.replace('_T1.nii', '.txt'))
                if os.path.exists(img_path.replace('.nii', '_seg.nii')) and os.path.exists(text_path):
                    if os.path.exists(img_path.replace('_T1.nii', '_T2.nii')) and os.path.exists(img_path.replace('_T1.nii', '_T2_seg.nii')):
                        namelist.append(item_name[:-7])
                        images.append(img_path)
                        reports.append(text_path)

    return images, reports, namelist


def preprocessing(img, mask=None):
    if mask is not None:
        img_temp = img.tensor
        mask = mask.tensor
        img_temp = img_temp * mask
        img = tio.ScalarImage(tensor=img_temp, affine=img.affine)

    # original_spacing = img.spacing
    new_spacing = (1, 1, 8)
    resample = tio.Resample(new_spacing)
    img = resample(img)

    l = min(img.shape[1], img.shape[2])
    target_size = (l, l, img.shape[3])
    center_crop_or_pad = tio.CropOrPad(target_shape=target_size)
    img = center_crop_or_pad(img)

    target_size = (256, 256, img.shape[3])
    resize_transform = tio.Resize(target_shape=target_size, image_interpolation='linear')
    img = resize_transform(img)

    target_size = (192, 192, 16)
    center_crop_or_pad = tio.CropOrPad(target_shape=target_size)
    img = center_crop_or_pad(img)

    # transforms_deformation_dict = {
    #     tio.RandomAffine(): 0.25,
    #     tio.RandomElasticDeformation(): 0.75,
    # }

    # deform_transform = tio.OneOf(transforms_deformation_dict)
    # img = deform_transform(img)
    rescale = tio.RescaleIntensity(out_min_max=(-1, 1))
    img = rescale(img)
    return img



class data_set(dataset_torch):
    def __init__(self, opt, saved=False):
        imgs_root = opt.img_path
        text_root = opt.report_path
        mode = opt.mode

        self.imgs_root = imgs_root
        self.text_root = text_root
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.prompt = opt.prompt
        self.imgs, self.reports, self.nlist = _make_image_namelist(os.path.join(self.imgs_root), os.path.join(self.text_root))
        self.saved = saved

        self.img_num = len(self.imgs)
        print("total training data num: ", self.img_num)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.prompt = opt.prompt
        self.tokenizer = init_tokenizer()
        print("tokenizer prepared")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path_img_t1 = self.imgs[index]
        path_text = self.reports[index]
        path_img_t2 = path_img_t1.replace('T1', 'T2')
        name = self.nlist[index]

        img_t1 = tio.ScalarImage(path_img_t1)
        # mask_t1 = tio.LabelMap(path_img_t1.replace('.nii', '_seg.nii'))
        img_t1 = preprocessing(img_t1)

        img_t2 = tio.ScalarImage(path_img_t2)
        # mask_t2 = tio.LabelMap(path_img_t2.replace('.nii', '_seg.nii'))
        img_t2 = preprocessing(img_t2)

        img_t1_data = img_t1.data
        img_t2_data = img_t2.data

        with open(path_text, 'r', encoding='utf-8') as file:
            text_data = file.read()

        # text_data_tokens = self.tokenizer(text_data, padding='longest', truncation=True, max_length=150, return_tensors="pt")

        # print(text_data_tokens.input_ids.shape)

        img_t1_data = torch.permute(img_t1_data, (0, 3, 1, 2))
        img_t2_data = torch.permute(img_t2_data, (0, 3, 1, 2))

        # tokens = self.tokenizer(text_data, padding="max_length", truncation=True, max_length=150)
        # input_id, types, masks = tokens['input_ids'], tokens['token_type_ids'], tokens['attention_mask']


        # data = {"t1": img_t1_data, "t2": img_t2_data, "report-id": np.array(input_id), "report-mask": np.array(masks), "affine": img_t1.affine}
        data = {"t1": img_t1_data, "t2": img_t2_data, #"report_tokens": text_data_tokens,
                "report": text_data, "affine": img_t1.affine}
        return data


if __name__ == "__main__":

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    LOCAL_PATH = "C:/Users/zhang/Desktop/chinese-roberta-wwm-ext-large"
    dataset = data_set(imgs_root= 'E:/data/301', text_root= 'E:/diagnosis_reports', bert_root = LOCAL_PATH)
    dataloader = DataLoader(dataset, batch_size=4, num_workers=1, shuffle=True)

    for i, imgs in enumerate(dataloader):
        print("i: ", i)
        img_t1 = imgs['t1']
        text = imgs['report']
        print("data shape", img_t1.shape)
        print("text shape", len(text))
        print("text 0", text[0])
