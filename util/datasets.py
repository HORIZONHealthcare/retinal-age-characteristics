# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os
from torchvision import transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


# def build_meta_dict():
#     csv_path = '/SAN/ioo/Alzeye10/yiqunlin/ukb/data_processing/cleaned_meta.csv'
#     df = pd.read_csv(csv_path)
#     disease_cols = [
#         'af', 'hf', 'mi', 'ich', 'istroke', 'as', 'ar', 'ms', 'mr', 
#         'dm', 'htn', 'dementia', 'parkinsons', 'glaucoma', 'cataract', 'dr'
#     ]
    
#     df['patient_condition'] = -1
#     mask_unhealthy = (df[disease_cols] == 1).any(axis=1)
#     mask_healthy = (df[disease_cols] == 0).all(axis=1)
    
#     df.loc[mask_healthy, 'patient_condition'] = 0
#     df.loc[mask_unhealthy, 'patient_condition'] = 1
    
#     path_to_label = pd.Series(
#         df.patient_condition.values, 
#         index=df.cfp_path
#     ).to_dict()
    
#     return path_to_label


def build_dataset(split, args):
    transform = build_transform(split == 'train', args)
    if args.dataset_name == 'alzeye':
        # csv_root = '/SAN/ioo/Alzeye10/yiqunlin/ukb/data_processing/splits/'
        # csv_root = '/SAN/ioo/Alzeye10/yiqunlin/alzeye25/data_processing/splits_20260112_195512/'
        csv_root = '/SAN/ioo/Alzeye10/yiqunlin/alzeye25/data_processing/splits_20260114_000313'
        if split == 'train':
            pacond = args.patient_condition
            split = f'{split}_{pacond}'
        csv_path = os.path.join(csv_root, f'{split}.csv')
        # dataset = RETFound_loader(args.modalities, csv_path, transform, args.patient_condition)
        dataset = RETFound_loader(args.modalities, csv_path, transform)
    
    ## external datasets
    elif args.dataset_name == 'mbrset':
        csv_path = '/SAN/ioo/Alzeye10/yukunzhou/public_dataset/external/mbrset/1.0/labels_mbrset.csv'
        dataset = mbrset(args.modalities, csv_path, transform)
    elif args.dataset_name == 'brset':
        csv_path = '/SAN/ioo/Alzeye10/yukunzhou/public_dataset/external/brazilian-ophthalmological/1.0.1/labels_brset.csv'
        dataset = brset(args.modalities, csv_path, transform)
    elif args.dataset_name == 'ukb':
        csv_path = '/SAN/ioo/Alzeye10/yiqunlin/ukb/data_processing/splits'
        dataset = ukb(args.modalities, csv_path, transform)
    else:
        raise ValueError(f'Invalid dataset name: {args.dataset_name}')
    return dataset


class ukb(Dataset):
    def __init__(self, modalities, csv_path, transform):
        self.csv_path = csv_path
        print(self.csv_path)
        # df = pd.read_csv(self.csv_path)[['patient_id', 'age_at_scan', 'cfp_path']]

        df = pd.concat(
            [pd.read_csv(os.path.join(csv_path, f'{split}.csv')) for split in ['train', 'val', 'test']],
            ignore_index=True
        )[['patient_id', 'age', 'cfp_path']]
        self.modalities = modalities

        self.samples = []
        n_total = 0
        for row in df.values.tolist():
            n_total += 1
            pid, age, cfp = row
            age = int(age)
            self.samples.append({
                'patient_id': cfp,
                'age': age,
                'paths': {
                    'CFP': cfp,
                }
            })
        self.transform = transform
        print(len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        age = sample['age']
        patient_id = sample['patient_id']
        image_list = []
        for modality in self.modalities:
            path = sample['paths'][modality]

            image = Image.open(path)
            image = image.convert("RGB")
            
            if self.transform is not None:
                image_processed = self.transform(image)
            
            image_list.append(image_processed)
        
        image_processed = torch.stack(image_list, dim=0)
        age = torch.tensor(age).type(torch.float)
        age = normalize_age(age)
        return patient_id, image_processed.type(torch.FloatTensor), age


class brset(Dataset):
    def __init__(self, modalities,csv_path, transform):
        self.csv_path = csv_path
        print(self.csv_path)
        df = pd.read_csv(self.csv_path)[['patient_age', 'image_id']]
        self.modalities = modalities

        data_dir = '/SAN/ioo/Alzeye10/yukunzhou/public_dataset/external/brazilian-ophthalmological/1.0.1/images'
        self.samples = []
        for row in df.values.tolist():
            age, name = row
            cfp = os.path.join(data_dir, f'{name}.jpg')
            if not os.path.exists(cfp):
                print(f'skip {name}, cfp: {cfp}')
                continue
            try:
                age = int(age)
            except (ValueError, TypeError):
                print(f'skip {name}, age: {age}')
                continue
            self.samples.append({
                'name': name,
                'age': age,
                'paths': {
                    'CFP': cfp
                }
            }) 

        self.transform = transform
        print(len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        index = index % len(self.samples)
        sample = self.samples[index]
        age = sample['age']
        data_name = sample['name']
        image_list = []
        for modality in self.modalities:
            path = sample['paths'][modality]
            image = Image.open(path)
            image = image.convert("RGB")
            if self.transform is not None:
                image_processed = self.transform(image)
            image_list.append(image_processed)
        
        image_processed = torch.stack(image_list, dim=0)
        age = torch.tensor(age).type(torch.float)
        age = normalize_age(age)
        return data_name, image_processed.type(torch.FloatTensor), age


class mbrset(Dataset):
    def __init__(self, modalities,csv_path, transform):
        self.csv_path = csv_path
        print(self.csv_path)
        df = pd.read_csv(self.csv_path)[['age', 'file']]
        self.modalities = modalities

        data_dir = '/SAN/ioo/Alzeye10/yukunzhou/public_dataset/external/mbrset/1.0/images'

        self.samples = []
        for row in df.values.tolist():
            age, name = row
            cfp = os.path.join(data_dir, name)
            if not os.path.exists(cfp):
                print(f'skip {name}, cfp: {cfp}')
                continue
            try:
                age = int(age)
            except (ValueError, TypeError):
                print(f'skip {name}, age: {age}')
                continue
            self.samples.append({
                'name': name,
                'age': age,
                'paths': {
                    'CFP': cfp
                }
            })

        self.transform = transform
        print(len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        index = index % len(self.samples)
        sample = self.samples[index]
        age = sample['age']
        data_name = sample['name']
        image_list = []
        for modality in self.modalities:
            path = sample['paths'][modality]
            image = Image.open(path)
            image = image.convert("RGB")
            if self.transform is not None:
                image_processed = self.transform(image)
            image_list.append(image_processed)
        
        image_processed = torch.stack(image_list, dim=0)
        age = torch.tensor(age).type(torch.float)
        age = normalize_age(age)
        return data_name, image_processed.type(torch.FloatTensor), age


class RETFound_loader(Dataset):
    # def __init__(self, modalities, csv_path, transform, patient_condition='mixed'):
    def __init__(self, modalities, csv_path, transform):
        self.csv_path = csv_path
        print(self.csv_path)
        # df = pd.read_csv(self.csv_path)[['age', 'cfp_path', 'fa_path', 'oct_path']]
        # df = pd.read_csv(self.csv_path)[['age_at_scan', 'cfp_path', 'oct_path']]
        df = pd.read_csv(self.csv_path)[['ageAtTimeOfExam', 'cs_save_path_image_new']]
        self.modalities = modalities

        # if patient_condition != 'mixed':
        #     pa_cond = build_meta_dict()

        self.samples = []
        n_total = 0
        for row in df.values.tolist():
            n_total += 1
            # age, cfp, oct = row
            age, cfp = row
            age = int(age)
            # age, cfp, fa, oct = row
            # 0: healthy, 1: unhealthy, -1: uncertain
            # if patient_condition == 'healthy':
            #     if pa_cond[cfp] != 0:
            #         continue
            # elif patient_condition == 'unhealthy':
            #     if pa_cond[cfp] != 1:
            #         continue
            self.samples.append({
                'age': age,
                'paths': {
                    'CFP': cfp,
                    # 'FA': fa,
                    # 'OCT': oct
                }
            })

        # print(f"{patient_condition}: {len(self.samples)} / {n_total}")
        self.transform = transform
        self.repeat = 1
        # if 'train' in self.csv_path:
        #     self.repeat = 5

    def __len__(self):
        return len(self.samples) * self.repeat

    def __getitem__(self, index):
        index = index % len(self.samples)
        sample = self.samples[index]
        age = sample['age']
        image_list = []
        for modality in self.modalities:
            path = sample['paths'][modality]
            img_name = path

            image = Image.open(path)
            image = image.convert("RGB")
            
            if self.transform is not None:
                image_processed = self.transform(image)
            
            image_list.append(image_processed)
        
        image_processed = torch.stack(image_list, dim=0)
        age = torch.tensor(age).type(torch.float)
        age = normalize_age(age)
        return img_name, image_processed.type(torch.FloatTensor), age


AGE_MEAN = 50
AGE_STD = 50

def normalize_age(age):
    return (age - AGE_MEAN) / AGE_STD

def denormalize_age(age):
    return age * AGE_STD + AGE_MEAN

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), 
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
    
