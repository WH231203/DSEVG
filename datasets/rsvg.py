import os
import re
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from PIL import Image
import util
from util.transforms import letterbox
import datasets.transforms_image as T
import torch.utils.data as data
import random
import torch

def filelist(root, file_type):
    return [os.path.join(directory_path, f) for directory_path, directory_name, files in os.walk(root) for f in files if f.endswith(file_type)]

class RSVGDataset(data.Dataset):
    def __init__(self, images_path, anno_path, imsize=800, transform= None, augment= False,
                 split='train', testmode=False):
        self.images = []
        self.images_path = images_path
        self.anno_path = anno_path
        self.imsize = imsize
        self.augment = augment
        self.transform = transform
        self.split = split
        self.testmode = testmode

        file = open('data/DIOR_RSVG/' + split + '.txt', "r").readlines()
        Index = [int(index.strip('\n')) for index in file]
        count = 0
        annotations = filelist(anno_path, '.xml')
        for anno_path in annotations:
            root = ET.parse(anno_path).getroot()
            for member in root.findall('object'):
                if count in Index:
                    imageFile = str(images_path) + '/' + root.find("./filename").text
                    box = np.array([int(member[2][0].text), int(member[2][1].text), int(member[2][2].text), int(member[2][3].text)],dtype=np.float32)
                    text = member[3].text
                    self.images.append((imageFile, box, text))
                count += 1

    def pull_item(self, idx):
        img_path, bbox, phrase = self.images[idx]
        bbox = np.array(bbox, dtype=int)
        img = Image.open(img_path).convert('RGB')

        return img, phrase, bbox, img_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, phrase, bbox, img_path  = self.pull_item(idx)
        # phrase = phrase.lower()
        caption = " ".join(phrase.lower().split())

        w, h = img.size
        mask = np.zeros_like(img)

        if self.testmode:
            img = np.array(img)
            img, mask, ratio, dw, dh = letterbox(img, mask, self.imsize)
            bbox[0], bbox[2] = bbox[0] * ratio + dw, bbox[2] * ratio + dw
            bbox[1], bbox[3] = bbox[1] * ratio + dh, bbox[3] * ratio + dh
        bbox = torch.tensor([bbox[0], bbox[1], bbox[2], bbox[3]]).to(torch.float)
        bbox = bbox.unsqueeze(0)

        target = {}
        target["dataset_name"] = "DIOR-RSVG"
        target["boxes"] = bbox
        target["labels"] = torch.tensor([1])
        if caption is not None:
            target["caption"] = caption
        target["valid"] = torch.tensor([1])
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        # Norm, to tensor
        if self.transform is not None:
            img, target = self.transform(img, target)

        if self.testmode:
            return img.unsqueeze(0), target, dw, dh, img_path, ratio
        else:
            return img.unsqueeze(0), target


def make_coco_transforms(image_set, cautious):

    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    scales = [480, 560, 640, 720, 800]

    max_size = 800
    if image_set == "train":
        return T.Compose(
            [T.RandomResize(scales, max_size=max_size),
            normalize]
        )

    else:
        return T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])


from pathlib import Path

def build(image_set, args):
    root = Path(args.rsvg_path)
    assert root.exists(), f'provided YTVOS path {root} does not exist'
    input_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    img_folder = root / "JPEGImages"
    ann_file = root / "Annotations"

    dataset = RSVGDataset(img_folder, ann_file, transform=make_coco_transforms(image_set, False), split=image_set, testmode=(image_set=='test'))
    return dataset


