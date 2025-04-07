import os
import pandas as pd
import numpy as np
import warnings
import random
from glob import glob

from typing import Any, Optional, List

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import rasterio
from rasterio import logging

import xml.etree.ElementTree as ET
import torchvision.transforms as T
from torchvision.transforms import functional as F
from torch.utils.data import Dataset
log = logging.getLogger()
log.setLevel(logging.ERROR)

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)


CATEGORIES = ["airport", "airport_hangar", "airport_terminal", "amusement_park",
              "aquaculture", "archaeological_site", "barn", "border_checkpoint",
              "burial_site", "car_dealership", "construction_site", "crop_field",
              "dam", "debris_or_rubble", "educational_institution", "electric_substation",
              "factory_or_powerplant", "fire_station", "flooded_road", "fountain",
              "gas_station", "golf_course", "ground_transportation_station", "helipad",
              "hospital", "impoverished_settlement", "interchange", "lake_or_pond",
              "lighthouse", "military_facility", "multi-unit_residential",
              "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park",
              "parking_lot_or_garage", "place_of_worship", "police_station", "port",
              "prison", "race_track", "railway_bridge", "recreational_facility",
              "road_bridge", "runway", "shipyard", "shopping_mall",
              "single-unit_residential", "smokestack", "solar_farm", "space_facility",
              "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth",
              "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility",
              "wind_farm", "zoo"]
class DIORDataset(Dataset):
    def __init__(self, root, split="trainval", transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        self.image_dir = os.path.join(root, f"JPEGImages-{split}")
        self.ann_dir = os.path.join(root, "Annotations", "Horizontal Bounding Boxes")

        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image folder not found: {self.image_dir}")
        if not os.path.exists(self.ann_dir):
            raise FileNotFoundError(f"Annotation folder not found: {self.ann_dir}")

        self.image_filenames = [f.split(".")[0] for f in os.listdir(self.image_dir) if f.endswith(".jpg")]
        self.annotation_filenames = [f.split(".")[0] for f in os.listdir(self.ann_dir) if f.endswith(".xml")]

        self.image_filenames = sorted(list(set(self.image_filenames) & set(self.annotation_filenames)))

        # 🔹 Define class mapping (Ensure it matches your dataset)
        self.CLASS_MAPPING = {
            "airplane": 0, "airport": 1, "baseballfield": 2, "basketballcourt": 3,
            "bridge": 4, "chimney": 5, "dam": 6, "Expressway-Service-area": 7,
            "Expressway-toll-station": 8, "golffield": 9, "groundtrackfield": 10,
            "harbor": 11, "overpass": 12, "ship": 13, "stadium": 14, "storagetank": 15,
            "tenniscourt": 16, "trainstation": 17, "vehicle": 18, "windmill": 19
        }

    def __getitem__(self, idx):
        image_id = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        xml_path = os.path.join(self.ann_dir, f"{image_id}.xml")

        image = Image.open(img_path).convert("RGB")

        target_data = self.parse_voc_xml(xml_path)
        boxes = target_data["boxes"]
        labels = target_data["labels"]

        if self.transform is not None:
            # Albumentations expects numpy image and Pascal VOC format
            transformed = self.transform(
                image=np.array(image),
                bboxes=boxes,
                class_labels=labels
            )
            image = transformed["image"]
            boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            labels = torch.tensor(transformed["class_labels"], dtype=torch.int64)
        else:
            # For Faster R-CNN — convert image and data manually
            image = F.to_tensor(image)
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        # Filter out invalid boxes (zero width/height)
        if boxes.shape[0] > 0:
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            keep = (widths > 0) & (heights > 0)
            boxes = boxes[keep]
            labels = labels[keep]
        if self.transform is not None:
            multi_hot = torch.zeros(21, dtype=torch.float32)
            for label in labels:
                multi_hot[label] = 1.0
            target = {
                "labels": multi_hot,
                "image_id": torch.tensor([idx])
            }

        else:
            target = {
                "boxes": boxes,
                "labels": labels,
                "image_id": torch.tensor([idx])
            }

        return image, target

    def parse_voc_xml(self, xml_path):
        """Parses a Pascal VOC XML file and extracts bounding box & label information."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        labels = []

        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])

            # 🔹 Get label as a string and convert it to an integer index
            label_str = obj.find("name").text
            if label_str in self.CLASS_MAPPING:
                labels.append(self.CLASS_MAPPING[label_str])
            else:
                raise ValueError(f"Unknown class label '{label_str}' in {xml_path}")

        return {"boxes": boxes, "labels": labels}

    def transform_image_and_boxes(self, image, boxes, orig_w, orig_h):
        """Resizes image & bounding boxes while maintaining aspect ratio."""
        # 🔹 Resize image
        transformed_image = F.resize(image, (224, 224))  # Ensure consistent resizing
        transformed_image = F.to_tensor(transformed_image)  # Convert to tensor

        # 🔹 Rescale bounding boxes
        scale_x = 224 / orig_w
        scale_y = 224 / orig_h
        transformed_boxes = torch.tensor(
            [[x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y] for x1, y1, x2, y2 in boxes],
            dtype=torch.float32
        )

        return transformed_image, transformed_boxes

    def __len__(self):
        return len(self.image_filenames)




class SatelliteDataset(Dataset):
    """
    Abstract class.
    """
    def __init__(self, in_c):
        self.in_c = in_c

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        """
        Builds train/eval data transforms for the dataset class.
        :param is_train: Whether to yield train or eval data transform/augmentation.
        :param input_size: Image input size (assumed square image).
        :param mean: Per-channel pixel mean value, shape (c,) for c channels
        :param std: Per-channel pixel std. value, shape (c,)
        :return: Torch data transform for the input image before passing to model
        """
        # mean = IMAGENET_DEFAULT_MEAN
        # std = IMAGENET_DEFAULT_STD

        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(mean, std))
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        t.append(
            transforms.Resize(size, interpolation=interpol_mode),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        # t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)


class CustomDatasetFromImages(SatelliteDataset):
    mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
    std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]

    def __init__(self, csv_path, transform):
        """
        Creates Dataset for regular RGB image classification (usually used for fMoW-RGB dataset).
        :param csv_path: csv_path (string): path to csv file.
        :param transform: pytorch transforms for transforms and tensor conversion.
        """
        super().__init__(in_c=3)
        # Transforms
        self.transforms = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len


class FMoWTemporalStacked(SatelliteDataset):
    mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
    std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]
    
    def __init__(self, csv_path: str, transform: Any):
        """
        Creates Dataset for temporal RGB image classification. Stacks images along temporal dim.
        Usually used for fMoW-RGB-temporal dataset.
        :param csv_path: path to csv file.
        :param transform: pytorch transforms for transforms and tensor conversion
        """
        super().__init__(in_c=9)
        # Transforms
        self.transforms = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # Calculate len
        self.data_len = len(self.data_info.index)

        self.min_year = 2002

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name_1 = self.image_arr[index]

        splt = single_image_name_1.rsplit('/', 1)
        base_path = splt[0]
        fname = splt[1]
        suffix = fname[-15:]
        prefix = fname[:-15].rsplit('_', 1)
        regexp = '{}/{}_*{}'.format(base_path, prefix[0], suffix)
        temporal_files = glob(regexp)
        temporal_files.remove(single_image_name_1)
        if temporal_files == []:
            single_image_name_2 = single_image_name_1
            single_image_name_3 = single_image_name_1
        elif len(temporal_files) == 1:
            single_image_name_2 = temporal_files[0]
            single_image_name_3 = temporal_files[0]
        else:
            single_image_name_2 = random.choice(temporal_files)
            while True:
                single_image_name_3 = random.choice(temporal_files)
                if single_image_name_3 != single_image_name_2:
                    break

        img_as_img_1 = Image.open(single_image_name_1)
        img_as_tensor_1 = self.transforms(img_as_img_1)  # (3, h, w)

        img_as_img_2 = Image.open(single_image_name_2)
        img_as_tensor_2 = self.transforms(img_as_img_2)  # (3, h, w)

        img_as_img_3 = Image.open(single_image_name_3)
        img_as_tensor_3 = self.transforms(img_as_img_3)  # (3, h, w)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        img = torch.cat((img_as_tensor_1, img_as_tensor_2, img_as_tensor_3), dim=0)  # (9, h, w)
        return (img, single_image_label)

    def __len__(self):
        return self.data_len


class CustomDatasetFromImagesTemporal(SatelliteDataset):
    def __init__(self, csv_path: str):
        """
        Creates temporal dataset for fMoW RGB
        :param csv_path: Path to csv file containing paths to images
        :param meta_csv_path: Path to csv metadata file for each image
        """
        super().__init__(in_c=3)

        # Transforms
        self.transforms = transforms.Compose([
            # transforms.Scale(224),
            transforms.RandomCrop(224),
        ])
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # Calculate len
        self.data_len = len(self.data_info)

        self.dataset_root_path = os.path.dirname(csv_path)

        self.timestamp_arr = np.asarray(self.data_info.iloc[:, 2])
        #print("DEBUG: self.image_arr =", self.image_arr[:10])  # Print first 10 elements

        self.name2index = dict(zip(
            [os.path.join(self.dataset_root_path, x) for x in self.image_arr],
            np.arange(self.data_len)
        ))

        self.min_year = 2002  # hard-coded for fMoW

        mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
        std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]
        self.normalization = transforms.Normalize(mean, std)
        self.totensor = transforms.ToTensor()
        self.scale = transforms.Resize(224)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name_1 = self.image_arr[index]

        suffix = single_image_name_1[-5:]
        prefix = single_image_name_1[:-5].rsplit('_', 1)
        regexp = '{}_*{}'.format(prefix[0], suffix)
        regexp = os.path.join(self.dataset_root_path, regexp)
        single_image_name_1 = os.path.join(self.dataset_root_path, single_image_name_1)

        temporal_files = glob(regexp)
        #print("DEBUG:",single_image_name_1 in temporal_files,single_image_name_1,temporal_files, suffix, prefix)

        if single_image_name_1 in temporal_files:
            temporal_files.remove(single_image_name_1)
        if temporal_files == []:
            single_image_name_2 = single_image_name_1
            single_image_name_3 = single_image_name_1
        elif len(temporal_files) == 1:
            single_image_name_2 = temporal_files[0]
            single_image_name_3 = temporal_files[0]
        else:
            single_image_name_2 = random.choice(temporal_files)
            while True:
                single_image_name_3 = random.choice(temporal_files)
                if single_image_name_3 != single_image_name_2:
                    break

        img_as_img_1 = Image.open(single_image_name_1)
        img_as_img_2 = Image.open(single_image_name_2)
        img_as_img_3 = Image.open(single_image_name_3)
        img_as_tensor_1 = self.totensor(img_as_img_1)
        img_as_tensor_2 = self.totensor(img_as_img_2)
        img_as_tensor_3 = self.totensor(img_as_img_3)
        del img_as_img_1
        del img_as_img_2
        del img_as_img_3
        img_as_tensor_1 = self.scale(img_as_tensor_1)
        img_as_tensor_2 = self.scale(img_as_tensor_2)
        img_as_tensor_3 = self.scale(img_as_tensor_3)
        try:
            if img_as_tensor_1.shape[2] > 224 and \
                    img_as_tensor_2.shape[2] > 224 and \
                    img_as_tensor_3.shape[2] > 224:
                min_w = min(img_as_tensor_1.shape[2], min(img_as_tensor_2.shape[2], img_as_tensor_3.shape[2]))
                img_as_tensor = torch.cat([
                    img_as_tensor_1[..., :min_w],
                    img_as_tensor_2[..., :min_w],
                    img_as_tensor_3[..., :min_w]
                ], dim=-3)
            elif img_as_tensor_1.shape[1] > 224 and \
                    img_as_tensor_2.shape[1] > 224 and \
                    img_as_tensor_3.shape[1] > 224:
                min_w = min(img_as_tensor_1.shape[1], min(img_as_tensor_2.shape[1], img_as_tensor_3.shape[1]))
                img_as_tensor = torch.cat([
                    img_as_tensor_1[..., :min_w, :],
                    img_as_tensor_2[..., :min_w, :],
                    img_as_tensor_3[..., :min_w, :]
                ], dim=-3)
            else:
                img_as_img_1 = Image.open(single_image_name_1)
                img_as_tensor_1 = self.totensor(img_as_img_1)
                img_as_tensor_1 = self.scale(img_as_tensor_1)
                img_as_tensor = torch.cat([img_as_tensor_1, img_as_tensor_1, img_as_tensor_1], dim=-3)
        except:
            print(img_as_tensor_1.shape, img_as_tensor_2.shape, img_as_tensor_3.shape)
            assert False

        del img_as_tensor_1
        del img_as_tensor_2
        del img_as_tensor_3

        img_as_tensor = self.transforms(img_as_tensor)
        img_as_tensor_1, img_as_tensor_2, img_as_tensor_3 = torch.chunk(img_as_tensor, 3, dim=-3)
        del img_as_tensor
        img_as_tensor_1 = self.normalization(img_as_tensor_1)
        img_as_tensor_2 = self.normalization(img_as_tensor_2)
        img_as_tensor_3 = self.normalization(img_as_tensor_3)

        ts1 = self.parse_timestamp(single_image_name_1)
        ts2 = self.parse_timestamp(single_image_name_2)
        ts3 = self.parse_timestamp(single_image_name_3)

        ts = np.stack([ts1, ts2, ts3], axis=0)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        imgs = torch.stack([img_as_tensor_1, img_as_tensor_2, img_as_tensor_3], dim=0)

        del img_as_tensor_1
        del img_as_tensor_2
        del img_as_tensor_3

        return (imgs, ts, single_image_label)

    def parse_timestamp(self, name):
        if name not in self.name2index or self.name2index[name] not in self.timestamp_arr:
            return np.array([0,0,0])
        timestamp = self.timestamp_arr[self.name2index[name]]
        year = int(timestamp[:4])
        month = int(timestamp[5:7])
        hour = int(timestamp[11:13])
        return np.array([year - self.min_year, month - 1, hour])

    def __len__(self):

        return self.data_len


#########################################################
# SENTINEL DEFINITIONS
#########################################################


class SentinelNormalize:
    """
    Normalization for Sentinel-2 imagery, inspired from
    https://github.com/ServiceNow/seasonal-contrast/blob/8285173ec205b64bc3e53b880344dd6c3f79fa7a/datasets/bigearthnet_dataset.py#L111
    """
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, x, *args, **kwargs):
        min_value = self.mean - 2 * self.std
        max_value = self.mean + 2 * self.std
        img = (x - min_value) / (max_value - min_value) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


class SentinelIndividualImageDataset(SatelliteDataset):
    label_types = ['value', 'one-hot']
    mean = [1370.19151926, 1184.3824625 , 1120.77120066, 1136.26026392,
            1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
            1972.62420416,  582.72633433,   14.77112979, 1732.16362238, 1247.91870117]
    std = [633.15169573,  650.2842772 ,  712.12507725,  965.23119807,
           948.9819932 , 1108.06650639, 1258.36394548, 1233.1492281 ,
           1364.38688993,  472.37967789,   14.3114637 , 1310.36996126, 1087.6020813]

    def __init__(self,
                 csv_path: str,
                 transform: Any,
                 years: Optional[List[int]] = [*range(2000, 2021)],
                 categories: Optional[List[str]] = None,
                 label_type: str = 'value',
                 masked_bands: Optional[List[int]] = None,
                 dropped_bands: Optional[List[int]] = None):
        """
        Creates dataset for multi-spectral single image classification.
        Usually used for fMoW-Sentinel dataset.
        :param csv_path: path to csv file.
        :param transform: pytorch Transform for transforms and tensor conversion
        :param years: List of years to take images from, None to not filter
        :param categories: List of categories to take images from, None to not filter
        :param label_type: 'values' for single label, 'one-hot' for one hot labels
        :param masked_bands: List of indices corresponding to which bands to mask out
        :param dropped_bands:  List of indices corresponding to which bands to drop from input image tensor
        """
        super().__init__(in_c=13)
        self.df = pd.read_csv(csv_path) \
            .sort_values(['category', 'location_id', 'timestamp'])

        # Filter by category
        self.categories = CATEGORIES
        if categories is not None:
            self.categories = categories
            self.df = self.df.loc[categories]

        # Filter by year
        if years is not None:
            self.df['year'] = [int(timestamp.split('-')[0]) for timestamp in self.df['timestamp']]
            self.df = self.df[self.df['year'].isin(years)]

        self.indices = self.df.index.unique().to_numpy()

        self.transform = transform

        if label_type not in self.label_types:
            raise ValueError(
                f'FMOWDataset label_type {label_type} not allowed. Label_type must be one of the following:',
                ', '.join(self.label_types))
        self.label_type = label_type

        self.masked_bands = masked_bands
        self.dropped_bands = dropped_bands
        if self.dropped_bands is not None:
            self.in_c = self.in_c - len(dropped_bands)

    def __len__(self):
        return len(self.df)

    def open_image(self, img_path):
        with rasterio.open(img_path) as data:
            # img = data.read(
            #     out_shape=(data.count, self.resize, self.resize),
            #     resampling=Resampling.bilinear
            # )
            img = data.read()  # (c, h, w)

        return img.transpose(1, 2, 0).astype(np.float32)  # (h, w, c)

    def __getitem__(self, idx):
        """
        Gets image (x,y) pair given index in dataset.
        :param idx: Index of (image, label) pair in dataset dataframe. (c, h, w)
        :return: Torch Tensor image, and integer label as a tuple.
        """
        selection = self.df.iloc[idx]

        # images = [torch.FloatTensor(rasterio.open(img_path).read()) for img_path in image_paths]
        images = self.open_image(selection['image_path'])  # (h, w, c)
        if self.masked_bands is not None:
            images[:, :, self.masked_bands] = np.array(self.mean)[self.masked_bands]

        labels = self.categories.index(selection['category'])

        img_as_tensor = self.transform(images)  # (c, h, w)
        if self.dropped_bands is not None:
            keep_idxs = [i for i in range(img_as_tensor.shape[0]) if i not in self.dropped_bands]
            img_as_tensor = img_as_tensor[keep_idxs, :, :]

        sample = {
            'images': images,
            'labels': labels,
            'image_ids': selection['image_id'],
            'timestamps': selection['timestamp']
        }
        return img_as_tensor, labels

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(SentinelNormalize(mean, std))  # use specific Sentinel normalization to avoid NaN
            t.append(transforms.ToTensor())
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(SentinelNormalize(mean, std))
        t.append(transforms.ToTensor())
        t.append(
            transforms.Resize(size, interpolation=interpol_mode),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        return transforms.Compose(t)


class EuroSat(SatelliteDataset):
    mean = [1370.19151926, 1184.3824625, 1120.77120066, 1136.26026392,
            1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
            1972.62420416, 582.72633433, 14.77112979, 1732.16362238, 1247.91870117]
    std = [633.15169573, 650.2842772, 712.12507725, 965.23119807,
           948.9819932, 1108.06650639, 1258.36394548, 1233.1492281,
           1364.38688993, 472.37967789, 14.3114637, 1310.36996126, 1087.6020813]

    def __init__(self, file_path, transform, masked_bands=None, dropped_bands=None):
        """
        Creates dataset for multi-spectral single image classification for EuroSAT.
        :param file_path: path to txt file containing paths to image data for EuroSAT.
        :param transform: pytorch Transform for transforms and tensor conversion
        :param masked_bands: List of indices corresponding to which bands to mask out
        :param dropped_bands:  List of indices corresponding to which bands to drop from input image tensor
        """
        super().__init__(13)
        with open(file_path, 'r') as f:
            data = f.read().splitlines()
        self.img_paths = [row.split()[0] for row in data]
        self.labels = [int(row.split()[1]) for row in data]

        self.transform = transform

        self.masked_bands = masked_bands
        self.dropped_bands = dropped_bands
        if self.dropped_bands is not None:
            self.in_c = self.in_c - len(dropped_bands)

    def __len__(self):
        return len(self.img_paths)

    def open_image(self, img_path):
        with rasterio.open(img_path) as data:
            img = data.read()  # (c, h, w)

        return img.transpose(1, 2, 0).astype(np.float32)  # (h, w, c)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        img = self.open_image(img_path)  # (h, w, c)
        if self.masked_bands is not None:
            img[:, :, self.masked_bands] = np.array(self.mean)[self.masked_bands]

        img_as_tensor = self.transform(img)  # (c, h, w)
        if self.dropped_bands is not None:
            keep_idxs = [i for i in range(img_as_tensor.shape[0]) if i not in self.dropped_bands]
            img_as_tensor = img_as_tensor[keep_idxs, :, :]

        return img_as_tensor, label


def build_fmow_dataset(is_train: bool, args) -> SatelliteDataset:
    """
    Initializes a SatelliteDataset object given provided args.
    :param is_train: Whether we want the dataset for training or evaluation
    :param args: Argparser args object with provided arguments
    :return: SatelliteDataset object.
    """
    csv_path = os.path.join(args.train_path if is_train else args.test_path)

    if args.dataset_type == 'rgb':
        mean = CustomDatasetFromImages.mean
        std = CustomDatasetFromImages.std
        transform = CustomDatasetFromImages.build_transform(is_train, args.input_size, mean, std)
        dataset = CustomDatasetFromImages(csv_path, transform)
    elif args.dataset_type == 'temporal':
        dataset = CustomDatasetFromImagesTemporal(csv_path)
    elif args.dataset_type == 'sentinel':
        mean = SentinelIndividualImageDataset.mean
        std = SentinelIndividualImageDataset.std
        transform = SentinelIndividualImageDataset.build_transform(is_train, args.input_size, mean, std)
        dataset = SentinelIndividualImageDataset(csv_path, transform, masked_bands=args.masked_bands,
                                                 dropped_bands=args.dropped_bands)
    elif args.dataset_type == 'rgb_temporal_stacked':
        mean = FMoWTemporalStacked.mean
        std = FMoWTemporalStacked.std
        transform = FMoWTemporalStacked.build_transform(is_train, args.input_size, mean, std)
        dataset = FMoWTemporalStacked(csv_path, transform)
    elif args.dataset_type == 'euro_sat':
        mean, std = EuroSat.mean, EuroSat.std
        transform = EuroSat.build_transform(is_train, args.input_size, mean, std)
        dataset = EuroSat(csv_path, transform, masked_bands=args.masked_bands, dropped_bands=args.dropped_bands)
    elif args.dataset_type == 'naip':
        from util.naip_loader import NAIP_train_dataset, NAIP_test_dataset, NAIP_CLASS_NUM
        dataset = NAIP_train_dataset if is_train else NAIP_test_dataset
        args.nb_classes = NAIP_CLASS_NUM
    else:
        raise ValueError(f"Invalid dataset type: {args.dataset_type}")
    print(dataset)

    return dataset
