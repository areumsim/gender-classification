### ref. https://github.com/Robocup-Lyontech/liris_person_attributes/blob/master/loader_peta_dataset.py

import os
import random
from glob import glob
import pathlib
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import yaml
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt

import torch
from einops import rearrange
from tqdm import tqdm
import fnmatch


def show_original_image(image):
    image = torch.clamp(image, -1, 1)
    img = image.cpu().numpy().copy()
    img *= np.array([0.229, 0.224, 0.225])[:, None, None]
    img += np.array([0.485, 0.456, 0.406])[:, None, None]

    img = rearrange(img, "c h w -> h w c")
    img = img * 255
    img = img.astype(np.uint8)
    return img


def create_transforms(height, width):
    """
    Create a torchvision transforms pipeline for image preprocessing.

    Parameters:
    - height (int): The height to resize images to.
    - width (int): The width to resize images to.

    Returns:
    - A torchvision.transforms.Compose object that resizes, converts images to tensors,
      and normalizes them.
    """
    return transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def make_augmentation(height=224, width=224):
    """
    Create an imgaug augmentation pipeline.
    """
    return iaa.Sequential(
        [
            iaa.Sometimes(
                0.6,
                iaa.SomeOf(
                    (1, 2),  # Apply 1 to 3 of the following augmentations
                    [
                        iaa.Fliplr(
                            0.5
                        ),  # Apply horizontal flips with a probability of 50%
                        iaa.Crop(
                            percent=(0, 0.1)
                        ),  # Randomly crop the image by 0% to 10%
                        iaa.GaussianBlur(
                            sigma=(0, 0.5)
                        ),  # Apply Gaussian blur with sigma between 0 and 0.5
                        iaa.LinearContrast(
                            (0.75, 1.5)
                        ),  # Improve or reduce contrast by a factor between 0.75 and 1.5
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                        ),  # Add Gaussian noise to the image
                        iaa.Multiply(
                            (0.8, 1.2)
                        ),  # Change brightness by multiplying pixel values by values between 0.8 and 1.2
                        iaa.Affine(
                            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                            rotate=(-25, 25),
                            shear=(-8, 8),
                        ),  # Apply affine transformations: scaling, translation, rotation, and shearing
                        iaa.Resize(
                            {"height": 0.8, "width": 0.8}
                        ),  # Resize the image to 80% of its original dimensions
                        iaa.PadToFixedSize(
                            width=width, height=height, position="uniform"
                        ),
                        iaa.MultiplyHue((0.5, 1.5)),
                        iaa.ChangeColorTemperature((1100, 10000)),
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                    ],
                ),
            ),
        ]
    )


class PETADataset(Dataset):
    """
    A PyTorch Dataset class for loading PETA dataset images with gender labels.

    Attributes:
    - train (bool): Flag to determine if the dataset is for training or validation.
    - cfg (dict): Configuration dictionary containing 'data_dir', 'height', and 'width'.
    - split_ratio (float): Fraction of data to use as training data.
    - seed (int): Random seed for shuffling the data.
    """

    def __init__(self, train, cfg, split_ratio=0.8, seed=42, augmentation=False):
        self.is_train = train
        self.data_dir = cfg["PETA"]["data_dir"]
        self.transform = create_transforms(cfg["PETA"]["height"], cfg["PETA"]["width"])

        if not os.path.exists("./dataset_tmp/paths.txt") and not os.path.exists(
            "labels.txt"
        ):
            self.image_files, self.labels = self.load_dataset()
        else:
            with open("./dataset_tmp/paths.txt", "r") as file:
                self.image_files = file.read().splitlines()
            with open("./dataset_tmp/labels.txt", "r") as file:
                self.labels = file.read().splitlines()
            self.labels = [int(label) for label in self.labels]

        # Shuffle and split the dataset
        self.image_files, self.labels = self.shuffle_and_split(
            self.image_files, self.labels, self.is_train, split_ratio, seed
        )

        self.augmentation = augmentation if augmentation is not None else False
        if self.augmentation:
            self.augmentation_seq = make_augmentation(
                cfg["PETA"]["height"], cfg["PETA"]["width"]
            )

    def load_dataset(self):
        """
        Loads dataset images and labels from the directory.

        Returns:
        - image_files (list): List of image file paths.
        - labels (list): List of labels corresponding to image_files.
        """
        image_files = []
        labels = []
        for s_dataset in glob(os.path.join(self.data_dir, "*")):
            label_path = pathlib.Path(os.path.join(s_dataset, "archive", "Label.txt"))
            s_labels = label_path.read_text().splitlines()

            # Extract filenames for male and female images
            m_files = [x.split()[0] for x in s_labels if "personalMale" in x]
            fm_files = [x.split()[0] for x in s_labels if "personalFemale" in x]
            m_files = list(set(m_files))
            fm_files = list(set(fm_files))

            # Build file paths and labels
            all_fm_paths = []
            for x in tqdm(fm_files):
                all_fm_paths.extend(glob(os.path.join(s_dataset, "archive", x) + "_*"))
                all_fm_paths.extend(glob(os.path.join(s_dataset, "archive", x)))
            for image_path in all_fm_paths:
                image_files.append(image_path)
                labels.append(0)

            all_m_paths = []
            for x in tqdm(m_files):
                all_m_paths.extend(glob(os.path.join(s_dataset, "archive", x) + "_*"))
                all_m_paths.extend(glob(os.path.join(s_dataset, "archive", x)))
            for image_path in all_m_paths:
                image_files.append(image_path)
                labels.append(1)

        # Write the file paths to the file
        file_path = "./dataset_tmp/paths.txt"
        with open(file_path, "w") as file:
            for path in image_files:
                file.write(path + "\n")

        # Write the labels to the file
        file_path = "./dataset_tmp/labels.txt"
        with open(file_path, "w") as file:
            for path in labels:
                file.write(str(path) + "\n")

        return image_files, labels

    def shuffle_and_split(self, image_files, labels, train, split_ratio, seed):
        """
        Shuffles and splits the dataset into training and validation sets.

        Parameters:
        - image_files (list): List of image file paths.
        - labels (list): List of labels.
        - train (bool): If True, returns training set; otherwise, validation set.
        - split_ratio (float): Fraction of data to use as training data.
        - seed (int): Random seed for shuffling.

        Returns:
        - A tuple (image_files, labels) corresponding to the chosen dataset split.
        """
        random.seed(seed)
        indices = list(range(len(image_files)))
        random.shuffle(indices)

        split_idx = int(len(indices) * split_ratio)
        train_indices = indices[:split_idx]  #### test  [:1500]
        valid_indices = indices[split_idx:]  #### test  [:100]

        ### TODO: (train/valid/test로) split_ratio를 (0.7, 0.2)로 입력 ###
        # train_end = int(len(indices) * split_ratios[0])
        # valid_end = train_end + int(len(indices) * split_ratios[1])

        # train_indices = indices[:train_end]
        # valid_indices = indices[train_end:valid_end]
        # test_indices = indices[valid_end:]

        ## (train=T/F)를 mode를 변경
        # elif mode == "test":
        #     return [image_files[i] for i in test_indices], [
        #         labels[i] for i in test_indices
        #     ]

        #  count each label (0 or 1 ) in train and valid
        print(f"train_indices: {len(train_indices)}")
        print(f"valid_indices: {len(valid_indices)}")
        print(f"train_labels: {sum([labels[i] for i in train_indices])}")
        print(f"valid_labels: {sum([labels[i] for i in valid_indices])}")

        if train:
            return [image_files[i] for i in train_indices], [
                labels[i] for i in train_indices
            ]
        else:
            return [image_files[i] for i in valid_indices], [
                labels[i] for i in valid_indices
            ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Fetch an image and its label by index, applying transformations to the image.

        Parameters:
        - idx (int): Index of the image to retrieve.

        Returns:
        - A tuple of (image, label), where image is a tensor.
        """
        img_name = self.image_files[idx]
        label = self.labels[idx]
        image = Image.open(img_name).convert("RGB")  # Convert image to RGB

        if self.is_train and self.augmentation:
            original_image = image
            image = self.augmentation_seq(image=np.array(original_image))
            image = Image.fromarray(image)

        ### Show concat image  the original image and the augment image
        # original_img = show_original_image(self.transform(original_image))
        # augment_img = show_original_image(self.transform(image))
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.imshow(original_img)
        # plt.title(f"Original Image")
        # plt.axis("off")
        # plt.subplot(1, 2, 2)
        # plt.imshow(augment_img)
        # plt.title(f"Augment Image")
        # plt.axis("off")

        image = self.transform(image)

        return (image, label)


if __name__ == "__main__":
    with open("config.yaml", "r") as stream:
        cfg = yaml.safe_load(stream)

    train_dataset = PETADataset(True, cfg["data"], augmentation=True)
    valid_dataset = PETADataset(False, cfg["data"])

    # Example to fetch an image and its label from the validation set
    image, label = train_dataset[0]
    print(f"Label: {label}, Image shape: {image.shape}")

    # Show the original image
    original_img = show_original_image(image)
    label = ["male" if label == 1 else "female"]
    plt.imshow(original_img)
    plt.title(f"{label[0]}")
    plt.axis("off")
