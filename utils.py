import torch
import torch.utils.data as data
import torchvision
import os
from PIL import Image
import glob

'''
def load_data():
    # read addresses and labels from the 'train' folder
    images = sorted(glob.glob('./DIV2K/DIV2K_train_LR_bicubic/X4/*.png'))
    labels = sorted(glob.glob('./DIV2K/DIV2K_train_HR/*.png'))

    images = torchvision.datasets.ImageFolder(root='./DIV2K/DIV2K_train_LR_bicubic/X4')
    labels = torchvision.datasets.ImageFolder(root='./DIV2K/DIV2K_train_HR')
'''

class load_data(data.Dataset):
    """
        CARVANA dataset that contains car images as .jpg. Each car has 16 images
        taken in different angles and a unique id: id_01.jpg, id_02.jpg, ..., id_16.jpg
        The labels are provided as a .gif image that contains the manually cutout mask
        for each training image
    """

    def __init__(self, img_addr, lbl_addr, subset="train", transform=None):
        """

        :param root: it has to be a path to the folder that contains the dataset folders
        :param train: boolean true if you want the train set false for the test one
        :param transform: transform the images and labels
        """

        # initialize variables
        self.img_addr = os.path.expanduser(img_addr)
        self.lbl_addr = os.path.expanduser(lbl_addr)
        self.transform = transform
        self.subset = subset
        self.data_path, self.labels_path = [], []

        def load_images(path):
            """
            returns all the sorted image paths.

            :param path:
            :return: array with all the paths to the images
            """
            images_dir = [os.path.join(path, f) for f in os.listdir(path)]
            images_dir.sort()

            return images_dir

        # load the data regarding the subset
        if self.subset == "train":
            self.data_path = load_images(self.img_addr + "/train")
            self.labels_path = load_images(self.lbl_addr + "/train")
        elif self.subset == "test":
            self.data_path = load_images(self.img_addr + "/test")
            self.labels_path = load_images(self.lbl_addr + "/test")
        else:
            raise RuntimeError('Invalid subset ' + self.subset + ', it must be one of:'
                                                                 ' \'train\' or \'test\'')

    def __getitem__(self, index):
        """
        :param index:
        :return: tuple (img, target) with the input data and its label
        """

        # load image and labels
        img = Image.open(self.data_path[index])
        target = Image.open(self.labels_path[index])

        # apply transforms to both
        if self.transform is not None:
            img = self.transform(img)
            target = self.transform(target)

        return img, target

    def __len__(self):
        return len(self.data_path)