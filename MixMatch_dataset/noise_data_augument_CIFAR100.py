
import numpy as np
#from PIL import Image

import torchvision
import torch


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


def get_cifar100_select_ind(all_data_information, transform_train=None, transform_val=None,
                            download=True):

    root = "./data"
    train_labeled_idxs, weight_list, train_unlabeled_idxs, val_idxs, noise_label_50000 = all_data_information
    train_labeled_dataset = CIFAR100_labeled_noise_label(root, noise_label_50000, train_labeled_idxs, train=True, transform=transform_train)
    train_unlabeled_dataset = CIFAR100_unlabeled(root, train_unlabeled_idxs, train=True, transform=TransformTwice(transform_train))
    
    val_dataset = CIFAR100_labeled(root, val_idxs, train=True, transform=transform_val, download=True)
    test_dataset = CIFAR100_labeled(root, train=False, transform=transform_val, download=True)

    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_idxs)}")
    return train_labeled_dataset, weight_list, train_unlabeled_dataset, val_dataset, test_dataset

cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761) 


def normalise(x, mean=cifar100_mean, std=cifar100_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x


def transpose(x, source='NHWC', target='NCHW'): # batch_size*channnel*height*width
    return x.transpose([source.index(d) for d in target]) 


def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border)], mode='reflect')


class RandomPadandCrop(object):
    """Crop randomly the image.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, x):
        x = pad(x, 4)

        h, w = x.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        x = x[:, top: top + new_h, left: left + new_w]

        return x


class RandomFlip(object):
    """Flip randomly the image.
    """
    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]

        return x.copy()


class GaussianNoise(object):
    """Add gaussian noise to the image.
    """
    def __call__(self, x):
        c, h, w = x.shape
        x += np.random.randn(c, h, w) * 0.15
        return x


class ToTensor(object):
    """Transform the image to tensor.
    """
    def __call__(self, x):
        x = torch.from_numpy(x)
        return x


class CIFAR100_labeled(torchvision.datasets.CIFAR100):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):

        super(CIFAR100_labeled, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        self.train = train
        
        if train == True:
            if indexs is not None:
                self.train_data = self.train_data[indexs]
                self.train_labels = np.array(self.train_labels)[indexs]
                
            self.train_data = transpose(normalise(self.train_data))
        else:
            if indexs is not None:
                self.test_data = self.test_data[indexs]
                self.test_labels = np.array(self.test_labels)[indexs]
                
            self.test_data = transpose(normalise(self.test_data))
            

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train == True:
            img, target = self.train_data[index], self.train_labels[index]
        else: # train = False
            img, target = self.test_data[index], self.test_labels[index]
            
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    

class CIFAR100_labeled_noise_label(torchvision.datasets.CIFAR100):

    def __init__(self, root, noise_labels, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):

        super(CIFAR100_labeled_noise_label, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        self.train = train
        self.noise_labels = noise_labels
        
        if train == True:
            if indexs is not None:
                self.train_data = self.train_data[indexs]
                self.train_labels = np.array(self.noise_labels, dtype=int)[indexs]
                
            self.train_data = transpose(normalise(self.train_data))  # using confident noise label
            
        else:
            if indexs is not None:
                self.test_data = self.test_data[indexs]
                self.test_labels = np.array(self.test_labels)[indexs]
                
            self.test_data = transpose(normalise(self.test_data))
            

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train == True:
            img, target = self.train_data[index], self.train_labels[index]
        else: # train = False
            img, target = self.test_data[index], self.test_labels[index]
            
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100_unlabeled(CIFAR100_labeled):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):

        super(CIFAR100_unlabeled, self).__init__(root, indexs, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)

        self.train_labels = np.array([-1 for i in range(len(self.train_labels))])
        
