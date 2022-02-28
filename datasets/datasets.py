import os

import numpy as np
import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms

from utils.utils import set_random_seed

from PIL import Image
import torch.utils.data as data

DATA_PATH = '/home/silvia/data/'

if os.path.isdir("/scratch/ImageNet"):
    IMAGENET_PATH = os.path.expanduser('/scratch/ImageNet/')
else:
    IMAGENET_PATH = os.path.expanduser('~/data/ImageNet/')

    print(f"Loading data from: {IMAGENET_PATH}")



CIFAR10_SUPERCLASS = list(range(10))  # one class
IMAGENET_SUPERCLASS = list(range(1000))  # one class

CIFAR100_SUPERCLASS = [
    [4, 31, 55, 72, 95],
    [1, 33, 67, 73, 91],
    [54, 62, 70, 82, 92],
    [9, 10, 16, 29, 61],
    [0, 51, 53, 57, 83],
    [22, 25, 40, 86, 87],
    [5, 20, 26, 84, 94],
    [6, 7, 14, 18, 24],
    [3, 42, 43, 88, 97],
    [12, 17, 38, 68, 76],
    [23, 34, 49, 60, 71],
    [15, 19, 21, 32, 39],
    [35, 63, 64, 66, 75],
    [27, 45, 77, 79, 99],
    [2, 11, 36, 46, 98],
    [28, 30, 44, 78, 93],
    [37, 50, 65, 74, 80],
    [47, 52, 56, 59, 96],
    [8, 13, 48, 58, 90],
    [41, 69, 81, 85, 89],
]

class Dataset(data.Dataset):
    def __init__(self, names,labels, path_dataset,img_transformer=None,dataset=None):
        self.data_path = path_dataset
        self.names = names
        self.labels = labels
        self._image_transformer = img_transformer
        self.dataset = dataset


    def __getitem__(self, index):

        if 'DomainNet' in self.dataset or 'DN_IO' in self.dataset or 'DN_P' in self.dataset or 'DN_S' in self.dataset:
            framename = self.data_path + '/DomainNet/' + self.names[index]
        else:
            framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')

        img = self._image_transformer(img)

        return img,int(self.labels[index])

    def __len__(self):
        return len(self.names)

def dataset_info(txt_labels):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []

    for row in images_list:
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels


class MultiDataTransform(object):
    def __init__(self, transform):
        self.transform1 = transform
        self.transform2 = transform

    def __call__(self, sample):
        x1 = self.transform1(sample)
        x2 = self.transform2(sample)
        return x1, x2


class MultiDataTransformList(object):
    def __init__(self, transform, clean_trasform, sample_num):
        self.transform = transform
        self.clean_transform = clean_trasform
        self.sample_num = sample_num

    def __call__(self, sample):
        set_random_seed(0)

        sample_list = []
        for i in range(self.sample_num):
            sample_list.append(self.transform(sample))

        return sample_list, self.clean_transform(sample)


def get_transform(image_size=None):
    # Note: data augmentation is implemented in the layers
    # Hence, we only define the identity transformation here
    if image_size:  # use pre-specified image size
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
    else:  # use default image size
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.ToTensor()

    return train_transform, test_transform


def get_subset_with_len(dataset, length, shuffle=False):
    set_random_seed(0)
    dataset_size = len(dataset)

    index = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(index)

    index = torch.from_numpy(index[0:length])
    subset = Subset(dataset, index)

    assert len(subset) == length

    return subset


def get_transform_imagenet():

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_transform = MultiDataTransform(train_transform)

    return train_transform, test_transform


def get_dataset(P, dataset, test_only=False, image_size=None, download=False, eval=False):
    if dataset in ['imagenet', 'cub', 'stanford_dogs', 'flowers102',
                   'places365', 'food_101', 'caltech_256', 'dtd', 'pets']:
        if eval:
            train_transform, test_transform = get_simclr_eval_transform_imagenet(P.ood_samples,
                                                                                 P.resize_factor, P.resize_fix)
        else:
            train_transform, test_transform = get_transform_imagenet()
    else:
        train_transform, test_transform = get_transform(image_size=(224,224))

    if dataset == 'cifar10':
        image_size = (32, 32, 3)
        n_classes = 10
        train_set = datasets.CIFAR10(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR10(DATA_PATH, train=False, download=download, transform=test_transform)

    elif dataset == 'cifar100':
        image_size = (32, 32, 3)
        n_classes = 100
        train_set = datasets.CIFAR100(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR100(DATA_PATH, train=False, download=download, transform=test_transform)

    elif dataset == 'svhn':
        assert test_only and image_size is not None
        test_set = datasets.SVHN(DATA_PATH, split='test', download=download, transform=test_transform)

    elif dataset == 'lsun_resize':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'LSUN_resize')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'lsun_fix':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'LSUN_fix')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'imagenet_resize':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'Imagenet_resize')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'imagenet_fix':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'Imagenet_fix')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'imagenet':
        image_size = (224, 224, 3)
        n_classes = 1000
        train_dir = os.path.join(IMAGENET_PATH, 'Data/CLS-LOC/train')
        test_dir = os.path.join(IMAGENET_PATH, 'Data/CLS-LOC/train')
        train_set = datasets.ImageFolder(train_dir, transform=train_transform)
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'OfficeHome_DG':
        image_size = (224, 224, 3)
        n_classes = 54

        target = P.target

        path_source = 'data_txt/'+dataset+'/no_'+target+'.txt'

        names, labels = dataset_info(path_source)
        train_set = Dataset(names,labels, DATA_PATH,img_transformer=train_transform,dataset=dataset)

        names, labels = dataset_info(path_source)
        test_set = Dataset(names,labels, DATA_PATH,img_transformer=test_transform,dataset=dataset)


    elif dataset=='PACS_DG':
        image_size = (224, 224, 3)
        n_classes = 6

        target = P.target

        path_source = 'data_txt/'+dataset+'/no_'+target+'.txt'

        names, labels = dataset_info(path_source)
        train_set = Dataset(names,labels, DATA_PATH,img_transformer=train_transform,dataset=dataset)

        names, labels = dataset_info(path_source)
        test_set = Dataset(names,labels, DATA_PATH,img_transformer=test_transform,dataset=dataset)

    elif dataset=='MultiDatasets_DG':
        image_size = (224, 224, 3)
        n_classes = 48

        target = P.target

        path_source = 'data_txt/'+dataset+'/Sources.txt'

        names, labels = dataset_info(path_source)
        train_set = Dataset(names,labels, DATA_PATH,img_transformer=train_transform,dataset=dataset)

        names, labels = dataset_info(path_source)
        test_set = Dataset(names,labels, DATA_PATH,img_transformer=test_transform,dataset=dataset)

    elif dataset=='DTD' :
        image_size = (224, 224, 3)
        n_classes = 23

        target = P.target

        path_source = 'data_txt/'+dataset+'/source.txt'

        names, labels = dataset_info(path_source)
        train_set = Dataset(names,labels, DATA_PATH,img_transformer=train_transform,dataset=dataset)

        names, labels = dataset_info(path_source)
        test_set = Dataset(names,labels, DATA_PATH,img_transformer=test_transform,dataset=dataset)


    elif  dataset=='DomainNet_IN_OUT' or dataset=='DomainNet_Painting' or dataset =='DomainNet_Sketch':
        image_size = (224, 224, 3)
        n_classes = 25

        target = P.target

        path_source = 'data_txt/'+dataset+'/source.txt'

        names, labels = dataset_info(path_source)
        train_set = Dataset(names,labels, DATA_PATH,img_transformer=train_transform,dataset=dataset)

        names, labels = dataset_info(path_source)
        test_set = Dataset(names,labels, DATA_PATH,img_transformer=test_transform,dataset=dataset)


    elif  dataset=='OfficeHome_SS_DG':
        image_size = (224, 224, 3)
        n_classes = 25

        target = P.target

        path_source = 'data_txt/'+dataset+'/source.txt'

        names, labels = dataset_info(path_source)
        train_set = Dataset(names,labels, DATA_PATH,img_transformer=train_transform,dataset=dataset)

        names, labels = dataset_info(path_source)
        test_set = Dataset(names,labels, DATA_PATH,img_transformer=test_transform,dataset=dataset)

    elif dataset=='PACS_SS_DG':
        image_size = (224, 224, 3)
        n_classes = 6

        target = P.target

        path_source = 'data_txt/'+dataset+'/source.txt'

        names, labels = dataset_info(path_source)
        train_set = Dataset(names,labels, DATA_PATH,img_transformer=train_transform,dataset=dataset)

        names, labels = dataset_info(path_source)
        test_set = Dataset(names,labels, DATA_PATH,img_transformer=test_transform,dataset=dataset)

    elif dataset == 'Clipart_MD' or dataset == 'Real_MD' or dataset == 'Painting_MD' or dataset == 'Sketch_MD':

        image_size = (224, 224, 3)
        n_classes = 6

        target = P.target

        path_target_in = 'data_txt/MultiDatasets_DG/'+target+'_in.txt'
        path_target_out = 'data_txt/MultiDatasets_DG/' + target + '_out.txt'

        names_in, labels_in = dataset_info(path_target_in)
        names_out, labels_out = dataset_info(path_target_out)
        test_set_in = Dataset(names_in,labels_in, DATA_PATH,img_transformer=test_transform,dataset=dataset)
        test_set_out = Dataset(names_out, labels_out, DATA_PATH, img_transformer=test_transform,dataset=dataset)

        test_set = [test_set_in,test_set_out]

    elif dataset == 'Art' or dataset == 'Clipart' or dataset == 'Product' or dataset == 'RealWorld':

        image_size = (224, 224, 3)
        n_classes = 54

        target = P.target

        path_target_in = 'data_txt/OfficeHome_DG/'+target+'_in.txt'
        path_target_out = 'data_txt/OfficeHome_DG/' + target + '_out.txt'

        names_in, labels_in = dataset_info(path_target_in)
        names_out, labels_out = dataset_info(path_target_out)
        test_set_in = Dataset(names_in,labels_in, DATA_PATH,img_transformer=test_transform,dataset=dataset)
        test_set_out = Dataset(names_out, labels_out, DATA_PATH, img_transformer=test_transform,dataset=dataset)

        test_set = [test_set_in, test_set_out]


    elif dataset == 'ArtPainting' or dataset == 'Cartoon' or dataset == 'Sketch' or dataset == 'Photo':
        image_size = (224, 224, 3)
        n_classes = 6

        target = P.target

        path_target_in = 'data_txt/PACS_DG/'+target+'_in.txt'
        path_target_out = 'data_txt/PACS_DG/' + target + '_out.txt'

        names_in, labels_in = dataset_info(path_target_in)
        names_out, labels_out = dataset_info(path_target_out)
        test_set_in = Dataset(names_in,labels_in, DATA_PATH,img_transformer=test_transform,dataset=dataset)
        test_set_out = Dataset(names_out, labels_out, DATA_PATH, img_transformer=test_transform,dataset=dataset)

        test_set = [test_set_in, test_set_out]

    elif dataset == 'out':
        image_size = (224, 224, 3)
        n_classes = 23

        target = P.target

        path_target_in = 'data_txt/DTD/target_in.txt'
        path_target_out = 'data_txt/DTD/target_out.txt'

        names_in, labels_in = dataset_info(path_target_in)
        names_out, labels_out = dataset_info(path_target_out)
        test_set_in = Dataset(names_in,labels_in, DATA_PATH,img_transformer=test_transform,dataset=dataset)
        test_set_out = Dataset(names_out, labels_out, DATA_PATH, img_transformer=test_transform,dataset=dataset)

        test_set = [test_set_in, test_set_out]


    elif dataset == 'DN_IO':
        image_size = (224, 224, 3)
        n_classes = 25

        target = P.target

        path_target_in = 'data_txt/DomainNet_IN_OUT/DN_IO_target_in.txt'
        path_target_out = 'data_txt/DomainNet_IN_OUT/DN_IO_target_out.txt'

        names_in, labels_in = dataset_info(path_target_in)
        names_out, labels_out = dataset_info(path_target_out)
        test_set_in = Dataset(names_in,labels_in, DATA_PATH,img_transformer=test_transform,dataset=dataset)
        test_set_out = Dataset(names_out, labels_out, DATA_PATH, img_transformer=test_transform,dataset=dataset)

        test_set = [test_set_in, test_set_out]

    elif dataset == 'DN_P':
        image_size = (224, 224, 3)
        n_classes = 25

        target = P.target

        path_target_in = 'data_txt/DomainNet_Painting/DN_P_target_in.txt'
        path_target_out = 'data_txt/DomainNet_Painting/DN_P_target_out.txt'

        names_in, labels_in = dataset_info(path_target_in)
        names_out, labels_out = dataset_info(path_target_out)
        test_set_in = Dataset(names_in,labels_in, DATA_PATH,img_transformer=test_transform,dataset=dataset)
        test_set_out = Dataset(names_out, labels_out, DATA_PATH, img_transformer=test_transform,dataset=dataset)

        test_set = [test_set_in, test_set_out]


    elif dataset == 'DN_S':
        image_size = (224, 224, 3)
        n_classes = 25

        target = P.target

        path_target_in = 'data_txt/DomainNet_Sketch/DN_S_target_in.txt'
        path_target_out = 'data_txt/DomainNet_Sketch/DN_S_target_out.txt'

        names_in, labels_in = dataset_info(path_target_in)
        names_out, labels_out = dataset_info(path_target_out)
        test_set_in = Dataset(names_in,labels_in, DATA_PATH,img_transformer=test_transform,dataset=dataset)
        test_set_out = Dataset(names_out, labels_out, DATA_PATH, img_transformer=test_transform,dataset=dataset)

        test_set = [test_set_in, test_set_out]

    elif dataset == 'stanford_dogs':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'stanford_dogs')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'cub':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'cub200')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'flowers102':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'flowers102')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'places365':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'places365')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'food_101':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'food-101', 'images')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'caltech_256':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'caltech-256')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'dtd':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'dtd', 'images')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'pets':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'pets')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    else:
        raise NotImplementedError()

    if test_only:
        return test_set
    else:
        return train_set, test_set, image_size, n_classes


def get_superclass_list(dataset):
    if dataset == 'cifar10':
        return CIFAR10_SUPERCLASS
    elif dataset == 'cifar100':
        return CIFAR100_SUPERCLASS
    elif dataset == 'imagenet':
        return IMAGENET_SUPERCLASS
    else:
        raise NotImplementedError()


def get_subclass_dataset(dataset, classes):
    if not isinstance(classes, list):
        classes = [classes]

    indices = []
    for idx, tgt in enumerate(dataset.targets):
        if tgt in classes:
            indices.append(idx)

    dataset = Subset(dataset, indices)
    return dataset


def get_simclr_eval_transform_imagenet(sample_num, resize_factor, resize_fix):

    resize_scale = (resize_factor, 1.0)  # resize scaling factor
    if resize_fix:  # if resize_fix is True, use same scale
        resize_scale = (resize_factor, resize_factor)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=resize_scale),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    clean_trasform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    transform = MultiDataTransformList(transform, clean_trasform, sample_num)

    return transform, transform


