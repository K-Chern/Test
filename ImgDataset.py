import numpy as np
import glob
import torch.utils.data
import os
import math
from skimage import io, transform
from PIL import Image
import torch
import torchvision as vision
from torchvision import transforms, datasets
import random

class MultiviewImgDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='train', scale_aug=False, rot_aug=False, num_models=0, num_views=12, test_mode=False):
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.num_views = num_views
        self.split = split
        self.classnames = sorted(class_folders)

        class_folders = sorted([os.path.join(root_dir, class_name) for class_name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, class_name))])
        
        class_id = 0

        class_folders = []  # Initialize class_folders list
        for class_folder in class_folders:
            class_split_folder = os.path.join(class_folder, self.split)
            if os.path.isdir(class_split_folder):
                model_folders = sorted([os.path.join(class_split_folder, model_name) for model_name in os.listdir(class_split_folder) if os.path.isdir(os.path.join(class_split_folder, model_name))])
                for model_folder in model_folders:
                    self.filepaths.append([])
                    for view in range(self.num_views):
                        self.filepaths[-1].append(os.path.join(model_folder, '%02d.png' % view))
                    self.labels.append(class_id)
            class_id += 1

        self.classnames = sorted(folders)

        self.filepaths = []
        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(os.path.join(root_dir, self.classnames[i], 'train', '*.png')))

            # Add print statement here to check file paths
            print(f"Loading class: {self.classnames[i]}")
            if all_files:
                print(f"Sample file path: {all_files[0]}")

            ## Select subset for different number of views
            stride = int(12/self.num_views) # 12 6 4 3 2 1
            all_files = all_files[::stride]

            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models,len(all_files))])

        if shuffle:
            # permute
            rand_idx = np.random.permutation(int(len(self.filepaths)/num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.filepaths[rand_idx[i]*num_views:(rand_idx[i]+1)*num_views])
            self.filepaths = filepaths_new

        if self.test_mode:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])    
        else:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            print("Loaded models:", len(self.filepaths))
            for class_folder in os.listdir(root_dir):
                print("Found class folder:", class_folder)


    def __len__(self):
        return int(len(self.filepaths)/self.num_views)

    def __getitem__(self, idx):
        path = self.filepaths[idx*self.num_views]
        class_name = os.path.split(os.path.dirname(os.path.dirname(path)))[-1]
        class_id = self.classnames.index(class_name)
        imgs = []
        for i in range(self.num_views):
            im = Image.open(self.filepaths[idx*self.num_views+i]).convert('RGB')
            if self.transform:
                im = self.transform(im)
            imgs.append(im)

        return (class_id, torch.stack(imgs), self.filepaths[idx*self.num_views:(idx+1)*self.num_views])

class SingleImgDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
                 num_models=0, num_views=12):
        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode

        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/',2)[0]
        self.filepaths = []
        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(parent_dir+'/'+self.classnames[i]+'/'+set_+'/*shaded*.png'))
            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models,len(all_files))])

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        class_name=path.split('/')[-2]
        class_id = self.classnames.index(class_name)

        # Use PIL instead
        im = Image.open(self.filepaths[idx]).convert('RGB')
        if self.transform:
            im = self.transform(im)

        return (class_id, im, path)


