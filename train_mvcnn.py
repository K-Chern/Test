import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os, shutil, json
import argparse
import gc

from tools.Trainer import ModelNetTrainer
from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from models.MVCNN import MVCNN, SVCNN

# Clear GPU cache and release any unreleased memory
torch.cuda.empty_cache()
gc.collect()
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="MVCNN")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=2)  # it will be *12 images in each batch for mvcnn
parser.add_argument("-num_models", type=int, help="number of models per class", default=1000)
parser.add_argument("-lr", type=float, help="learning rate", default=5e-5)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.0)
parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true')
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="vgg11")
parser.add_argument("-num_views", type=int, help="number of views", default=12)
parser.add_argument('-train_path', type=str, default='modelnet40_images_new_12x/train', help='Path to training set')
parser.add_argument('-val_path', type=str, default='modelnet40_images_new_12x/val', help='Path to validation set')
parser.set_defaults(train=False)

def create_folder(log_dir):
    # make summary folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)

if __name__ == '__main__':
    args = parser.parse_args()

    pretraining = not args.no_pretraining
    log_dir = args.name
    create_folder(args.name)
    config_f = open(os.path.join(log_dir, 'config.json'), 'w')
    json.dump(vars(args), config_f)
    config_f.close()

    # STAGE 1
    log_dir = args.name+'_stage_1'
    create_folder(log_dir)
    cnet = SVCNN(args.name, nclasses=40, pretraining=pretraining, cnn_name=args.cnn_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnet.to(device)  # Move the model to GPU

    # Add this line after creating the cnet object
    device = torch.device("cuda:0")

    optimizer = optim.Adam(cnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    n_models_train = args.num_models*args.num_views

    train_dataset = MultiviewImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views)
    print(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)

    val_dataset = MultiviewImgDataset(args.val_path, scale_aug=False, rot_aug=False, num_models=n_models_test, num_views=args.num_views)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
class_folders
print('num_train_files: '+str(len(train_dataset.filepaths)))
print('num_val_files: '+str(len(val_dataset.filepaths)))

print("Train dataset length:", len(train_dataset))
print("Val dataset length:", len(val_dataset))

trainer = ModelNetTrainer(cnet, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'svcnn', log_dir, num_views=1)
trainer.train(1)

# STAGE 2
log_dir = args.name+'_stage_2'
create_folder(log_dir)
cnet_2 = MVCNN(args.name, cnet, nclasses=40, cnn_name=args.cnn_name, num_views=args.num_views)
del cnet

cnet_2.to(device)  # Move the model to GPU

optimizer = optim.Adam(cnet_2.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

train_dataset = MultiviewImgDataset(args.train_path, num_views=args.num_views)
print("Number of samples in train_dataset:", len(train_dataset))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)

val_dataset = MultiviewImgDataset(args.val_path, num_views=args.num_views)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)

print('num_train_files: '+str(len(train_dataset.filepaths)))
print('num_val_files: '+str(len(val_dataset.filepaths)))

trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'mvcnn', log_dir, num_views=args.num_views)
trainer.train(1)

