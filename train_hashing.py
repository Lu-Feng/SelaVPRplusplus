
import math
import torch
import logging
import numpy as np
from tqdm import tqdm,trange
import torch.nn as nn
import multiprocessing
from os.path import join
from datetime import datetime
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
torch.backends.cudnn.benchmark= True  # Provides a speedup

import util
import test_hashing
import parser
import commons
import datasets_ws
from model import network
from model.sync_batchnorm import convert_model

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3" 

#### Initial setup: parser, logging...
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("logs", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")
logging.info(f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")

#### Creation of Validation Datasets
val_ds0 = datasets_ws.BaseDataset(args, args.datasets_folder, "pitts30k", "val")
logging.info(f"Val set0: {val_ds0}")
val_ds1 = datasets_ws.BaseDataset(args, args.datasets_folder, "msls", "val")
logging.info(f"Val set1: {val_ds1}")

#### Initialize model
model = network.GeoLocalizationNet(args)
model = model.to(args.device)
model = torch.nn.DataParallel(model)

for name, param in model.module.backbone.named_parameters():
    if "adapter" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# initialize Adapter
for n, m in model.named_modules():
    if 'adapter' in n:
        for n2, m2 in m.named_modules():
            if 'D_fc2' in n2:
                if isinstance(m2, nn.Linear):
                    nn.init.constant_(m2.weight, 0.)
                    nn.init.constant_(m2.bias, 0.)
        for n2, m2 in m.named_modules():
            if 'conv' in n2:
                if isinstance(m2, nn.Conv2d):
                    nn.init.constant_(m2.weight, 0.00001)
                    nn.init.constant_(m2.bias, 0.00001)

total = sum([param.nelement() for param in model.module.parameters()])
print("Number of model parameter: %.2fM" % (total/1e6))

total1 = sum([param.nelement() for param in model.module.backbone.parameters()])
print("Number of model backbone parameter: %.2fM" % (total1/1e6))
print("difference: %.2fM" % ((total-total1)/1e6))

# total2 = sum([param.nelement() for param in model.module.aggregation_hashing.parameters()])
# print("Number of aggregation parameter: %.2fM" % (total2/1e6))

total3 = sum([param.nelement() for param in model.module.parameters() if param.requires_grad])
print("Number of trainable parameter: %.2fM" % (total3/1e6))

total4 = sum([param.nelement() for param in model.module.backbone.parameters() if param.requires_grad])
print("Numer of trainable parameter introduced by fine-tuning/adaptation: %.2fM" % (total4/1e6))

#### Setup Optimizer and Loss
if args.optim == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optim == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)
elif args.optim == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=9.5e-9)

### Resume model, optimizer, and other training parameters
if args.resume:
    model, optimizer, best_r1r5, start_epoch_num, not_improved_num = util.resume_train(args, model, optimizer)
    logging.info(f"Resuming from epoch {start_epoch_num} with best recall@1+recall@5 {best_r1r5:.1f}")
    start_epoch_num += 1
else:
    best_r1r5 = start_epoch_num = not_improved_num = 0

if torch.cuda.device_count() >= 2:
    # When using more than 1GPU, use sync_batchnorm for torch.nn.DataParallel
    model = convert_model(model)
    model = model.cuda()

args.binary_features_dim = 512
logging.info(f"Output dimension of the binary features is {args.binary_features_dim}")

from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T
from dataloaders.train.GSVCitiesDataset import GSVCitiesDataset

IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 
                     'std': [0.229, 0.224, 0.225]}

if args.training_dataset == "gsv_cities":
    TRAIN_CITIES = [
        'Bangkok',
        'BuenosAires',
        'LosAngeles',
        'MexicoCity',
        'OSL', # refers to Oslo
        'Rome',
        'Barcelona',
        'Chicago',
        'Madrid',
        'Miami',
        'Phoenix',
        'TRT', # refers to Toronto
        'Boston',
        'Lisbon',
        'Medellin',
        'Minneapolis',
        'PRG', # refers to Prague
        'WashingtonDC',
        'Brussels',
        'London',
        'Melbourne',
        'Osaka',
        'PRS', # refers to Paris
    ]
else:
    TRAIN_CITIES = [
        "SFXL",
        'Bangkok',
        'BuenosAires',
        'LosAngeles',
        'MexicoCity',
        'OSL', # refers to Oslo
        'Rome',
        'Barcelona',
        'Chicago',
        'Madrid',
        'Miami',
        'Phoenix',
        'TRT', # refers to Toronto
        'Boston',
        'Lisbon',
        'Medellin',
        'Minneapolis',
        'PRG', # refers to Prague
        'WashingtonDC',
        'Brussels',
        'London',
        'Melbourne',
        'Osaka',
        'PRS', # refers to Paris
    ]
    citylist = [
        "Trondheim", 
        "Amsterdam",
        "Helsinki",
        "Tokyo",
        "Toronto",
        "Saopaulo",
        "Moscow",
        "Zurich",
        "Paris",
        "Budapest",
        "Austin",
        "Berlin",
        "Ottawa",
        "Goa",
        "Amman",
        "Nairobi",
        "Manila",
        "bangkok",
        "boston",
        "london",
        "melbourne",
        "phoenix",
        "Pitts30k",
    ]

    newcitylist = []
    for i in range(18):
        for cityname in citylist:
            if i==17 and (cityname=="Amman" or cityname=="Nairobi"):
                continue
            else:
                newcitylist.append(cityname+str(i))
    TRAIN_CITIES = TRAIN_CITIES + newcitylist

batch_size=args.train_batch_size
img_per_place=4
min_img_per_place=4
shuffle_all=False
image_size=(224, 224)
num_workers=4
cities=TRAIN_CITIES
mean_std=IMAGENET_MEAN_STD
random_sample_from_each_place=True

mean_dataset = mean_std['mean']
std_dataset = mean_std['std']
train_transform = T.Compose([
    T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
    T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=mean_dataset, std=std_dataset),
])

train_loader_config = {
    'batch_size': batch_size,
    'num_workers': num_workers,
    'drop_last': False,
    'pin_memory': True,
    'shuffle': shuffle_all}

train_dataset = GSVCitiesDataset(
            cities=cities,
            img_per_place=img_per_place,
            min_img_per_place=min_img_per_place,
            random_sample_from_each_place=random_sample_from_each_place,
            transform=train_transform)

# Multi-Similarity Loss and Miner
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity
loss_fn = losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=DotProductSimilarity())
miner = miners.MultiSimilarityMiner(epsilon=0.1, distance=CosineSimilarity())
#  The loss function call (this method will be called at each training iteration)
def loss_function(descriptors, labels):
    # we mine the pairs/triplets if there is an online mining strategy
    if miner is not None:
        miner_outputs = miner(descriptors, labels)
        loss = loss_fn(descriptors, labels, miner_outputs)

        # calculate the % of trivial pairs/triplets 
        # which do not contribute in the loss value
        nb_samples = descriptors.shape[0]
        nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
        batch_acc = 1.0 - (nb_mined/nb_samples)

    else: # no online mining
        loss = loss_fn(descriptors, labels)
        batch_acc = 0.0
    return loss, miner_outputs

# loading training datasets
ds = DataLoader(dataset=train_dataset, **train_loader_config)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(ds)*3, gamma=0.5, last_epoch=-1)

# mixed precision training
from torch.cuda.amp import GradScaler,autocast
scaler = GradScaler()

#### Training loop
for epoch_num in range(start_epoch_num, args.epochs_num):
    logging.info(f"Start training epoch: {epoch_num:02d}")
    
    epoch_start_time = datetime.now()
    epoch_losses = np.zeros((0,1), dtype=np.float32)
    
    model = model.train()
    epoch_losses=[]
    for images, place_id in tqdm(ds):
        BS, N, ch, h, w = images.shape
        # reshape places and labels
        images = images.view(BS*N, ch, h, w)
        labels = place_id.view(-1)

        optimizer.zero_grad()
        with autocast():
            float_descriptors, quant_descriptors = model(images.to(args.device))
            quant_descriptors = quant_descriptors.cuda()
            loss, miner_outputs = loss_function(quant_descriptors, labels) # Call the loss_function we defined above. 
            index1 = torch.randperm(len(miner_outputs[0]))[:max(int(1.0*len(miner_outputs[0])),1)]  # Change 1.0 to 0.2 in this line and the next line
            index2 = torch.randperm(len(miner_outputs[2]))[:max(int(1.0*len(miner_outputs[2])),1)]  # to use only 20% of features pairs to reduce GPU memory usage.
            sim_posi = torch.sum(float_descriptors[miner_outputs[0][index1]] * float_descriptors[miner_outputs[1][index1]],dim=-1)
            sim_nega = torch.sum(float_descriptors[miner_outputs[2][index2]] * float_descriptors[miner_outputs[3][index2]],dim=-1)
            bin_sim_posi = torch.sum(quant_descriptors[miner_outputs[0][index1]] * quant_descriptors[miner_outputs[1][index1]],dim=-1) / quant_descriptors.shape[-1]
            bin_sim_nega = torch.sum(quant_descriptors[miner_outputs[2][index2]] * quant_descriptors[miner_outputs[3][index2]],dim=-1) / quant_descriptors.shape[-1]
            loss2 = torch.mean(torch.cat([(sim_posi - bin_sim_posi).pow(2),(sim_nega - bin_sim_nega).pow(2)],dim=0))
            loss = loss+0.1*loss2
            del quant_descriptors, float_descriptors

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()     
        
        # Keep track of all losses by appending them to epoch_losses
        batch_loss = loss.item()
        epoch_losses = np.append(epoch_losses, batch_loss)
        del loss
    
    logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                 f"average epoch triplet loss = {epoch_losses.mean():.4f}")
    
    # Compute recalls on validation set
    recalls0, recalls_str0 = test_hashing.test_hashing(args, val_ds0, model)
    logging.info(f"Recalls on val set0 {val_ds0}: {recalls_str0}")
    recalls1, recalls_str1 = test_hashing.test_hashing(args, val_ds1, model)
    logging.info(f"Recalls on val set1 {val_ds1}: {recalls_str1}")
    
    is_best = recalls1[0]+recalls1[1] > best_r1r5
    
    # Save checkpoint, which contains all training parameters
    util.save_checkpoint(args, {"epoch_num": epoch_num, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "recalls": recalls1, "best_r5": best_r1r5,
        "not_improved_num": not_improved_num
    }, is_best, filename="last_model.pth")
    
    # If recall@1+recall@5 did not improve for "many" epochs, stop training
    if is_best:
        logging.info(f"Improved: previous best R@1+R@5 = {best_r1r5:.1f}, current R@1+R@5 = {(recalls1[0]+recalls1[1]):.1f}")
        best_r1r5 = (recalls1[0]+recalls1[1])
        not_improved_num = 0
    else:
        not_improved_num += 1
        logging.info(f"Not improved: {not_improved_num} / {args.patience}: best R@1+R@5 = {best_r1r5:.1f}, current R@1+R@5 = {(recalls1[0]+recalls1[1]):.1f}")
        if not_improved_num >= args.patience:
            logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
            break

logging.info(f"Best R@1+R@5: {best_r1r5:.1f}")
logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

#### Test best model on test set
best_model_state_dict = torch.load(join(args.save_dir, "best_model.pth"))["model_state_dict"]
model.load_state_dict(best_model_state_dict)

logging.debug(f"Loading dataset {args.dataset_name} from folder {args.datasets_folder}")
test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
logging.info(f"Test set: {test_ds}")

recalls, recalls_str = test_hashing.test_hashing(args, test_ds, model, test_method=args.test_method)
logging.info(f"Recalls on {test_ds}: {recalls_str}")