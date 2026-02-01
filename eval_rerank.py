
import os
import sys
import torch
import parser
import logging
import sklearn
from os.path import join
from datetime import datetime
from torch.utils.model_zoo import load_url

import test_rerank
import util
import commons
import datasets_ws
from model import network
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

######################################### SETUP #########################################
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("test", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")

######################################### MODEL #########################################
model = network.GeoLocalizationNet(args)
model = model.to(args.device)

if args.resume is not None:
    logging.info(f"Resuming model from {args.resume}")
    model = util.resume_model(args, model)
# Enable DataParallel after loading checkpoint, otherwise doing it before
# would append "module." in front of the keys of the state dict triggering errors
model = torch.nn.DataParallel(model)

if args.backbone == "dinov2-base" and args.aggregation == "gem":
    args.features_dim=2048
elif args.backbone == "dinov2-large" and args.aggregation == "gem":
    args.features_dim=4096
elif args.aggregation == "boq":
    args.features_dim=12288
elif args.aggregation == "salad":
    args.features_dim=8448
args.binary_features_dim=512
logging.info(f"Output dimension of the binary features is {args.binary_features_dim}")
logging.info(f"Output dimension of the float features for reranking is {args.features_dim}")

######################################### DATASETS #########################################
test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
logging.info(f"Test set: {test_ds}")

######################################### TEST on TEST SET #########################################
recalls, recalls_str = test_rerank.test_rerank(args, test_ds, model, args.test_method)
logging.info(f"Recalls on {test_ds}: {recalls_str}")

logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")