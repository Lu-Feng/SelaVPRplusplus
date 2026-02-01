
import os
import torch
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="SelaVPR++",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=120,
                        help="Number of triplets (query, pos, negs) in a batch. Each triplet consists of 12 images")
    parser.add_argument("--infer_batch_size", type=int, default=16,
                        help="Batch size for inference (caching and testing)")
    parser.add_argument("--criterion", type=str, default='triplet', help='loss to be used',
                        choices=["triplet", "sare_ind", "sare_joint"])
    parser.add_argument("--margin", type=float, default=0.1,
                        help="margin for the triplet loss")
    parser.add_argument("--epochs_num", type=int, default=25,
                        help="number of epochs to train for")
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.0004, help="_")
    parser.add_argument("--lr_crn_net", type=float, default=5e-4, help="Learning rate to finetune pretrained network when using CRN")
    parser.add_argument("--optim", type=str, default="adam", help="_", choices=["adam", "sgd", "adamw"])
    # parser.add_argument("--cache_refresh_rate", type=int, default=1000,
    #                     help="How often to refresh cache, in number of queries")
    # parser.add_argument("--queries_per_epoch", type=int, default=5000,
    #                     help="How many queries to consider for one epoch. Must be multiple of cache_refresh_rate")
    # parser.add_argument("--negs_num_per_query", type=int, default=10,
    #                     help="How many negatives to consider per each query in the loss")
    # parser.add_argument("--neg_samples_num", type=int, default=1000,
    #                     help="How many negatives to use to compute the hardest ones")
    # parser.add_argument("--mining", type=str, default="partial", choices=["partial", "full", "random", "msls_weighted"])
    # Model parameters
    parser.add_argument("--backbone", type=str, default="dinov2-large",
                        choices=["dinov2-large","dinov2-base"], help="_")
    parser.add_argument("--aggregation", type=str, default="gem", choices=["gem", "boq", "salad"])
    parser.add_argument('--pca_dim', type=int, default=None, help="PCA dimension (number of principal components). If None, PCA is not used.")
    # parser.add_argument('--fc_output_dim', type=int, default=None,
    #                     help="Output dimension of fully connected layer. If None, don't use a fully connected layer.")
    parser.add_argument('--hashing', action='store_true', help="_")
    parser.add_argument("--rerank", action="store_true", help="_")
    # Initialization parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--foundation_model_path", type=str, default=None,
                        help="Path to load foundation model checkpoint.")
    parser.add_argument("--training_dataset", type=str, default="gsv_cities", choices=["gsv_cities", "unified_dataset"],
                        help="Dataset for model training")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to load checkpoint from, for resuming training or testing.")
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=8, help="num_workers for all dataloaders")
    parser.add_argument('--resize', type=int, default=[322, 322], nargs=2, help="Resizing shape for images (HxW).")
    parser.add_argument('--test_method', type=str, default="hard_resize",
                        choices=["hard_resize", "single_query", "central_crop", "five_crops", "nearest_crop", "maj_voting"],
                        help="This includes pre/post-processing methods and prediction refinement")
    parser.add_argument("--majority_weight", type=float, default=0.01, 
                        help="only for majority voting, scale factor, the higher it is the more importance is given to agreement")
    parser.add_argument("--val_positive_dist_threshold", type=int, default=25, help="_")
    parser.add_argument("--train_positives_dist_threshold", type=int, default=10, help="_")
    parser.add_argument('--recall_values', type=int, default=[1, 5, 10, 100], nargs="+",
                        help="Recalls to be computed, such as R@5.")
    parser.add_argument("--rerank_num", type=int, default=100, help="_")
    # Data augmentation parameters
    parser.add_argument("--brightness", type=float, default=None, help="_")
    parser.add_argument("--contrast", type=float, default=None, help="_")
    parser.add_argument("--saturation", type=float, default=None, help="_")
    parser.add_argument("--hue", type=float, default=None, help="_")
    parser.add_argument("--rand_perspective", type=float, default=None, help="_")
    parser.add_argument("--horizontal_flip", action='store_true', help="_")
    parser.add_argument("--random_resized_crop", type=float, default=None, help="_")
    parser.add_argument("--random_rotation", type=float, default=None, help="_")
    # Paths parameters
    parser.add_argument("--datasets_folder", type=str, default=None, help="Path with all datasets")
    parser.add_argument("--dataset_name", type=str, default="pitts30k", help="Relative path of the dataset")
    parser.add_argument("--pca_dataset_folder", type=str, default=None,
                        help="Path with images to be used to compute PCA (ie: pitts30k/images/train")
    parser.add_argument("--save_dir", type=str, default="default",
                        help="Folder name of the current run (saved in ./logs/)")
    args = parser.parse_args()
    
    if args.datasets_folder == None:
        try:
            args.datasets_folder = os.environ['DATASETS_FOLDER']
        except KeyError:
            raise Exception("You should set the parameter --datasets_folder or export " +
                            "the DATASETS_FOLDER environment variable as such \n" +
                            "export DATASETS_FOLDER=../datasets_vg/datasets")
    
    # if args.queries_per_epoch % args.cache_refresh_rate != 0:
    #     raise ValueError("Ensure that queries_per_epoch is divisible by cache_refresh_rate, " +
    #                      f"because {args.queries_per_epoch} is not divisible by {args.cache_refresh_rate}")
    
    if torch.cuda.device_count() >= 2 and args.criterion in ['sare_joint', "sare_ind"]:
        raise NotImplementedError("SARE losses are not implemented for multiple GPUs, " +
                                  f"but you're using {torch.cuda.device_count()} GPUs and {args.criterion} loss.")
    
    if args.pca_dim != None and args.pca_dataset_folder == None:
        raise ValueError("Please specify --pca_dataset_folder when using pca")
    
    return args

