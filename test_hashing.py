import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import time
import datetime

def test_hashing(args, eval_ds, model, test_method="hard_resize", pca=None):
    """Compute features of the given dataset and compute the recalls."""

    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database features for evaluation/testing")
        # For database use "hard_resize", although it usually has no effect because database images have same resolution
        eval_ds.test_method = "hard_resize"
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                        batch_size=args.infer_batch_size, pin_memory=(args.device=="cuda"))
        
        all_features = np.empty((len(eval_ds), args.binary_features_dim), dtype="uint8")

        for inputs, indices in tqdm(database_dataloader, ncols=100):
            float_features, quant_features = model(inputs.to(args.device))
            quant_features = quant_features.cpu().numpy()
            all_features[indices.numpy(), :] = quant_features
        
        logging.debug("Extracting queries features for evaluation/testing")
        queries_infer_batch_size = 1 if test_method == "single_query" else args.infer_batch_size
        eval_ds.test_method = test_method
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device=="cuda"))
        for inputs, indices in tqdm(queries_dataloader, ncols=100):
            float_features, quant_features = model(inputs.to(args.device))
            quant_features = quant_features.cpu().numpy()
            all_features[indices.numpy(), :] = quant_features
                
    queries_features = all_features[eval_ds.database_num:]
    database_features = all_features[:eval_ds.database_num]
    
    binary_queries_features = ((queries_features + 1) // 2).astype("uint8")
    packed_queries_features = np.packbits(binary_queries_features, axis=1)  # Compress into a byte array
    binary_database_features = ((database_features + 1) // 2).astype("uint8")
    packed_database_features = np.packbits(binary_database_features, axis=1)  # Compress into a byte array
    
    faiss_index = faiss.IndexBinaryFlat(args.binary_features_dim)
    faiss_index.add(packed_database_features)
    logging.debug("Calculating recalls")
    distances, predictions = faiss_index.search(packed_queries_features, max(args.recall_values))

    # # Compute single image retrieval efficiency
    # predictions = []
    # elapsed_time = 0
    # for query_feature in packed_queries_features:
    #     start_time = datetime.datetime.now()    #time.time()
    #     distance, prediction = faiss_index.search(query_feature.reshape(1,-1), max(args.recall_values))
    #     end_time = datetime.datetime.now()      #time.time()
    #     elapsed_time += (end_time -start_time).total_seconds()
    #     predictions.append(prediction)
    # predictions = np.vstack(predictions)
    # print(f"Initial Retrieval: {elapsed_time:.6f} s")

    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    # args.recall_values by default is [1, 5, 10, 20]
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
    return recalls, recalls_str