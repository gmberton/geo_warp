
import os
import sys
import torch
import logging
import argparse
from datetime import datetime

import test
import util
import network
import commons
import dataset_geoloc  # Used for testing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Test parameters
    parser.add_argument("--num_reranked_preds", type=int, default=5,
                        help="number of predictions to re-rank at test time")
    
    # Model parameters
    parser.add_argument("--arch", type=str, default="resnet50",
                        choices=["alexnet", "vgg16", "resnet50"],
                        help="model to use for the encoder")
    parser.add_argument("--pooling", type=str, default="netvlad",
                        choices=["netvlad", "gem"],
                        help="pooling layer used in the baselines")
    parser.add_argument("--kernel_sizes", nargs='+', default=[7, 5, 5, 5, 5, 5],
                        help="size of kernels in conv layers of Homography Regression")
    parser.add_argument("--channels", nargs='+', default=[225, 128, 128, 64, 64, 64, 64],
                        help="num channels in conv layers of Homography Regression")
    
    # Others
    parser.add_argument("--exp_name", type=str, default="default",
                        help="name of generated folders with logs and checkpoints")
    parser.add_argument("--resume_fe", type=str, default=None,
                        help="path to resume for Feature Extractor")
    parser.add_argument("--resume_hr", type=str, default=None,
                        help="path to resume for Homography Regression")
    parser.add_argument("--positive_dist_threshold", type=int, default=25,
                        help="treshold distance for positives (in meters)")
    parser.add_argument("--datasets_folder", type=str, default="../datasets",
                        help="path with the datasets")
    parser.add_argument("--dataset_name", type=str, default="pitts30k",
                        help="name of folder with dataset")
    
    args = parser.parse_args()
    
    # Sanity check
    if len(args.kernel_sizes) != len(args.channels) - 1:
        raise ValueError("len(kernel_sizes) must be equal to len(channels)-1; "
                         f"but you set them to {args.kernel_sizes} and {args.channels}")
    
    # Setup
    output_folder = f"runs/{args.exp_name}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    commons.setup_logging(output_folder, console="info")
    logging.info("python " + " ".join(sys.argv))
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {output_folder}")
    os.makedirs(f"{output_folder}/checkpoints")
    start_time = datetime.now()
    
    ############### MODEL ###############
    features_extractor = network.FeaturesExtractor(args.arch, args.pooling)
    global_features_dim = commons.get_output_dim(features_extractor, args.pooling)
    homography_regression = network.HomographyRegression(kernel_sizes=args.kernel_sizes, channels=args.channels, padding=1)
    
    if args.resume_fe is not None:
        state_dict = torch.load(args.resume_fe)
        features_extractor.load_state_dict(state_dict)
        del state_dict
    else:
        logging.warning("WARNING: --resume_fe is set to None, meaning that the "
                        "Feature Extractor is not initialized!")
    if args.resume_hr is not None:
        state_dict = torch.load(args.resume_hr)
        homography_regression.load_state_dict(state_dict)
        del state_dict
    else:
        logging.warning("WARNING: --resume_hr is set to None, meaning that the "
                        "Homography Regression is not initialized!")
    
    model = network.Network(features_extractor, homography_regression).cuda().eval()
    model = torch.nn.DataParallel(model)
    ############### MODEL ###############
    
    ############### DATASETS & DATALOADERS ###############
    geoloc_test_dataset = dataset_geoloc.GeolocDataset(args.datasets_folder, args.dataset_name, split="test",
                                                       positive_dist_threshold=args.positive_dist_threshold)
    logging.info(f"Geoloc test set: {geoloc_test_dataset}")
    ############### DATASETS & DATALOADERS ###############
    
    ############### TEST ###############
    test_baseline_recalls, test_baseline_recalls_pretty_str, test_baseline_predictions, _, _ = \
        util.compute_features(geoloc_test_dataset, model, global_features_dim)
    logging.info(f"baseline test: {test_baseline_recalls_pretty_str}")
    _, reranked_test_recalls_pretty_str = test.test(model, test_baseline_predictions, geoloc_test_dataset,
                                                    num_reranked_predictions=args.num_reranked_preds,
                                                    recall_values=[1, 5, 10, 20])
    logging.info(f"test after warping - {reranked_test_recalls_pretty_str}")
    ############### TEST ###############
    
