
import os
import sys
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torchvision.transforms.functional import hflip

import test
import util
import network
import commons
import dataset_qp  # Used for weakly supervised losses, it yields query-positive pairs
import dataset_warp  # Used to train the warping regressiong module in a self-supervised fashion
import dataset_geoloc  # Used for testing


def hor_flip(points):
    """Flip points horizontally.
    
    Parameters
    ----------
    points : torch.Tensor of shape [B, 8, 2]
    """
    new_points = torch.zeros_like(points)
    new_points[:, 0::2, :] = points[:, 1::2, :]
    new_points[:, 1::2, :] = points[:, 0::2, :]
    new_points[:, :, 0] *= -1
    return new_points


def to_cuda(list_):
    """Move to cuda all items of the list."""
    return [item.cuda() for item in list_]


def compute_loss(loss, weight):
    """Compute loss and gradients separately for each loss, and free the
    computational graph to reduce memory consumption.
    """
    loss *= weight
    loss.backward()
    return loss.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate")
    parser.add_argument("--optim", type=str,  default="sgd",
                        choices=["adam", "sgd"],
                        help="optimizer")
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="epochs")
    parser.add_argument("--iterations_per_epoch", type=int, default=500,
                        help="how many iterations each epoch should last")
    parser.add_argument("--k", type=int, default=0.6,
                        help="parameter k, defining the difficulty of ss training data")
    parser.add_argument("--ss_w", type=float, default=1,
                        help="weight of self-supervised loss")
    parser.add_argument("--consistency_w", type=float, default=0.1,
                        help="weight of consistency loss")
    parser.add_argument("--features_wise_w", type=float, default=10,
                        help="weight of features-wise loss")
    parser.add_argument("--qp_threshold", type=float, default=1.2,
                        help="Threshold distance (in features space) for query-positive pairs")
    parser.add_argument("--batch_size_ss", type=int, default=16,
                        help="batch size for self-supervised loss")
    parser.add_argument("--batch_size_consistency", type=int, default=16,
                        help="batch size for consistency loss")
    parser.add_argument("--batch_size_features_wise", type=int, default=16,
                        help="batch size for features-wise loss")
    parser.add_argument("--ss_num_workers", type=int, default=8,
                        help="num_workers for self-supervised loss")
    parser.add_argument("--qp_num_workers", type=int, default=4,
                        help="num_workers for weakly supervised losses")
    
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
    commons.setup_logging(output_folder)
    logging.info("python " + " ".join(sys.argv))
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {output_folder}")
    os.makedirs(f"{output_folder}/checkpoints")
    start_time = datetime.now()
    
    ############### MODEL ###############
    features_extractor = network.FeaturesExtractor(args.arch, args.pooling)
    global_features_dim = commons.get_output_dim(features_extractor, args.pooling)
    
    if args.resume_fe is not None:
        state_dict = torch.load(args.resume_fe)
        features_extractor.load_state_dict(state_dict)
        del state_dict
    else:
        logging.warning("WARNING: --resume_fe is set to None, meaning that the "
                        "Feature Extractor is not initialized!")
    
    homography_regression = network.HomographyRegression(kernel_sizes=args.kernel_sizes, channels=args.channels, padding=1)
    model = network.Network(features_extractor, homography_regression).cuda().eval()
    model = torch.nn.DataParallel(model)
    ############### MODEL ###############
    
    ############### DATASETS & DATALOADERS ###############
    geoloc_train_dataset = dataset_geoloc.GeolocDataset(args.datasets_folder, args.dataset_name, split="train",
                                                        positive_dist_threshold=args.positive_dist_threshold)
    logging.info(f"Geoloc train set: {geoloc_train_dataset}")
    geoloc_test_dataset = dataset_geoloc.GeolocDataset(args.datasets_folder, args.dataset_name, split="test",
                                                       positive_dist_threshold=args.positive_dist_threshold)
    logging.info(f"Geoloc test set: {geoloc_test_dataset}")
    
    ss_dataset = dataset_warp.HomographyDataset(root_path=f"{args.datasets_folder}/{args.dataset_name}/images/train", k=args.k)
    ss_dataloader = commons.InfiniteDataLoader(ss_dataset, shuffle=True, batch_size=args.batch_size_ss,
                                               num_workers=args.ss_num_workers, pin_memory=True, drop_last=True)
    ss_data_iter = iter(ss_dataloader)
    
    if args.consistency_w != 0 or args.features_wise_w != 0:
        dataset_qp = dataset_qp.DatasetQP(model, global_features_dim, geoloc_train_dataset, qp_threshold=args.qp_threshold)
        dataloader_qp = commons.InfiniteDataLoader(dataset_qp, shuffle=True,
                                                   batch_size=max(args.batch_size_consistency, args.batch_size_features_wise),
                                                   num_workers=args.qp_num_workers, pin_memory=True, drop_last=True)
        data_iter_qp = iter(dataloader_qp)
    ############### DATASETS & DATALOADERS ###############
    
    ############### LOSS & OPTIMIZER ###############
    mse = torch.nn.MSELoss()
    if args.optim == "adam":
        optim = torch.optim.Adam(homography_regression.parameters(), lr=args.lr)
    elif args.optim == "sgd":
        optim = torch.optim.SGD(homography_regression.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)
    ############### LOSS & OPTIMIZER ###############
    
    ############### TRAIN ###############
    for epoch in range(args.n_epochs):
        
        homography_regression = homography_regression.train()
        epoch_losses = np.zeros((0, 3), dtype=np.float32)
        
        for iteration in tqdm(range(args.iterations_per_epoch), desc=f"Train epoch {epoch}", ncols=100):
            warped_img_1, warped_img_2, warped_intersection_points_1, warped_intersection_points_2 = to_cuda(next(ss_data_iter))
            if args.consistency_w != 0 or args.features_wise_w != 0:
                queries, positives = to_cuda(next(data_iter_qp))
            
            with torch.no_grad():
                similarity_matrix_1to2, similarity_matrix_2to1 = model("similarity", [warped_img_1, warped_img_2])
                if args.consistency_w != 0:
                    queries_cons = queries[:args.batch_size_consistency]
                    positives_cons = positives[:args.batch_size_consistency]
                    similarity_matrix_q2p, similarity_matrix_p2q = model("similarity", [queries_cons, positives_cons])
                    fl_similarity_matrix_q2p, fl_similarity_matrix_p2q = model("similarity", [hflip(queries_cons), hflip(positives_cons)])
                    del queries_cons, positives_cons
            
            optim.zero_grad()
            
            # ss_loss
            if args.ss_w != 0:
                pred_warped_intersection_points_1 = model("regression", similarity_matrix_1to2)
                pred_warped_intersection_points_2 = model("regression", similarity_matrix_2to1)
                ss_loss = (mse(pred_warped_intersection_points_1[:, :4], warped_intersection_points_1) +
                           mse(pred_warped_intersection_points_1[:, 4:], warped_intersection_points_2) +
                           mse(pred_warped_intersection_points_2[:, :4], warped_intersection_points_2) +
                           mse(pred_warped_intersection_points_2[:, 4:], warped_intersection_points_1))
                ss_loss = compute_loss(ss_loss, args.ss_w)
                del pred_warped_intersection_points_1, pred_warped_intersection_points_2
            else:
                ss_loss = 0
            
            # consistency_loss
            if args.consistency_w != 0:
                pred_intersection_points_q2p = model("regression", similarity_matrix_q2p)
                pred_intersection_points_p2q = model("regression", similarity_matrix_p2q)
                fl_pred_intersection_points_q2p = model("regression", fl_similarity_matrix_q2p)
                fl_pred_intersection_points_p2q = model("regression", fl_similarity_matrix_p2q)
                four_predicted_points = [
                    torch.cat((pred_intersection_points_q2p[:, 4:], pred_intersection_points_q2p[:, :4]), 1),
                    pred_intersection_points_p2q,
                    hor_flip(torch.cat((fl_pred_intersection_points_q2p[:, 4:], fl_pred_intersection_points_q2p[:, :4]), 1)),
                    hor_flip(fl_pred_intersection_points_p2q)
                ]
                four_predicted_points_centroids = torch.cat([p[None] for p in four_predicted_points]).mean(0).detach()
                consistency_loss = sum([mse(pred, four_predicted_points_centroids) for pred in four_predicted_points])
                consistency_loss = compute_loss(consistency_loss, args.consistency_w)
                del pred_intersection_points_q2p, pred_intersection_points_p2q
                del fl_pred_intersection_points_q2p, fl_pred_intersection_points_p2q
                del four_predicted_points
            else:
                consistency_loss = 0
            
            # features_wise_loss
            if args.features_wise_w != 0:
                queries_fw = queries[:args.batch_size_features_wise]
                positives_fw = positives[:args.batch_size_features_wise]
                # Add random weights to avoid numerical instability
                random_weights = (torch.rand(args.batch_size_features_wise, 4)**0.1).cuda()
                w_queries, w_positives, _, _ = dataset_warp.compute_warping(model, queries_fw, positives_fw, weights=random_weights)
                f_queries = model("features_extractor", [w_queries, "local"])
                f_positives = model("features_extractor", [w_positives, "local"])
                features_wise_loss = compute_loss(mse(f_queries, f_positives), args.features_wise_w)
                del queries, positives, queries_fw, positives_fw, w_queries, w_positives, f_queries, f_positives
            else:
                features_wise_loss = 0
            
            epoch_losses = np.concatenate((epoch_losses, np.array([[ss_loss, consistency_loss, features_wise_loss]])))
            optim.step()
        
        epoch_losses_means = epoch_losses.mean()
        def format_(array): return " - ".join([f"{e:.4f}" for e in array])
        logging.debug(f"epoch: {epoch:>3} / {args.n_epochs}, losses: {epoch_losses_means:.4f} ({format_(epoch_losses.mean(0))})")
        torch.save(homography_regression.state_dict(),
                   f"{output_folder}/checkpoints/homography_regression_{epoch:03d}.torch")
        
        logging.debug(f"Current total loss = {epoch_losses_means:.4f}")
    ############### TRAIN ###############
    
    ############### TEST ###############
    logging.info(f"The training is over in {str(datetime.now() - start_time)[:-7]}, now it's test time")
    
    homography_regression = homography_regression.eval()
    
    test_baseline_recalls, test_baseline_recalls_pretty_str, test_baseline_predictions, _, _ = \
        util.compute_features(geoloc_test_dataset, model, global_features_dim)
    logging.info(f"baseline test: {test_baseline_recalls_pretty_str}")
    _, reranked_test_recalls_pretty_str = test.test(model, test_baseline_predictions, geoloc_test_dataset,
                                                    num_reranked_predictions=args.num_reranked_preds,
                                                    recall_values=[1, 5, 10, 20])
    logging.info(f"test after warping - {reranked_test_recalls_pretty_str}")
    ############### TEST ###############
