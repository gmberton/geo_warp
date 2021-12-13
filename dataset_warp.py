
import torch
import kornia
import random
from glob import glob
from shapely.geometry import Polygon

import datasets_util


def get_random_trapezoid(k=1):
    """Get the points (with shape [4, 2] of a random trapezoid with two vertical sides.
    With k=0, the trapezoid is a rectangle with points:
        [[-1., -1.], [ 1., -1.], [ 1.,  1.], [-1.,  1.]]
    
    Parameters
    ----------
    k : float, with 0 <= k <= 1, indicates the "randomness" of the trapezoid.
        The higher k, the more "random" the trapezoid.
    """
    assert 0 <= k <= 1
    def rand(k):
        return 1 - (random.random() * k)
    left = -rand(k); right = rand(k)
    return torch.tensor([(left, -rand(k)), (right, -rand(k)), (right, rand(k)), (left, rand(k))])


def compute_warping(model, tensor_img_1, tensor_img_2, weights=None):
    """Computes the pairwise warping of two (batches of) images, using a given model.
    Given that the operations in the model is not commutative (i.e. the order of 
    the tensor matters), this function computes the mean passing the tensor images
    in both orders.
    
    Parameters
    ----------
    model : network.Network, used to compute the homography.
    tensor_img_1 : torch.Tensor, the query images, with shape [B, 3, H, W].
    tensor_img_2 : torch.Tensor, the gallery images, with shape [B, 3, H, W].
    weights : torch.Tensor, random weights to avoid numerical instability,
        usually they're not needed.

    Returns
    -------
    warped_tensor_img_1 : torch.Tensor, the warped query images, with shape [B, 3, H, W].
    warped_tensor_img_2 : torch.Tensor, the warped gallery images, with shape [B, 3, H, W].
    mean_pred_points_1 : torch.Tensor, the predicted points, used to compute homography
        on the query images, with shape [B, 4, 2]
    mean_pred_points_2 : torch.Tensor, the predicted points, used to compute homography
        on the gallery images, with shape [B, 4, 2]
    """
    # Get both predictions
    pred_points_1to2, pred_points_2to1 = model("similarity_and_regression", [tensor_img_1, tensor_img_2])
    # Average them
    mean_pred_points_1 = (pred_points_1to2[:,:4] + pred_points_2to1[:,4:]) / 2
    mean_pred_points_2 = (pred_points_1to2[:,4:] + pred_points_2to1[:,:4]) / 2
    # Apply the homography
    warped_tensor_img_1, _ = warp_images(tensor_img_1, mean_pred_points_1, weights)
    warped_tensor_img_2, _ = warp_images(tensor_img_2, mean_pred_points_2, weights)
    return warped_tensor_img_1, warped_tensor_img_2, mean_pred_points_1, mean_pred_points_2


def warp_images(tensor_img, warping_points, weights=None):
    """Apply the homography to a batch of images using the points specified in warping_points.
    
    Parameters
    ----------
    tensor_img : torch.Tensor, the images, with shape [B, 3, H, W].
    warping_points : torch.Tensor, the points used to compute homography, with shape [B, 4, 2]
    weights : torch.Tensor, random weights to avoid numerical instability,
        usually they're not needed.
    
    Returns
    -------
    warped_images : torch.Tensor, the warped images, with shape [B, 3, H, W].
    theta : theta matrix of the homography, usually not needed, with shape [B, 3, 3].
    """
    B, C, H, W = tensor_img.shape
    assert warping_points.shape == torch.Size([B, 4, 2])
    rectangle_points = torch.repeat_interleave(get_random_trapezoid(k=0).unsqueeze(0), B, 0)
    rectangle_points = rectangle_points.to(tensor_img.device)
    # NB for older versions of kornia use kornia.find_homography_dlt
    theta = kornia.geometry.homography.find_homography_dlt(rectangle_points, warping_points, weights)
    # NB for older versions of kornia use kornia.homography_warp
    warped_images = kornia.geometry.homography_warp(tensor_img, theta, dsize=(H, W))
    return warped_images, theta


def get_random_homographic_pair(source_img, k, is_debugging=False):
    """Given a source image, returns a pair of warped images generate in a self-supervised
    fashion, together with the points used to generate the projections (homography).
    
    Parameters
    ----------
    source_img : torch.Tensor, with shape [3, H, W].
    k : float, the k parameter indicates how "difficult" the generated pair is,
        i.e. it's an upper bound on how strong the warping can be.
    is_debugging : bool, if True return extra information
    
    """
    
    # Compute two random trapezoids and their intersection
    trap_points_1 = get_random_trapezoid(k)
    trap_points_2 = get_random_trapezoid(k)
    points_trapezoids = torch.cat((trap_points_1.unsqueeze(0), trap_points_2.unsqueeze(0)))
    trap_1 = Polygon(trap_points_1)
    trap_2 = Polygon(trap_points_2)
    intersection = trap_2.intersection(trap_1)
    # Some operations to get the intersection points as a torch.Tensor of shape [4, 2]
    list_x, list_y = intersection.exterior.coords.xy
    a3, d3 = sorted(list(set([(x,y) for x,y in zip(list_x, list_y) if x == min(list_x)])))
    b3, c3 = sorted(list(set([(x,y) for x,y in zip(list_x, list_y) if x == max(list_x)])))
    intersection_points = torch.tensor([a3, b3, c3, d3]).type(torch.float)
    intersection_points = torch.repeat_interleave(intersection_points.unsqueeze(0), 2, 0)
    
    image_repeated_twice = torch.repeat_interleave(source_img.unsqueeze(0), 2, 0)
    
    warped_images, theta = warp_images(image_repeated_twice, points_trapezoids)
    theta = torch.inverse(theta)
    # Compute positions of intersection_points projected on the warped images
    xs = [(theta[:,0,0]*intersection_points[:,p,0] + theta[:,0,1]*intersection_points[:,p,1] + theta[:,0,2])/
          (theta[:,2,0]*intersection_points[:,p,0] + theta[:,2,1]*intersection_points[:,p,1] + theta[:,2,2]) for p in range(4)]
    ys = [(theta[:,1,0]*intersection_points[:,p,0] + theta[:,1,1]*intersection_points[:,p,1] + theta[:,1,2])/
          (theta[:,2,0]*intersection_points[:,p,0] + theta[:,2,1]*intersection_points[:,p,1] + theta[:,2,2]) for p in range(4)]
    # Refactor the projected intersection points as a torch.Tensor with shape [2, 4, 2]
    warped_intersection_points = torch.cat((torch.stack(xs).T.reshape(2,4,1), torch.stack(ys).T.reshape(2,4,1)), 2)
    if is_debugging:
        warped_images_intersection, inverse_theta = warp_images(warped_images, warped_intersection_points)
        return (source_img[0], *warped_images, *warped_images_intersection), (theta, inverse_theta), \
               (*points_trapezoids, *intersection_points, *warped_intersection_points)
    else:
        return warped_images[0], warped_images[1], warped_intersection_points[0], warped_intersection_points[1]


class HomographyDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, k=0.1, is_debugging=False):
        super().__init__()
        self.images_paths = sorted(glob(f"{root_path}/**/*.jpg", recursive=True))
        self.k = k
        self.is_debugging = is_debugging
    def __getitem__(self, index):
        image_path = self.images_paths[index]
        source_img = datasets_util.open_image_and_apply_transform(image_path)
        return get_random_homographic_pair(source_img, self.k, is_debugging=self.is_debugging)
    def __len__(self):
        return len(self.images_paths)

