
import cv2
import torch
import argparse
import numpy as np
from skimage import io

import dataset_warp
import datasets_util


def tensor_to_numpy(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize a tensor image and transform it to a numpy image."""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    tensor = torch.clamp(tensor, 0, 1) * 255
    image = np.array(tensor)
    image = np.transpose(image, (1, 2, 0))
    image = np.ascontiguousarray(image)
    return image


def draw_quadrilateral(image, points=None, color=(0, 255, 0), thickness=20):
    """Draw on image the quadrilateral defined by points."""
    if points is None:  # If points is None, draw on the edges of the image.
        points = torch.tensor([[-1, -1], [ 1, -1], [ 1,  1], [-1,  1]])
        points = points.type(torch.float)
    h, w, _ = image.shape
    points = np.array(points)
    points = (points + 1) / 2
    points[:,0] *= w
    points[:,1] *= h
    for i in range(3):
        cv2.line(image, (points[i,0], points[i,1]), (points[i+1,0], points[i+1,1]),
                 color, thickness=thickness)
    image = cv2.line(image, (points[3,0], points[3,1]), (points[0,0], points[0,1]),
                     color, thickness=thickness)
    return image


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--image_path", type=str, default="data/example.jpg",
                    help="path of image")
parser.add_argument("--k", type=int, default=0.8,
                    help="parameter k, defining the difficulty of ss training data")
args = parser.parse_args()

img_source_img = datasets_util.open_image_and_apply_transform(args.image_path)
images, _, points = dataset_warp.get_random_homographic_pair(img_source_img, args.k, is_debugging=True)

images = [tensor_to_numpy(i) for i in images]
# The two proj_intersections on the img_source image are equal
(t_x, t_y, inters, inters, t_a, t_b) = points

(img_source, proj_a, proj_b, proj_intersection, _) = images

# On each image, draw the colored quadrilateral
img_source = draw_quadrilateral(img_source, t_x, color=(255,174,133))
img_source = draw_quadrilateral(img_source, t_y, color=(154,179,255))
img_source = draw_quadrilateral(img_source, inters)
proj_a = draw_quadrilateral(proj_a, t_a)
proj_a = draw_quadrilateral(proj_a, color=(255,174,133))
proj_b = draw_quadrilateral(proj_b, t_b)
proj_b = draw_quadrilateral(proj_b, color=(154,179,255))
proj_intersection = draw_quadrilateral(proj_intersection)

img_source = img_source.astype(np.uint8)
proj_a = proj_a.astype(np.uint8)
proj_b = proj_b.astype(np.uint8)
proj_intersection = proj_intersection.astype(np.uint8)

io.imsave("data/ss_img_source.jpg", img_source)
io.imsave("data/ss_proj_a.jpg", proj_a)
io.imsave("data/ss_proj_b.jpg", proj_b)
io.imsave("data/ss_proj_inters.jpg", proj_intersection)

