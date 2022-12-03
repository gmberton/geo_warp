
"""
This file contains some functions adapted from the following repositories:
    https://github.com/ignacio-rocco/cnngeometric_pytorch
    for the main architecture;
    
    https://github.com/filipradenovic/cnnimageretrieval-pytorch
    for the GeM layer;
    
    https://github.com/lyakaap/NetVLAD-pytorch
    for the NetVLAD layer.
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import commons


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = gem(x, p=self.p, eps=self.eps)
        x = x.reshape(B, C)
        return x


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class L2Norm(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return feature_L2_norm(x)


def feature_L2_norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1)+epsilon, 0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature.contiguous(), norm)


class FeaturesExtractor(torch.nn.Module):
    """The FeaturesExtractor is composed of two parts: the backbone encoder and the
    pooling/aggregation layer.
    The pooling/aggregation layer is used only to compute global features.
    """
    def __init__(self, arch, pooling):
        super().__init__()
        if arch == "resnet50":
            model = torchvision.models.resnet50(pretrained=True)
            layers = list(model.children())[:-3]
        elif arch == "vgg16":
            model = torchvision.models.vgg16(pretrained=True)
            layers = list(model.features.children())[:-2]
        elif arch == "alexnet":
            model = torchvision.models.alexnet(pretrained=True)
            layers = list(model.features.children())[:-2]
        self.encoder = torch.nn.Sequential(*layers)
        
        self.avgpool = nn.AdaptiveAvgPool2d((15, 15))
        self.l2norm = L2Norm()
        if pooling == "netvlad":
            encoder_dim = commons.get_output_dim(self.encoder)
            self.pool = NetVLAD(dim=encoder_dim)
        elif pooling == "gem":
            self.pool = nn.Sequential(L2Norm(), GeM())
        
    def forward(self, x, f_type="local"):
        x = self.encoder(x)
        if f_type == "local":
            x = self.avgpool(x)
            return self.l2norm(x)
        elif f_type == "global":
            return self.pool(x)
        else:
            raise ValueError(f"Invalid features type: {f_type}")


def compute_similarity(features_a, features_b):
    b, c, h, w = features_a.shape
    features_a = features_a.transpose(2, 3).contiguous().view(b, c, h*w)
    features_b = features_b.view(b, c, h*w).transpose(1, 2)
    features_mul = torch.bmm(features_b, features_a)
    correlation_tensor = features_mul.view(b, h, w, h*w).transpose(2, 3).transpose(1, 2)
    correlation_tensor = feature_L2_norm(F.relu(correlation_tensor))
    return correlation_tensor


class HomographyRegression(nn.Module):
    def __init__(self, output_dim=16, kernel_sizes=[7, 5], channels=[225, 128, 64], padding=0):
        super().__init__()
        assert len(kernel_sizes) == len(channels) - 1, \
            f"In HomographyRegression the number of kernel_sizes must be less than channels, but you said {kernel_sizes} and {channels}"
        nn_modules = []
        for in_channels, out_channels, kernel_size in zip(channels[:-1], channels[1:], kernel_sizes):
            nn_modules.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
            nn_modules.append(nn.BatchNorm2d(out_channels))
            nn_modules.append(nn.ReLU())
        self.conv = nn.Sequential(*nn_modules)
        # Find out output size of last conv, aka the input of the fully connected
        shape = self.conv(torch.ones([2, 225, 15, 15])).shape
        output_dim_last_conv = shape[1] * shape[2] * shape[3]
        self.linear = nn.Linear(output_dim_last_conv, output_dim)
        # Initialize the weights/bias with identity transformation
        init_points = torch.tensor([-1, -1, 1, -1, 1, 1, -1, 1]).type(torch.float)
        init_points = torch.cat((init_points, init_points))
        self.linear.bias.data = init_points
        self.linear.weight.data = torch.zeros_like((self.linear.weight.data))
    
    def forward(self, x):
        B = x.shape[0]
        x = self.conv(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.linear(x)
        return x.reshape(B, 8, 2)


class Network(nn.Module):
    """
    Overview of the network:
    name                 input                                       output
    FeaturesExtractor:   (2B x 3 x H x W)                            (2B x 256 x 15 x 15)
    compute_similarity:  (B x 256 x 15 x 15), (B x 256 x 15 x 15)    (B x 225 x 15 x 15)
    HomographyRegression:(B x 225 x 15 x 15)                         (B x 16)
    """
    
    def __init__(self, features_extractor, homography_regression):
        super().__init__()
        self.features_extractor = features_extractor
        self.homography_regression = homography_regression
    
    def forward(self, operation, args):
        """Compute a forward pass, which can be of different types.
        This "ugly" step of passing the operation as a string has been adapted
        to allow calling different methods through the Network.forward().
        This is because only Network.forward() works on multiple GPUs when using torch.nn.DataParallel().
        
        Parameters
        ----------
        operation : str, defines the type of forward pass.
        args : contains the tensor(s) on which to apply the operation.
        
        """
        assert operation in ["features_extractor", "similarity", "regression", "similarity_and_regression"]
        if operation == "features_extractor":
            if len(args) == 2:
                tensor_images, features_type = args
                return self.features_extractor(tensor_images, features_type)
            else:
                tensor_images = args
                return self.features_extractor(tensor_images, "local")
        
        elif operation == "similarity":
            tensor_img_1, tensor_img_2 = args
            return self.similarity(tensor_img_1, tensor_img_2)
        
        elif operation == "regression":
            similarity_matrix = args
            return self.regression(similarity_matrix)
        
        elif operation == "similarity_and_regression":
            tensor_img_1, tensor_img_2 = args
            similarity_matrix_1to2, similarity_matrix_2to1 = self.similarity(tensor_img_1, tensor_img_2)
            return self.regression(similarity_matrix_1to2), self.regression(similarity_matrix_2to1)
    
    def similarity(self, tensor_img_1, tensor_img_2):
        features_1 = self.features_extractor(tensor_img_1.cuda())
        features_2 = self.features_extractor(tensor_img_2.cuda())
        similarity_matrix_1to2 = compute_similarity(features_1, features_2)
        similarity_matrix_2to1 = compute_similarity(features_2, features_1)
        return similarity_matrix_1to2, similarity_matrix_2to1
    
    def regression(self, similarity_matrix):
        return self.homography_regression(similarity_matrix)


class NetVLAD(nn.Module):
    def __init__(self, num_clusters=64, dim=256, normalize_input=True):
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
    
    def __forward_vlad__(self, x_flatten, soft_assign, N, D):
        vlad = torch.zeros([N, self.num_clusters, D], dtype=x_flatten.dtype, device=x_flatten.device)
        for D in range(self.num_clusters):  # Slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[D:D+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual = residual * soft_assign[:, D:D+1, :].unsqueeze(2)
            vlad[:, D:D+1, :] = residual.sum(dim=-1)
        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.view(N, -1)
        vlad = F.normalize(vlad, p=2, dim=1)
        return vlad
    
    def forward(self, x):
        N, D, H, W = x.shape[:]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)
        x_flatten = x.view(N, D, -1)
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        vlad = self.__forward_vlad__(x_flatten, soft_assign, N, D)
        return vlad
