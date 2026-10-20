"""Define MicroNet architecture.

Raza, SEA et al., “Micro-Net: A unified model for segmentation of
various objects in microscopy images,” Medical Image Analysis,
Dec. 2018, vol. 52, p. 160-173.

"""
from __future__ import annotations
from pathlib import Path
from cellnura_release.baseline_runtime import resource, real_classes, test_image_ids, frozen_roles, setting, inference_only
from collections import OrderedDict
import numpy as np
import torch
from scipy import ndimage
from skimage import morphology
from torch import nn
from torch.nn import functional
import cv2

def group1_forward_branch(layer: nn.Module, in_tensor: torch.Tensor, resized_feat: torch.Tensor) -> torch.Tensor:
    """Define group 1 connections.

    Args:
        layer (torch.nn.Module):
            Network layer.
        in_tensor (torch.Tensor):
            Input tensor.
        resized_feat (torch.Tensor):
            Resized input.

    Returns:
        torch.Tensor:
            Output of group 1 layer.

    """
    a = layer['conv1'](in_tensor)
    a = layer['conv2'](a)
    a = layer['pool'](a)
    b = layer['conv3'](resized_feat)
    b = layer['conv4'](b)
    return torch.cat(tensors=(a, b), dim=1)

def group2_forward_branch(layer: nn.Module, in_tensor: torch.Tensor) -> torch.Tensor:
    """Define group 1 connections.

    Args:
        layer (torch.nn.Module):
            Network layer.
        in_tensor (torch.Tensor):
            Input tensor.

    Returns:
        torch.Tensor:
            Output of group 1 layer.

    """
    a = layer['conv1'](in_tensor)
    return layer['conv2'](a)

def group3_forward_branch(layer: nn.Module, main_feat: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    """Define group 1 connections.

    Args:
        layer (torch.nn.Module):
            Network layer.
        main_feat (torch.Tensor):
            Input tensor.
        skip (torch.Tensor):
            Skip connection.

    Returns:
        torch.Tensor: Output of group 1 layer.

    """
    a = layer['up1'](main_feat)
    a = layer['conv1'](a)
    a = layer['conv2'](a)
    b1 = layer['up2'](a)
    b2 = layer['up3'](skip)
    b = torch.cat(tensors=(b1, b2), dim=1)
    return layer['conv3'](b)

def group4_forward_branch(layer: nn.Module, in_tensor: torch.Tensor) -> torch.Tensor:
    """Define group 1 connections.

    Args:
        layer (torch.nn.Module):
            Network layer.
        in_tensor (torch.Tensor):
            Input tensor.

    Returns:
        torch.Tensor: Output of group 1 layer.

    """
    a = layer['up1'](in_tensor)
    return layer['conv1'](a)

def group1_arch_branch(in_ch: int, resized_in_ch: int, out_ch: int) -> nn.ModuleDict:
    """Group1 branch for MicroNet.

    Args:
        in_ch (int):
            Number of input channels.
        resized_in_ch (int):
            Number of input channels from resized input.
        out_ch (int):
            Number of output channels.

    Returns:
        :class:`torch.nn.ModuleDict`:
            An output of type :class:`torch.nn.ModuleDict`

    """
    module_dict = OrderedDict()
    module_dict['conv1'] = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=0, bias=True), nn.Tanh(), nn.BatchNorm2d(out_ch))
    module_dict['conv2'] = nn.Sequential(nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=0, bias=True), nn.Tanh())
    module_dict['pool'] = nn.MaxPool2d(2, padding=0)
    module_dict['conv3'] = nn.Sequential(nn.Conv2d(resized_in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=0, bias=True), nn.Tanh(), nn.BatchNorm2d(out_ch))
    module_dict['conv4'] = nn.Sequential(nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=0, bias=True), nn.Tanh())
    return nn.ModuleDict(module_dict)

def group2_arch_branch(in_ch: int, out_ch: int) -> nn.ModuleDict:
    """Group2 branch for MicroNet.

    Args:
        in_ch (int):
            Number of input channels.
        out_ch (int):
            Number of output channels.

    Returns:
        torch.nn.ModuleDict:
            An output of type :class:`torch.nn.ModuleDict`

    """
    module_dict = OrderedDict()
    module_dict['conv1'] = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=0, bias=True), nn.Tanh())
    module_dict['conv2'] = nn.Sequential(nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=0, bias=True), nn.Tanh())
    return nn.ModuleDict(module_dict)

def group3_arch_branch(in_ch: int, skip: int, out_ch: int) -> nn.ModuleDict:
    """Group3 branch for MicroNet.

    Args:
        in_ch (int):
            Number of input channels.
        skip (int):
            Number of channels for the skip connection.
        out_ch (int):
            Number of output channels.

    Returns:
        torch.nn.ModuleDict:
            An output of type :class:`torch.nn.ModuleDict`

    """
    module_dict = OrderedDict()
    module_dict['up1'] = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(2, 2), stride=(2, 2))
    module_dict['conv1'] = nn.Sequential(nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=0, bias=True), nn.Tanh())
    module_dict['conv2'] = nn.Sequential(nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=0, bias=True), nn.Tanh())
    module_dict['up2'] = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=(5, 5), stride=(1, 1))
    module_dict['up3'] = nn.ConvTranspose2d(skip, out_ch, kernel_size=(5, 5), stride=(1, 1))
    module_dict['conv3'] = nn.Sequential(nn.Conv2d(2 * out_ch, out_ch, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True), nn.Tanh())
    return nn.ModuleDict(module_dict)

def group4_arch_branch(in_ch: int, out_ch: int, up_kernel: tuple[int, int]=(2, 2), up_strides: tuple[int, int]=(2, 2), activation: str='tanh') -> nn.ModuleDict:
    """Group4 branch for MicroNet.

    This branch defines architecture for decoder and
    provides input for the auxiliary and main output branch.

    Args:
        in_ch (int):
            Number of input channels.
        out_ch (int):
            Number of output channels.
        up_kernel (tuple of int):
            Kernel size for
            :class:`torch.nn.ConvTranspose2d`.
        up_strides (tuple of int):
            Stride size for
            :class:`torch.nn.ConvTranspose2d`.
        activation (str):
            Activation function, default="tanh".

    Returns:
        torch.nn.ModuleDict:
            An output of type :class:`torch.nn.ModuleDict`

    """
    activation = nn.ReLU() if activation == 'relu' else nn.Tanh()
    module_dict = OrderedDict()
    module_dict['up1'] = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=up_kernel, stride=up_strides)
    module_dict['conv1'] = nn.Sequential(nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=0, bias=True), activation)
    return nn.ModuleDict(module_dict)

def out_arch_branch(in_ch: int, num_output_channels: int=2, activation: str='softmax') -> torch.nn.Sequential:
    """Group5 branch for MicroNet.

    This branch defines architecture for auxiliary and the main output.

    Args:
        in_ch (int):
            Number of input channels.
        num_output_channels (int):
            Number of output channels. default=2.
        activation (str):
            Activation function, default="softmax".

    Returns:
        torch.nn.Sequential:
            An output of type :class:`torch.nn.Sequential`

    """
    activation = nn.ReLU() if activation == 'relu' else nn.Softmax()
    return nn.Sequential(nn.Dropout2d(p=0.5), nn.Conv2d(in_ch, num_output_channels, kernel_size=(3, 3), stride=(1, 1), padding=0, bias=True), activation)

class MicroNet(nn.Module):
    """Initialize MicroNet [1].

    The following models have been included in tiatoolbox:

    1. `micronet-consep`:
        This is trained on `CoNSeP dataset
        <https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/>`_ The
        model is retrained in torch as the original model with results
        on CoNSeP [2] was trained in TensorFlow.

    The tiatoolbox model should produce the following results on the CoNSeP dataset:

    .. list-table:: MicroNet performance
       :widths: 15 15 15 15 15 15 15
       :header-rows: 1

       * - Model name
         - Data set
         - DICE
         - AJI
         - DQ
         - SQ
         - PQ
       * - micronet-consep
         - CoNSeP
         - 0.80
         - 0.49
         - 0.62
         - 0.75
         - 0.47

    Args:
        num_input_channels (int):
            Number of channels in input. default=3.
        num_output_channels (int):
            Number of output channels. default=2.
        out_activation (str):
            Activation to use at the output. MapDe inherits MicroNet
            but uses ReLU activation.

    References:
        [1] Raza, Shan E Ahmed, et al. "Micro-Net: A unified model for
        segmentation of various objects in microscopy images."
        Medical image analysis 52 (2019): 160-173.

        [2] Graham, Simon, et al. "Hover-net: Simultaneous segmentation
        and classification of nuclei in multi-tissue histology images."
        Medical Image Analysis 58 (2019): 101563.

    """

    def __init__(self: MicroNet, num_input_channels: int=3, num_output_channels: int=2, out_activation: str='softmax') -> None:
        """Initialize :class:`MicroNet`."""
        super().__init__()
        if num_output_channels < 2:
            msg = 'Number of classes should be >=2.'
            raise ValueError(msg)
        self.__num_output_channels = num_output_channels
        self.in_ch = num_input_channels
        module_dict = OrderedDict()
        module_dict['b1'] = group1_arch_branch(num_input_channels, num_input_channels, 64)
        module_dict['b2'] = group1_arch_branch(128, num_input_channels, 128)
        module_dict['b3'] = group1_arch_branch(256, num_input_channels, 256)
        module_dict['b4'] = group1_arch_branch(512, num_input_channels, 512)
        module_dict['b5'] = group2_arch_branch(1024, 2048)
        module_dict['b6'] = group3_arch_branch(2048, 1024, 1024)
        module_dict['b7'] = group3_arch_branch(1024, 512, 512)
        module_dict['b8'] = group3_arch_branch(512, 256, 256)
        module_dict['b9'] = group3_arch_branch(256, 128, 128)
        module_dict['fm1'] = group4_arch_branch(128, 64, (2, 2), (2, 2), activation=out_activation)
        module_dict['fm2'] = group4_arch_branch(256, 128, (4, 4), (4, 4), activation=out_activation)
        module_dict['fm3'] = group4_arch_branch(512, 256, (8, 8), (8, 8), activation=out_activation)
        module_dict['aux_out1'] = out_arch_branch(64, num_output_channels=self.__num_output_channels)
        module_dict['aux_out2'] = out_arch_branch(128, num_output_channels=self.__num_output_channels)
        module_dict['aux_out3'] = out_arch_branch(256, num_output_channels=self.__num_output_channels)
        module_dict['out'] = out_arch_branch(64 + 128 + 256, num_output_channels=self.__num_output_channels, activation=out_activation)
        self.layer = nn.ModuleDict(module_dict)

    def forward(self: MicroNet, input_tensor: torch.Tensor) -> list[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Logic for using layers defined in init.

        This method defines how layers are used in forward operation.

        Args:
            input_tensor (torch.Tensor):
                Input images, the tensor is in the shape of NCHW.

        Returns:
            list:
                A list of main and auxiliary outputs. The expected
                format is `[main_output, aux1, aux2, aux3]`.

        """
        b1 = group1_forward_branch(self.layer['b1'], input_tensor, functional.interpolate(input_tensor, size=(128, 128), mode='bicubic'))
        b2 = group1_forward_branch(self.layer['b2'], b1, functional.interpolate(input_tensor, size=(64, 64), mode='bicubic'))
        b3 = group1_forward_branch(self.layer['b3'], b2, functional.interpolate(input_tensor, size=(32, 32), mode='bicubic'))
        b4 = group1_forward_branch(self.layer['b4'], b3, functional.interpolate(input_tensor, size=(16, 16), mode='bicubic'))
        b5 = group2_forward_branch(self.layer['b5'], b4)
        b6 = group3_forward_branch(self.layer['b6'], b5, b4)
        b7 = group3_forward_branch(self.layer['b7'], b6, b3)
        b8 = group3_forward_branch(self.layer['b8'], b7, b2)
        b9 = group3_forward_branch(self.layer['b9'], b8, b1)
        fm1 = group4_forward_branch(self.layer['fm1'], b9)
        fm2 = group4_forward_branch(self.layer['fm2'], b8)
        fm3 = group4_forward_branch(self.layer['fm3'], b7)
        aux1 = self.layer['aux_out1'](fm1)
        aux2 = self.layer['aux_out2'](fm2)
        aux3 = self.layer['aux_out3'](fm3)
        out = torch.cat(tensors=(fm1, fm2, fm3), dim=1)
        out = self.layer['out'](out)
        return [out, aux1, aux2, aux3]

    @staticmethod
    def postproc(image: np.ndarray) -> tuple[np.ndarray, dict]:
        """Post-processing script for MicroNet.

        Args:
            image (ndarray):
                Input image of type numpy array.

        Returns:
            :class:`numpy.ndarray`:
                Pixel-wise nuclear instance segmentation
                prediction.

        """
        pred_bin = np.argmax(image[0], axis=2)
        pred_inst = ndimage.label(pred_bin)[0]
        pred_inst = morphology.remove_small_objects(pred_inst, min_size=50)
        canvas = np.zeros(pred_inst.shape[:2], dtype=np.int32)
        for inst_id in range(1, np.max(pred_inst) + 1):
            inst_map = np.array(pred_inst == inst_id, dtype=np.uint8)
            inst_map = ndimage.binary_fill_holes(inst_map)
            canvas[inst_map > 0] = inst_id
        nuc_inst_info_dict = get_instance_info(canvas)
        return (canvas, nuc_inst_info_dict)

    @staticmethod
    def preproc(image: np.ndarray) -> np.ndarray:
        """Preprocessing function for MicroNet.

        Performs per image standardization.

        Args:
            image (:class:`numpy.ndarray`):
                Input image of type numpy array.

        Returns:
            :class:`numpy.ndarray`:
                Pre-processed numpy array.

        """
        image = np.transpose(image, axes=(2, 0, 1))
        image = image / 255.0
        image = torch.from_numpy(image)
        image_mean = torch.mean(image, dim=(-1, -2, -3))
        stddev = torch.std(image, dim=(-1, -2, -3))
        num_pixels = torch.tensor(torch.numel(image), dtype=torch.float32)
        min_stddev = torch.rsqrt(num_pixels)
        adjusted_stddev = torch.max(stddev, min_stddev)
        image -= image_mean
        image = torch.div(image, adjusted_stddev)
        return np.transpose(image.numpy(), axes=(1, 2, 0))

    @staticmethod
    def infer_batch(model: torch.nn.Module, batch_data: torch.Tensor, *, device: str) -> list[np.ndarray]:
        """Run inference on an input batch.

        This contains logic for forward operation as well as batch I/O
        aggregation.

        Args:
            model (nn.Module):
                PyTorch defined model.
            batch_data (:class:`torch.Tensor`):
                A batch of data generated by
                `torch.utils.data.DataLoader`.
            device (str):
                Transfers model to the specified device. Default is "cpu".

        Returns:
            list(np.ndarray):
                Probability map as a numpy array.

        """
        patch_imgs = batch_data
        patch_imgs_gpu = patch_imgs.to(device).type(torch.float32)
        patch_imgs_gpu = patch_imgs_gpu.permute(0, 3, 1, 2).contiguous()
        model.eval()
        with torch.inference_mode():
            pred, _, _, _ = model(patch_imgs_gpu)
        pred = pred.permute(0, 2, 3, 1).contiguous()
        pred = pred.cpu().numpy()
        return [pred]

def get_bounding_box(mask):
    yy, xx = np.nonzero(mask)
    return np.asarray([xx.min(), yy.min(), xx.max() + 1, yy.max() + 1])

def get_instance_info(pred_inst: np.ndarray, pred_type: np.ndarray=None) -> dict:
    """To collect instance information and store it within a dictionary.

        Args:
            pred_inst (:class:`numpy.ndarray`):
                An image of shape (height, width) which contains the
                probabilities of a pixel being a nuclei.
            pred_type (:class:`numpy.ndarray`):
                An image of shape (height, width, 1) which contains the
                probabilities of a pixel being a certain type of nuclei.

        Returns:
            dict:
                A dictionary containing a mapping of each instance
                within `pred_inst` instance information. It has
                the following form::

                    {
                        0: {  # Instance ID
                            "box": [
                                x_min,
                                y_min,
                                x_max,
                                y_max,
                            ],
                            "centroid": [x, y],
                            "contour": [
                                [x, y],
                                ...
                            ],
                            "type": integer,
                            "prob": float,
                        },
                        ...
                    }

                where the instance ID is an integer corresponding to the
                instance at the same pixel value within `pred_inst`.

        """
    inst_id_list = np.unique(pred_inst)[1:]
    inst_info_dict = {}
    for inst_id in inst_id_list:
        inst_map = pred_inst == inst_id
        inst_box = get_bounding_box(inst_map)
        inst_box_tl = inst_box[:2]
        inst_map = inst_map[inst_box[1]:inst_box[3], inst_box[0]:inst_box[2]]
        inst_map = inst_map.astype(np.uint8)
        inst_moment = cv2.moments(inst_map)
        inst_contour = cv2.findContours(inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        inst_contour = inst_contour[0][0].astype(np.int32)
        inst_contour = np.squeeze(inst_contour)
        if inst_contour.shape[0] < 3:
            continue
        if len(inst_contour.shape) != 2:
            continue
        inst_centroid = [inst_moment['m10'] / inst_moment['m00'], inst_moment['m01'] / inst_moment['m00']]
        inst_centroid = np.array(inst_centroid)
        inst_contour += inst_box_tl[None]
        inst_centroid += inst_box_tl
        inst_info_dict[inst_id] = {'box': inst_box, 'centroid': inst_centroid, 'contour': inst_contour, 'prob': None, 'type': None}
    if pred_type is not None:
        for inst_id in list(inst_info_dict.keys()):
            c_min, r_min, c_max, r_max = inst_info_dict[inst_id]['box']
            inst_map_crop = pred_inst[r_min:r_max, c_min:c_max]
            inst_type_crop = pred_type[r_min:r_max, c_min:c_max]
            inst_map_crop = inst_map_crop == inst_id
            inst_type = inst_type_crop[inst_map_crop]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0 and len(type_list) > 1:
                inst_type = type_list[1][0]
            type_dict = {v[0]: v[1] for v in type_list}
            type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1e-06)
            inst_info_dict[inst_id]['type'] = int(inst_type)
            inst_info_dict[inst_id]['prob'] = float(type_prob)
    return inst_info_dict
