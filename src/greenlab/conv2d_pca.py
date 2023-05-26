import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t
from typing import Union
import cv2
import math
from .extract_patches import Patchify


class Conv2d_PCA(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "reflect",
        device=None,
        dtype=None,
        mode="pca",
        max_patches_per_image=100,
        name="",
        verbose=True,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size  # type: ignore
        self.stride = stride  # type: ignore
        self.padding = padding  # type: ignore
        self.dilation = dilation  # type: ignore
        self.groups = groups
        self.padding_mode = padding_mode
        self.mode = mode
        self.max_patches_per_image = max_patches_per_image
        self.name = name if len(name) > 0 else mode
        self.verbose = verbose

        assert mode in ["pca", "saab"], "mode must be either pca or saab"

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            print("[{}] ".format(self.name), "Training")
            print("[{}] ".format(self.name), "Input Shape:", input.shape)
            if self.mode == "pca":
                self._pca_init(input)
            elif self.mode == "saab":
                self._saab_init(input)
        else:
            if self.verbose:
                print("[{}] ".format(self.name), "Inference")
                print("[{}] ".format(self.name), "Input Shape:", input.shape)
        output = self._conv_forward(input, self.weight, self.bias)
        if self.verbose:
            print("[{}] ".format(self.name), "Output Shape:", output.shape)
        return output

    def inverse(self, output: Tensor) -> Tensor:
        # import matplotlib.pyplot as plt
        # assert self.kernel_size[0] % 2 == 1, 'kernel_size[0] must be odd'
        # assert self.kernel_size[1] % 2 == 1, 'kernel_size[1] must be odd'
        # assert self.padding[0] == self.kernel_size[0]//2, 'padding must be equal to kernel_size[0]//2'
        # assert self.padding[1] == self.kernel_size[1]//2, 'padding must be equal to kernel_size[1]//2'
        n, c, h, w = output.shape
        # print(output.shape)
        output = output.permute(0, 2, 3, 1).reshape(-1, c)
        output = torch.matmul(output, self.weight.reshape(self.out_channels, -1)[:c, :])
        output = output.reshape(
            n, h, w, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        )
        output = output.permute(0, 3, 1, 2, 4, 5)
        input = torch.zeros(
            n,
            self.in_channels,
            (h - 1) * self.stride[0] + self.kernel_size[0],
            (w - 1) * self.stride[1] + self.kernel_size[1],
        )
        count = torch.zeros(
            n,
            self.in_channels,
            (h - 1) * self.stride[0] + self.kernel_size[0],
            (w - 1) * self.stride[1] + self.kernel_size[1],
        )
        # print(input.shape, output.shape)
        for i in range(self.kernel_size[0]):
            for j in range(self.kernel_size[1]):
                input[
                    :,
                    :,
                    i : i + h * self.stride[0] : self.stride[0],
                    j : j + w * self.stride[1] : self.stride[1],
                ] += output[:, :, :, :, i, j]
                count[
                    :,
                    :,
                    i : i + h * self.stride[0] : self.stride[0],
                    j : j + w * self.stride[1] : self.stride[1],
                ] += 1
        input = input / count
        # print(input.dtype, count.dtype, output.dtype)
        # plt.figure(figsize=(5,5))
        # plt.imshow(count[0,0,:,:].detach().numpy())
        # plt.colorbar()
        # plt.show()
        return input[
            :,
            :,
            self.padding[0] : self.padding[0] + h * self.stride[0],  # type: ignore
            self.padding[1] : self.padding[1] + w * self.stride[1],  # type: ignore
        ]

    def get_samples(
        self, input: Tensor, labels: Tensor, interpolation=cv2.INTER_LANCZOS4
    ):
        print(input.shape, labels.shape)
        assert (
            input.shape[0] == labels.shape[0]
        ), "input and labels must have the same batch size"
        assert len(input.shape) == 4, "input must be a 4D tensor"
        assert len(labels.shape) == 4, "labels must be a 4D tensor"
        output = self._conv_forward(input, self.weight, self.bias)
        output_resized = F.interpolate(
            output, size=labels.shape[2:], mode="bilinear", align_corners=True
        )
        X = output_resized.permute(0, 2, 3, 1)
        X = X.reshape(-1, self.out_channels)
        # labels = Resize(size=output.shape[2:], interpolation=interpolation)(labels)
        y = labels.view(-1)
        return output, X.detach().numpy(), y.detach().numpy()

    def _pca_init(self, input: Tensor) -> None:
        print("[{}] ".format(self.name), "Starting PCA Transform")
        assert self.weight.data.shape == (
            self.out_channels,
            self.in_channels,
            self.kernel_size[0],
            self.kernel_size[1],
        )
        # get patches
        patches = Patchify(
            max_patches_per_image=self.max_patches_per_image,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            padding_mode=self.padding_mode,
            pad=False,
        ).extact(input)
        print("[{}] ".format(self.name), "Patches Shape:", patches.shape)
        # run PCA
        _, S, V = torch.pca_lowrank(patches, q=self.out_channels, center=True, niter=2)
        self.explained_variance_ratio_ = S**2 / torch.sum(S**2)

        # set weight
        w = V.T.reshape(
            self.out_channels,
            self.in_channels,
            self.kernel_size[0],
            self.kernel_size[1],
        )
        with torch.no_grad():
            self.weight.copy_(w)

        self.training = False
        print("[{}] ".format(self.name), "PCA Transform Complete")

    def _saab_init(self, input: Tensor) -> None:
        print("[{}] ".format(self.name), "Starting Saab Transform")
        assert self.weight.data.shape == (
            self.out_channels,
            self.in_channels,
            self.kernel_size[0],
            self.kernel_size[1],
        )
        # get patches
        patches = Patchify(
            max_patches_per_image=self.max_patches_per_image,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            padding_mode=self.padding_mode,
            pad=False,
        ).extact(input)
        patches -= torch.mean(patches, dim=0, keepdim=True)
        total_variance = torch.var(patches, dim=0).sum()
        dc_variance = torch.var(
            torch.mean(patches, dim=1) * math.sqrt(patches.shape[1])
        ).unsqueeze(0)
        patches -= torch.mean(patches, dim=1, keepdim=True)

        print("[{}] ".format(self.name), "Patches Shape:", patches.shape)
        # run PCA
        U, S, V = torch.pca_lowrank(
            patches, q=self.out_channels - 1, center=True, niter=2
        )
        dc_kernel = 1 / math.sqrt(patches.shape[1]) * torch.ones((patches.shape[1], 1))
        V = torch.cat([dc_kernel, V], dim=1)
        V = self._orthogonal(V)
        m = U.shape[0]
        # calculate total explained variance from patches
        self.explained_variance_ratio_ = torch.cat([dc_variance, S**2 / (m - 1)])
        self.explained_variance_ratio_ /= total_variance
        # set weight
        w = V.T.reshape(
            self.out_channels,
            self.in_channels,
            self.kernel_size[0],
            self.kernel_size[1],
        )

        with torch.no_grad():
            self.weight.copy_(w)

        self.training = False
        print("[{}] ".format(self.name), "Saab Transform Complete")

    def _orthogonal(self, V):
        return torch.linalg.qr(V).Q

    def print_total_explained_variance_ratio(self):
        print(self.explained_variance_ratio_.sum())
