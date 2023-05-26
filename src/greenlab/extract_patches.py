import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp


def random_sample(X, k):
    return X[torch.randperm(X.shape[0])[:k]]


class Patchify:
    def __init__(
        self,
        max_patches_per_image=100,
        batch_size=1,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        padding_mode="reflect",
        pad=False,
        n_jobs=-1,
    ):
        assert isinstance(kernel_size, tuple), "kernel_size must be tuple"
        assert isinstance(stride, tuple), "stride must be tuple"
        assert isinstance(padding, tuple), "padding must be tuple"
        assert isinstance(dilation, tuple), "dilation must be tuple"
        self.max_patches_per_image = max_patches_per_image
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.padding_mode = padding_mode
        self.pad = pad
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        if self.n_jobs == -1:
            self.n_jobs = mp.cpu_count()
        self.unfold = nn.Unfold(
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=0,
            stride=self.stride,
        )

    def _extact(self, input):
        if self.pad:
            input = F.pad(
                input,
                (self.padding[1], self.padding[1], self.padding[0], self.padding[0]),
                self.padding_mode,
            )
        output = self.unfold(input)
        output = output.permute(0, 2, 1)
        output = output.reshape(
            -1, input.shape[1] * self.kernel_size[0] * self.kernel_size[1]
        )
        output = random_sample(output, k=self.max_patches_per_image * self.batch_size)
        return output

    def extact(self, input):
        assert len(input.shape) == 4, "input must be 4D tensor"
        # batchify
        input = input.view(
            -1, self.batch_size, input.shape[1], input.shape[2], input.shape[3]
        ).detach()
        with mp.get_context("spawn").Pool(self.n_jobs) as pool:
            patches = pool.map(self._extact, input)
        patches = torch.cat(patches, dim=0)
        return patches


if __name__ == "__main__":
    patchify = Patchify(kernel_size=(3, 3))
    images = torch.randn(100, 64, 104, 104)
    patches = patchify.extact(images)
    print(patches.shape)
    patches -= torch.mean(patches, dim=0, keepdim=True)
    patches -= torch.mean(patches, dim=1, keepdim=True)
    print(torch.mean(patches, dim=0, keepdim=True).sum())
    print(torch.mean(patches, dim=1, keepdim=True).sum())
