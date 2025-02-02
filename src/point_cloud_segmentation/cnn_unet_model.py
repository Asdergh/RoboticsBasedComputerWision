import torch as th
from torch.nn import (
    Conv2d,
    ConvTranspose2d,
    BatchNorm2d,
    Tanh,
    ReLU,
    Dropout,
    Sigmoid,
    Softmax,
    Module,
    Sequential,
    ModuleList,
    Upsample,
    Flatten
)

__activations__ = {
    "tanh": Tanh,
    "relu": ReLU,
    "sigmoid": Sigmoid
}


class ConvDownsample(Module):


    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "relu"
    ) -> None:

        super().__init__()
        self._net_ = Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2,
                padding=0,
                kernel_size=(3, 3)
            ),
            BatchNorm2d(num_features=out_channels),
            __activations__[activation]()
        )
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        return self._net_(inputs)
    
class ConvUpsample(Module):


    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "relu"
    ) -> None:

        super().__init__()
        self._net_ = Sequential(
            ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2,
                padding=0,
                kernel_size=(3, 3)
            ),
            BatchNorm2d(num_features=out_channels),
            __activations__[activation]()
        )
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        return self._net_(inputs)
    
class UnetSegmenter(Module):

    def __init__(
        self,
        in_channels: int,
        hiden_channels: int,
        out_channels: int,
    ) -> None:
        
        super().__init__()
        self._flatten_ = Flatten(start_dim=2, end_dim=3)
        self._flatten_.requires_grad = False

        self._down_ = ModuleList([
            ConvDownsample(in_channels=in_channels, out_channels=hiden_channels),
            ConvDownsample(in_channels=hiden_channels, out_channels=hiden_channels),
            ConvDownsample(in_channels=hiden_channels, out_channels=hiden_channels),
        ])

        self._up_ = ModuleList([
            ConvUpsample(in_channels=hiden_channels, out_channels=hiden_channels),
            ConvUpsample(in_channels=hiden_channels, out_channels=hiden_channels),
            ConvUpsample(in_channels=hiden_channels * 2, out_channels=out_channels)
        ])

        self._out_ = Sequential(
            Conv2d(
                in_channels=out_channels + in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0
            ),
            BatchNorm2d(num_features=out_channels),
            __activations__["tanh"]()
        )
        
        
    
    def predict(self, inputs: th.Tensor) -> tuple[th.Tensor]:
        out = self._flatten_(self(inputs)).permute((0, 2, 1))
        return (
            out[:, :, :3],
            out[:, :, 3:]
        )
        
    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        x = inputs
        down_buffer_ = [x]
        for down in self._down_:
            x = down(x)
            down_buffer_.append(x)
        
        down_buffer_ = down_buffer_[::-1][1:]
        for (idx, up) in enumerate(self._up_):
            x = up(x)
            if idx != 0:
                x = th.cat([x, down_buffer_[idx]], dim=1)

        return self._out_(x)




