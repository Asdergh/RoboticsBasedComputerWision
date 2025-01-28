import torch as th
from torch.nn import (
    Conv2d,
    Parameter,
    Upsample,
    MaxPool2d,
    Sigmoid,
    MSELoss,
    CrossEntropyLoss,
    Softmax,
    Module,
    Sequential,
    BatchNorm2d,
    Dropout,
    functional,
    Tanh
)
from torch.utils.data import DataLoader




        
class ConvSS(Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        sampling: str = "down"
    ) -> None:
        
        super().__init__()
        sampling = {
            "down": MaxPool2d(
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=1
            ),
            "up": Upsample(scale_factor=2)
        }[sampling]

        self._net = Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ),
            BatchNorm2d(num_features=out_channels),
            sampling,
            Tanh()
        )
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        return self._net(inputs)


class FCNet(Module):

    def __init__(
        self, 
        in_channels: int, 
        h_channels: int,
        classes_n: int
    ) -> None:

        super().__init__()
        self.conv0_down = ConvSS(in_channels=in_channels, out_channels=128)
        self.conv1_down = ConvSS(in_channels=128, out_channels=64)
        self.conv2_down = ConvSS(in_channels=64, out_channels=32)
        self.conv3_down = ConvSS(in_channels=32, out_channels=h_channels)

        self.conv0_up = ConvSS(in_channels=h_channels, out_channels=32, sampling="up")
        self.conv1_up = ConvSS(in_channels=32 * 2 , out_channels=64, sampling="up")
        self.conv2_up = ConvSS(in_channels=64 * 2, out_channels=128, sampling="up")
        self.conv3_up = ConvSS(in_channels=128 * 2, out_channels=classes_n, sampling="up")

        self.out = Softmax(dim=1)
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        d_conv0 = self.conv0_down(inputs)
        d_conv1 = self.conv1_down(d_conv0)
        d_conv2 = self.conv2_down(d_conv1)
        d_conv3 = self.conv3_down(d_conv2)

        u_conv0 = self.conv0_up(d_conv3)
        u_conv1 = self.conv1_up(th.cat([u_conv0, d_conv2], dim=1))
        u_conv2 = self.conv2_up(th.cat([u_conv1, d_conv1], dim=1))
        u_conv3 = self.conv3_up(th.cat([u_conv2, d_conv0], dim=1))

        return self.out(u_conv3)



if __name__ == "__main__":

    input = th.normal(0.12, 1.12, (10, 3, 128, 128))
    net = FCNet(in_channels=3, h_channels=32, classes_n=30)
    print(net(input).size())
    
                
