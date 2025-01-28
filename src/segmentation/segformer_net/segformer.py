import torch as th
import numpy as np

from torch.nn import (
    Upsample,
    Linear,
    Softmax,
    Flatten,
    Parameter,
    Sequential,
    ModuleList,
    Module,
    LayerNorm,
    ModuleDict,
    BatchNorm2d,
    Softmax2d,
    Sigmoid,
    Tanh
)



class ResLayer(Module):

    def __init__(self, in_features: int, out_features: int) -> None:
        
        super().__init__()
        self._projection = Linear(in_features=in_features, out_features=out_features)
    
    def __call__(self, inputs: list[th.Tensor]) -> th.Tensor:
        return th.add(self._projection(inputs[0]), inputs[1])
    
class MulHeadAttention(Module):

    def __init__(
        self,
        in_features: int,
        out_features: int
    ) -> None:
        
        super().__init__()
        self.d = th.tensor(out_features)
        self._projections = ModuleDict({
            "q": Linear(in_features=in_features, out_features=out_features),
            "k": Linear(in_features=in_features, out_features=out_features),
            "v": Linear(in_features=in_features, out_features=out_features)
        })
        self._act = Softmax(dim=1)
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        return th.mm(
            self._act(th.mm(self._projections["q"](inputs), self._projections["k"](inputs).T)),
            self._projections["v"](inputs)
        )

class TransformerEncoder(Module):

    def __init__(
        self,
        img_size: int,
        att_features: int,
        hiden_features: int,
        out_features: int
    ) -> None:

        super().__init__()
        self._flt = Flatten()
        self._att = MulHeadAttention(in_features=(img_size ** 2) * 3, out_features=att_features)
        self._res0 = ResLayer(in_features=(img_size ** 2) * 3, out_features=att_features)

        self._linear = Sequential(
            Linear(in_features=att_features, out_features=hiden_features),
            Linear(in_features=hiden_features, out_features=hiden_features),
            Linear(in_features=hiden_features, out_features=out_features)
        )
        self._res1 = ResLayer(in_features=att_features, out_features=out_features)
        
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        flt = self._flt(inputs)
        att = self._att(flt)
        return self._linear(att)
    

class MaskTransformer(Module):

    def __init__(
        self,
        in_features: int,
        cls_n: int,
    ) -> None:

        super().__init__()
        self._weights = Parameter(th.normal(0.0, 1.0, (in_features, cls_n)))
        self._att = Softmax(dim=1)
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        return th.mm(inputs, self._weights)

    
class UpSampleNet(Module):

    def __init__(
        self,
        cls_n: int,
        patch_s: int,
        img_size: int
    ) -> None:
        
        super().__init__()
        self.ln = int((img_size / patch_s) / 2) - 1
        self.cls = cls_n
        self.patch_s = patch_s

        self._projection = Linear(in_features=cls_n, out_features=cls_n * (patch_s ** 2))
        self._net = ModuleList([
            Sequential(
                Upsample(scale_factor=2),
                BatchNorm2d(num_features=cls_n),
                Tanh()
            )
        for _ in range(self.ln)])
        self._att = Softmax(dim=1)  

    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        projection = self._projection(inputs)
        projection = projection.view((projection.size()[0], self.cls, self.patch_s, self.patch_s))
        x = projection
        for layer in self._net:
            x = layer(x)
        
        return self._att(x)

    
class Segformer(Module):

    def __init__(
        self,
        img_size: int,
        patch_s: int,
        att_features: int,
        hiden_features: int,
        tr_out_features: int,
        cls_n: int
    ) -> None:

        super().__init__()
        self._transformer = TransformerEncoder(
            img_size=img_size,
            att_features=att_features,
            hiden_features=hiden_features,
            out_features=tr_out_features
        )
        self._mask_transformer = MaskTransformer(
            in_features=tr_out_features,
            cls_n=cls_n
        )
        self._up_sample = UpSampleNet(
            cls_n=cls_n,
            patch_s=patch_s,
            img_size=img_size
        )
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        transformer = self._transformer(inputs)
        mask_tf = self._mask_transformer(transformer)
        return self._up_sample(mask_tf)


