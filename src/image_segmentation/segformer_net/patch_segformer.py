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
    Sigmoid
)



class ImagePatchSeparation(Module):

    def __init__(
        self,
        patch_s: int, 
        img_size: int,
        
    ) -> None:

        super().__init__()
        self.patch_s = patch_s
        self.n = int((img_size ** 2) / (self.patch_s ** 2))
        self.n_d = int(th.sqrt(th.tensor(self.n)).item())


    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        out = th.zeros((
            inputs.size()[0],
            self.n,
            inputs.size()[1],
            self.patch_s,
            self.patch_s
        ))
        k = 0
        for i in range(self.n_d):
            for j in range(self.n_d):
                out[:, k, :, :, :] = inputs[
                    :, :, 
                    i * self.patch_s: (i + 1) * self.patch_s, 
                    j * self.patch_s: (j + 1) * self.patch_s
                ]
                k += 1

        return out

class PatchMulHeadAttention(Module):

    def __init__(
        self,
        patch_n: int,
        in_features: int,
        out_features: int
    ) -> None:
        
        super().__init__()
        self.d = th.tensor(out_features)
        self._projections = ModuleList([
            ModuleDict({
                "q": Linear(in_features=in_features, out_features=out_features),
                "k": Linear(in_features=in_features, out_features=out_features),
                "v": Linear(in_features=in_features, out_features=out_features)
            })
            for _ in range(patch_n)
        ])
        self._act = Softmax(dim=1)
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        return th.cat([th.unsqueeze(
                th.mm(self._act(th.mm(att_m["q"](inputs[:, j, :]), att_m["k"](inputs[:, j, :]).T) 
                / th.sqrt(self.d)), att_m["v"](inputs[:, j, :])), dim=1)
                for j, att_m in enumerate(self._projections)
        ], dim=1)

    
class TransformerEncoder(Module):

    def __init__(
        self,
        img_size: int,
        patch_s: int,
        att_features: int,
        hiden_features: int,
        out_features: int
    ) -> None:
        
        super().__init__()
        self._img_sep = ImagePatchSeparation(img_size=img_size, patch_s=patch_s)
        self.p1 = Linear(in_features=(patch_s ** 2) * 3, out_features=att_features)
        self.p2 = Linear(in_features=att_features, out_features=out_features)
        self._att = Sequential(
            PatchMulHeadAttention(patch_n=self._img_sep.n, in_features=(patch_s ** 2) * 3, out_features=att_features),
            LayerNorm(normalized_shape=att_features)
        )

        self._linear_projections = ModuleList(
            [Linear(in_features=att_features, out_features=hiden_features), ] + 
            [Linear(in_features=hiden_features, out_features=hiden_features) for _ in range(3)] + 
            [Linear(in_features=hiden_features, out_features=out_features)]
        )
        self._act = Softmax(dim=2)

    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        img_sep = self._img_sep(inputs)
        flatten = Flatten(start_dim=2)(img_sep)
        att = th.add(self._att(flatten), self.p1(flatten))
        x = att
        for layer in self._linear_projections:
            x = layer(x)

        projection = th.add(x, self.p2(att))
        
        return self._act(projection)


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

        return th.cat([th.unsqueeze(
            th.mm(inputs[j, :, :], self._weights).T, dim=0)
            for j in range(inputs.size()[0])
        ], dim=0)

class UpSampleNet(Module):

    def __init__(
        self,
        cls_n: int,
        patch_s: int,
        img_size: int
    ) -> None:
        
        super().__init__()
        ln = int((img_size / patch_s) / 2) - 1
        n = int((img_size ** 2) / (patch_s ** 2))
        self.cls = cls_n
        self.patch_s = patch_s

        self._projection = Linear(in_features=n, out_features=patch_s ** 2)
        self._net = ModuleList([
            Sequential(
                Upsample(scale_factor=2),
                BatchNorm2d(num_features=cls_n),
                Sigmoid()
            )
        for _ in range(ln)])
        self._att = Softmax(dim=2)

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
        att_out_features: int,
        hiden_features: int,
        tr_out_features: int,
        cls_n: int,
        patch_s: int = 16

    ) -> None:
        
        super().__init__()
        self._transformer_encoder = TransformerEncoder(
            img_size=img_size,
            patch_s=patch_s,
            att_features=att_out_features,
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
    
    def total_params_n(self) -> str:
        
        total_n = 0
        for params in list(self.parameters()):
            total_n += th.prod(th.Tensor(list(params.size())))
        
        return f" total number of params: [{int(total_n)}]"
            
    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        
        tr_encodings = self._transformer_encoder(inputs)
        mask_weights = self._mask_transformer(tr_encodings)
        return self._up_sample(mask_weights)



        
        