# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from mmpretrain.models import HiViT, VisionTransformer, ConvNeXt
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from ..utils import build_2d_sincos_position_embedding
from .base import BaseSelfSupervisor


@MODELS.register_module()
class MAEViTself(VisionTransformer):
    """Vision Transformer for MAE pre-training.

    A PyTorch implement of: `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.
    This module implements the patch masking in MAE and initialize the
    position embedding with sine-cosine position embedding.

    Args:
        arch (str | dict): Vision Transformer architecture
            Default: 'b'
        img_size (int | tuple): Input image size
        patch_size (int | tuple): The patch size
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        out_type (str): The type of output features. Please choose from

            - ``"cls_token"``: The class token tensor with shape (B, C).
            - ``"featmap"``: The feature map tensor from the patch tokens
              with shape (B, C, H, W).
            - ``"avg_featmap"``: The global averaged feature map tensor
              with shape (B, C).
            - ``"raw"``: The raw feature tensor includes patch tokens and
              class tokens with shape (B, L, C).

            It only works without input mask. Defaults to ``"avg_featmap"``.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        mask_ratio (bool): The ratio of total number of patches to be masked.
            Defaults to 0.75.
        init_cfg (Union[List[dict], dict], optional): Initialization config
            dict. Defaults to None.
    """

    def __init__(self,
                 arch: Union[str, dict] = 'b',
                 img_size: int = 224,
                 patch_size: int = 16,
                 out_indices: Union[Sequence, int] = -1,
                 drop_rate: float = 0,
                 drop_path_rate: float = 0,
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 final_norm: bool = True,
                 out_type: str = 'raw',
                 interpolate_mode: str = 'bicubic',
                 patch_cfg: dict = dict(),
                 layer_cfgs: dict = dict(),
                 mask_ratio: float = 0.75,
                 init_cfg: Optional[Union[List[dict], dict]] = None,
                 mask_type:str = 'random',) -> None:
        super().__init__(
            arch=arch,
            img_size=img_size,
            patch_size=patch_size,
            out_indices=out_indices,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg,
            final_norm=final_norm,
            out_type=out_type,
            with_cls_token=True,
            interpolate_mode=interpolate_mode,
            patch_cfg=patch_cfg,
            layer_cfgs=layer_cfgs,
            init_cfg=init_cfg)

        # position embedding is not learnable during pretraining
        self.iter_ = 1
        self.mask_type = mask_type
        self.pos_embed.requires_grad = False
        self.mask_ratio = mask_ratio
        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]

    def init_weights(self) -> None:
        """Initialize position embedding, patch embedding and cls token."""
        super().init_weights()
        pos_embed = build_2d_sincos_position_embedding(
            int(self.num_patches**.5),
            self.pos_embed.shape[-1],
            cls_token=True)
        self.pos_embed.data.copy_(pos_embed.float())

        w = self.patch_embed.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)


    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.75
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate the random mask for MAE Pre-training.

        Args:
            x (torch.Tensor): Image with data augmentation applied, which is
                of shape B x L x C.
            mask_ratio (float): The mask ratio of total patches.
                Defaults to 0.75.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: masked image, mask
            and the ids to restore original image.

            - ``x_masked`` (torch.Tensor): masked image.
            - ``mask`` (torch.Tensor): mask used to mask image.
            - ``ids_restore`` (torch.Tensor): ids to restore original image.
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def block_masking(
        self,
        x: torch.Tensor,
        out_size: Tuple[torch.Tensor, torch.Tensor],
        mask_ratio: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate the block mask for MAE pre-training.

        Args:
            x (torch.Tensor): Image with data augmentation applied, which is
                of shape B x L x C.
            mask_ratio (float): The mask ratio of total blocks.
                Defaults to 0.75.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: masked image, mask
            and the ids to restore original image.

            - ``x_masked`` (torch.Tensor): masked image.
            - ``mask`` (torch.Tensor): mask used to mask image.
            - ``ids_restore`` (torch.Tensor): ids to restore original image.
        """
        mask_ratio = float(0.5**0.5)
        N, L, D = x.shape  # batch, length, dim
        out_x, out_y = out_size
        # Calculate the block size dynamically
        block_size_x = int(out_x * mask_ratio)
        block_size_y = int(out_y * mask_ratio)

        start_x = torch.randint(0, out_x-block_size_x-1, (N,))
        start_y = torch.randint(0, out_x-block_size_y-1, (N,))


        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.zeros([N, out_x, out_y], device=x.device)
        for i in range(N):
            mask[i,start_x[i]:(start_x[i]+block_size_x) ,start_y[i]:(start_y[i]+block_size_y)] =1
        mask = mask.reshape(N, L)

        # unshuffle to get the binary mask
        # Calculate ids_keep
        ids_keep = torch.nonzero(mask==0, as_tuple=True)[1].view(N, -1)

        # Calculate ids_restore
        ids_restore = torch.arange(0, L, device=x.device).unsqueeze(0).repeat(N, 1)

        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        return x_masked, mask, ids_restore

    def grid_masking(
        self,
        x: torch.Tensor,
        out_size: Tuple[torch.Tensor, torch.Tensor],
        mask_ratio: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate the block mask for MAE pre-training.

        Args:
            x (torch.Tensor): Image with data augmentation applied, which is
                of shape B x L x C.
            mask_ratio (float): The mask ratio of total blocks.
                Defaults to 0.75.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: masked image, mask
            and the ids to restore original image.

            - ``x_masked`` (torch.Tensor): masked image.
            - ``mask`` (torch.Tensor): mask used to mask image.
            - ``ids_restore`` (torch.Tensor): ids to restore original image.
        """
        # mask_ratio = float(0.5**0.5)
        N, L, D = x.shape  # batch, length, dim
        out_x, out_y = out_size
        # Calculate the block size dynamically
        block_size_x = 2
        block_size_y = 2

        # Calculate the number of blocks in each dimension
        num_blocks_x = out_x // block_size_x
        num_blocks_y = out_y // block_size_y

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.zeros([N, out_x, out_y], device=x.device)
        
        for i in range(N):
            flag=0
            for j in range(num_blocks_x):
                for k in range(num_blocks_y):
                    if flag == 1:  # gaply decide to mask or not
                        mask[i, j * block_size_x : (j + 1) * block_size_x, k * block_size_y : (k + 1) * block_size_y] = 1
                        flag = 0
                    else:
                        flag =1
        
        mask = mask.reshape(N, L)

        # unshuffle to get the binary mask
        # Calculate ids_keep
        ids_keep = torch.nonzero(mask==0, as_tuple=True)[1].view(N, -1)

        # Calculate ids_restore
        ids_restore = torch.arange(0, L, device=x.device).unsqueeze(0).repeat(N, 1)

        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        return x_masked, mask, ids_restore


    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[bool] = True,
        
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate features for masked images.

        The function supports two kind of forward behaviors. If the ``mask`` is
        ``True``, the function will generate mask to masking some patches
        randomly and get the hidden features for visible patches, which means
        the function will be executed as masked imagemodeling pre-training;
        if the ``mask`` is ``None`` or ``False``, the forward function will
        call ``super().forward()``, which extract features from images without
        mask.


        Args:
            x (torch.Tensor): Input images, which is of shape B x C x H x W.
            mask (bool, optional): To indicate whether the forward function
                generating ``mask`` or not.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Hidden features,
            mask and the ids to restore original image.

            - ``x`` (torch.Tensor): hidden features, which is of shape
              B x (L * mask_ratio) x C.
            - ``mask`` (torch.Tensor): mask used to mask image.
            - ``ids_restore`` (torch.Tensor): ids to restore original image.
        """
        if mask is None or False:
            return super().forward(x)

        else:
            B = x.shape[0]
            x, outsize = self.patch_embed(x)
            # add pos embed w/o cls token
            x = x + self.pos_embed[:, 1:, :]

            # masking: length -> length * mask_ratio
            if self.mask_type == 'random':
                x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
            elif self.mask_type =='block':
                x, mask, ids_restore = self.block_masking(x, outsize, self.mask_ratio)
            elif self.mask_type =='grid':
                x, mask, ids_restore = self.grid_masking(x, outsize, self.mask_ratio)
            elif self.mask_type == 'mix':
                if self.iter_ == 1:
                    x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
                    self.iter_ = 0
                else:
                    x, mask, ids_restore = self.block_masking(x, outsize, self.mask_ratio)
                    self.iter_ = 1
            # append cls token
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

            for _, layer in enumerate(self.layers):
                x = layer(x)
            # Use final norm
            x = self.norm1(x)

            return (x, mask, ids_restore)





@MODELS.register_module()
class MAEself(BaseSelfSupervisor):
    """MAE.

    Implementation of `Masked Autoencoders Are Scalable Vision Learners
    <https://arxiv.org/abs/2111.06377>`_.
    """
    def visualize(self, inputs):
        latent, mask, ids_restore = self.backbone(inputs)# , mask_type='grid'
        pred = self.neck(latent, ids_restore)
        return pred, mask


    def extract_feat(self, inputs: torch.Tensor):
        return self.backbone(inputs, mask=None)

    def loss(self, inputs: torch.Tensor, data_samples: List[DataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (torch.Tensor): The input images.
            data_samples (List[DataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        # ids_restore: the same as that in original repo, which is used
        # to recover the original order of tokens in decoder.
        latent, mask, ids_restore = self.backbone(inputs)
        pred = self.neck(latent, ids_restore)
        loss = self.head.loss(pred, inputs, mask)
        losses = dict(loss=loss)
        return losses



@MODELS.register_module()
class MAEHiViTself(HiViT):
    """HiViT for MAE pre-training.

    A PyTorch implement of: `HiViT: A Simple and More Efficient Design
    of Hierarchical Vision Transformer <https://arxiv.org/abs/2205.14949>`_.
    This module implements the patch masking in MAE and initialize the
    position embedding with sine-cosine position embedding.

    Args:
        arch (str | dict): Vision Transformer architecture
            Default: 'b'
        img_size (int | tuple): Input image size
        patch_size (int | tuple): The patch size
            Defaults to 4, to downsample 4x at the first stage
        inner_patches (int): The inner patches within a token
            Defaults to 4
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        ape (bool): the absolute position embedding
        rpe (bool): the relative position embedding
            Defaults to False
        layer_scale_init_value (float): the layer scale init value
        mask_ratio (bool): The ratio of total number of patches to be masked.
            Defaults to 0.75.
        init_cfg (Union[List[dict], dict], optional): Initialization config
            dict. Defaults to None.
    """

    def __init__(self,
                 arch: Union[str, dict] = 'b',
                 img_size: int = 224,
                 patch_size: int = 16,
                 inner_patches: int = 4,
                 out_indices: Union[list, int] = [23],
                 drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 ape: bool = True,
                 rpe: bool = False,
                 layer_scale_init_value: float = 0.0,
                 mask_ratio: float = 0.75,
                 init_cfg: Optional[Union[List[dict], dict]] = None,
                 mask_type:str = 'random',) -> None:
        super().__init__(
            arch=arch,
            img_size=img_size,
            patch_size=patch_size,
            inner_patches=inner_patches,
            out_indices=out_indices,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg,
            ape=ape,
            rpe=rpe,
            layer_scale_init_value=layer_scale_init_value,
            init_cfg=init_cfg)
        
        self.mask_type = mask_type
        self.pos_embed.requires_grad = False
        self.mask_ratio = mask_ratio
        self.num_patches = self.patch_embed.num_patches

    def init_weights(self) -> None:
        """Initialize position embedding, patch embedding."""
        super().apply(self._init_weights)
        pos_embed = build_2d_sincos_position_embedding(
            int(self.num_patches**.5),
            self.pos_embed.shape[-1],
            cls_token=False)
        self.pos_embed.data.copy_(pos_embed.float())

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def masking_id(
            self, batch_size,
            mask_ratio) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate the mask for MAE Pre-training.

        Args:
            batch_size: The batch size of input data
            mask_ratio: The mask ratio of total patches.
                Defaults to 0.75.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: the ids
            for the tokens retained, the ids to restore original image,
            and the mask
        """
        N, L = batch_size, self.pos_embed.size(1)
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(
            N, L, device=self.pos_embed.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=self.pos_embed.device)
        mask[:, :ids_keep.size(1)] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, ids_restore, mask

    def block_masking(
        self,
        x: torch.Tensor,
        out_size: Tuple[torch.Tensor, torch.Tensor],
        mask_ratio: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate the block mask for MAE pre-training.

        Args:
            x (torch.Tensor): Image with data augmentation applied, which is
                of shape B x L x C.
            mask_ratio (float): The mask ratio of total blocks.
                Defaults to 0.75.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: masked image, mask
            and the ids to restore original image.

            - ``x_masked`` (torch.Tensor): masked image.
            - ``mask`` (torch.Tensor): mask used to mask image.
            - ``ids_restore`` (torch.Tensor): ids to restore original image.
        """
        mask_ratio = float(0.5**0.5)
        N, L, D = x.shape  # batch, length, dim
        out_x, out_y = out_size
        # Calculate the block size dynamically
        block_size_x = int(out_x * mask_ratio)
        block_size_y = int(out_y * mask_ratio)

        start_x = torch.randint(0, out_x-block_size_x-1, (N,))
        start_y = torch.randint(0, out_x-block_size_y-1, (N,))


        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.zeros([N, out_x, out_y], device=x.device)
        for i in range(N):
            mask[i,start_x[i]:(start_x[i]+block_size_x) ,start_y[i]:(start_y[i]+block_size_y)] =1
        mask = mask.reshape(N, L)

        # unshuffle to get the binary mask
        # Calculate ids_keep
        ids_keep = torch.nonzero(mask==0, as_tuple=True)[1].view(N, -1)

        # Calculate ids_restore
        ids_restore = torch.arange(0, L, device=x.device).unsqueeze(0).repeat(N, 1)

        return ids_keep, ids_restore, mask

    def grid_masking(
        self,
        x: torch.Tensor,
        out_size: Tuple[torch.Tensor, torch.Tensor],
        mask_ratio: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate the block mask for MAE pre-training.

        Args:
            x (torch.Tensor): Image with data augmentation applied, which is
                of shape B x L x C.
            mask_ratio (float): The mask ratio of total blocks.
                Defaults to 0.75.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: masked image, mask
            and the ids to restore original image.

            - ``x_masked`` (torch.Tensor): masked image.
            - ``mask`` (torch.Tensor): mask used to mask image.
            - ``ids_restore`` (torch.Tensor): ids to restore original image.
        """
        # mask_ratio = float(0.5**0.5)
        N, L, D = x.shape  # batch, length, dim
        out_x, out_y = out_size
        # Calculate the block size dynamically
        block_size_x = 2
        block_size_y = 2

        # Calculate the number of blocks in each dimension
        num_blocks_x = out_x // block_size_x
        num_blocks_y = out_y // block_size_y

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.zeros([N, out_x, out_y], device=x.device)
        
        for i in range(N):
            flag=0
            for j in range(num_blocks_x):
                for k in range(num_blocks_y):
                    if flag == 1:  # gaply decide to mask or not
                        mask[i, j * block_size_x : (j + 1) * block_size_x, k * block_size_y : (k + 1) * block_size_y] = 1
                        flag = 0
                    else:
                        flag =1
        
        mask = mask.reshape(N, L)

        # unshuffle to get the binary mask
        # Calculate ids_keep
        ids_keep = torch.nonzero(mask==0, as_tuple=True)[1].view(N, -1)

        # Calculate ids_restore
        ids_restore = torch.arange(0, L, device=x.device).unsqueeze(0).repeat(N, 1)

        return ids_keep, ids_restore, mask

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[bool] = True,
        mask_type: Optional[str]='random'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate features for masked images.

        The function supports two kind of forward behaviors. If the ``mask`` is
        ``True``, the function will generate mask to masking some patches
        randomly and get the hidden features for visible patches, which means
        the function will be executed as masked imagemodeling pre-training;
        if the ``mask`` is ``None`` or ``False``, the forward function will
        call ``super().forward()``, which extract features from images without
        mask.


        Args:
            x (torch.Tensor): Input images, which is of shape B x C x H x W.
            mask (bool, optional): To indicate whether the forward function
                generating ``mask`` or not.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Hidden features,
            mask and the ids to restore original image.

            - ``x`` (torch.Tensor): hidden features, which is of shape
              B x (L * mask_ratio) x C.
            - ``mask`` (torch.Tensor): mask used to mask image.
            - ``ids_restore`` (torch.Tensor): ids to restore original image.
        """
        if mask is None or False:
            return super().forward(x)

        else:
            B, C, H, W = x.shape

            outsize = self.patch_embed.patches_resolution    
            x = self.patch_embed(x)
            if self.mask_type == 'random':
                ids_keep, ids_restore, mask = self.masking_id(B, self.mask_ratio)
            elif self.mask_type == 'block':
                ids_keep, ids_restore, mask = self.block_masking(x, outsize, self.mask_ratio)
            elif self.mask_type == 'grid':
                ids_keep, ids_restore, mask = self.grid_masking(x, outsize, self.mask_ratio)
            
            x = torch.gather(
                x,
                dim=1,
                index=ids_keep[:, :, None, None,
                               None].expand(-1, -1, *x.shape[2:]))

            for blk in self.blocks[:-self.num_main_blocks]:
                x = blk(x)

            x = x[..., 0, 0, :]
            if self.ape:
                pos_embed = self.interpolate_pos_encoding(x, H, W)
                pos_embed = torch.gather(
                    pos_embed.expand(B, -1, -1),
                    dim=1,
                    index=ids_keep[:, :, None].expand(-1, -1,
                                                      pos_embed.shape[2]),
                )
                x = x + pos_embed
            x = self.pos_drop(x)

            for blk in self.blocks[-self.num_main_blocks:]:
                x = blk(x)

            return (x, mask, ids_restore)



# class FCMAE(ConvNeXt):
#     """CNN for MAE pre-training.

#     A PyTorch implement of: `An Image is Worth 16x16 Words: Transformers
#     for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.
#     This module implements the patch masking in MAE and initialize the
#     position embedding with sine-cosine position embedding.

#     Args:
#         arch (str | dict): Vision Transformer architecture
#             Default: 'b'
#         img_size (int | tuple): Input image size
#         patch_size (int | tuple): The patch size
#         out_indices (Sequence | int): Output from which stages.
#             Defaults to -1, means the last stage.
#         drop_rate (float): Probability of an element to be zeroed.
#             Defaults to 0.
#         drop_path_rate (float): stochastic depth rate. Defaults to 0.
#         norm_cfg (dict): Config dict for normalization layer.
#             Defaults to ``dict(type='LN')``.
#         final_norm (bool): Whether to add a additional layer to normalize
#             final feature map. Defaults to True.
#         out_type (str): The type of output features. Please choose from

#             - ``"cls_token"``: The class token tensor with shape (B, C).
#             - ``"featmap"``: The feature map tensor from the patch tokens
#               with shape (B, C, H, W).
#             - ``"avg_featmap"``: The global averaged feature map tensor
#               with shape (B, C).
#             - ``"raw"``: The raw feature tensor includes patch tokens and
#               class tokens with shape (B, L, C).

#             It only works without input mask. Defaults to ``"avg_featmap"``.
#         interpolate_mode (str): Select the interpolate mode for position
#             embeding vector resize. Defaults to "bicubic".
#         patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
#         layer_cfgs (Sequence | dict): Configs of each transformer layer in
#             encoder. Defaults to an empty dict.
#         mask_ratio (bool): The ratio of total number of patches to be masked.
#             Defaults to 0.75.
#         init_cfg (Union[List[dict], dict], optional): Initialization config
#             dict. Defaults to None.
#     """

#     def __init__(self,
#                  arch: Union[str, dict] = 'b',
#                  in_channels: int =3,
#                  stem_patch_size: int =4,
#                  norm_cfg: dict = dict(type='LN2d', eps=1e-6),
#                  act_cfg: dict =dict(type='GELU'),
#                  linear_pw_conv: bool =True,
#                  use_grn: bool =False,
#                  drop_path_rate: float =0.,
#                  layer_scale_init_value=1e-6,
#                  out_indices=-1,
#                  frozen_stages=0,
#                  gap_before_final_norm=True,
#                  with_cp=False,
#                  init_cfg=[
#                      dict(
#                          type='TruncNormal',
#                          layer=['Conv2d', 'Linear'],
#                          std=.02,
#                          bias=0.),
#                      dict(
#                          type='Constant', layer=['LayerNorm'], val=1.,
#                          bias=0.),
#                  ]) -> None:
#         super().__init__(
#             arch=arch,
#             in_channels=in_channels,
#             stem_patch_size=stem_patch_size,
#             norm_cfg=norm_cfg,
#             act_cfg=act_cfg,
#             linear_pw_conv=linear_pw_conv,
#             use_grn=use_grn,
#             drop_path_rate=drop_path_rate,
#             layer_scale_init_value=layer_scale_init_value,
#             out_indices=out_indices,
#             frozen_stages=frozen_stages,
#             gap_before_final_norm=gap_before_final_norm,
#             with_cp=with_cp,
#             init_cfg=init_cfg,
#         )

#         # position embedding is not learnable during pretraining
#         self.pos_embed.requires_grad = False
#         self.mask_ratio = mask_ratio
#         self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]

#     def init_weights(self) -> None:
#         """Initialize position embedding, patch embedding and cls token."""
#         super().init_weights()

#     def random_masking(
#         self,
#         x: torch.Tensor,
#         mask_ratio: float = 0.75
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """Generate the random mask for MAE Pre-training.

#         Args:
#             x (torch.Tensor): Image with data augmentation applied, which is
#                 of shape B x L x C.
#             mask_ratio (float): The mask ratio of total patches.
#                 Defaults to 0.75.

#         Returns:
#             Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: masked image, mask
#             and the ids to restore original image.

#             - ``x_masked`` (torch.Tensor): masked image.
#             - ``mask`` (torch.Tensor): mask used to mask image.
#             - ``ids_restore`` (torch.Tensor): ids to restore original image.
#         """
#         N, L, D = x.shape  # batch, length, dim
#         len_keep = int(L * (1 - mask_ratio))

#         noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

#         # sort noise for each sample
#         ids_shuffle = torch.argsort(
#             noise, dim=1)  # ascend: small is keep, large is remove
#         ids_restore = torch.argsort(ids_shuffle, dim=1)

#         # keep the first subset
#         ids_keep = ids_shuffle[:, :len_keep]
#         x_masked = torch.gather(
#             x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

#         # generate the binary mask: 0 is keep, 1 is remove
#         mask = torch.ones([N, L], device=x.device)
#         mask[:, :len_keep] = 0
#         # unshuffle to get the binary mask
#         mask = torch.gather(mask, dim=1, index=ids_restore)

#         return x_masked, mask, ids_restore

#     def patchify(self, imgs):
#         """
#         imgs: (N, 3, H, W)
#         x: (N, L, patch_size**2 *3)
#         """
#         p = self.patch_resolution[0]
#         assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

#         h = w = imgs.shape[2] // p
#         x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
#         x = torch.einsum('nchpwq->nhwpqc', x)
#         x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
#         return x

#     def unpatchify(self, x):
#         """
#         x: (N, L, patch_size**2 *3)
#         imgs: (N, 3, H, W)
#         """
#         p = self.patch_resolution[0]
#         h = w = int(x.shape[1]**.5)
#         assert h * w == x.shape[1]
        
#         x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
#         x = torch.einsum('nhwpqc->nchpwq', x)
#         imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
#         return imgs


#     def block_masking(
#         self,
#         x: torch.Tensor,
#         out_size: Tuple[torch.Tensor, torch.Tensor],
#         mask_ratio: float = 0.75
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """Generate the block mask for MAE pre-training.

#         Args:
#             x (torch.Tensor): Image with data augmentation applied, which is
#                 of shape B x L x C.
#             mask_ratio (float): The mask ratio of total blocks.
#                 Defaults to 0.75.

#         Returns:
#             Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: masked image, mask
#             and the ids to restore original image.

#             - ``x_masked`` (torch.Tensor): masked image.
#             - ``mask`` (torch.Tensor): mask used to mask image.
#             - ``ids_restore`` (torch.Tensor): ids to restore original image.
#         """
#         N, L, D = x.shape  # batch, length, dim
#         out_x, out_y = out_size
#         # Calculate the block size dynamically
#         block_size_x = int(out_x * mask_ratio)
#         block_size_y = int(out_y * mask_ratio)

#         start_x = torch.randint(0, out_x-block_size_x -1)
#         start_y = torch.randint(0, out_x-block_size_y -1)


#         # generate the binary mask: 0 is keep, 1 is remove
#         mask = torch.zeros([N, out_x, out_y], device=x.device)
#         mask[:,start_x:(start_x+block_size_x) ,start_y:(start_y+block_size_y)] =1
#         mask = mask.reshape(N, L)

#         # unshuffle to get the binary mask
#         ids_remove = torch.nonzero(mask, as_tuple=True)
#         ids_remove = ids_remove[1].view(mask.shape)  # reshape to match the original shape

#         # Sort the indices to get the restore order
#         ids_restore = torch.argsort(ids_remove, dim=1)
#         x_masked = torch.gather(
#             x, dim=1, index=ids_remove.unsqueeze(-1).repeat(1, 1, D))

#         return x_masked, mask, ids_restore

#     def forward(
#         self,
#         x: torch.Tensor,
#         mask: Optional[bool] = True,
#         mask_type: Optional[str]='random'
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """Generate features for masked images.

#         The function supports two kind of forward behaviors. If the ``mask`` is
#         ``True``, the function will generate mask to masking some patches
#         randomly and get the hidden features for visible patches, which means
#         the function will be executed as masked imagemodeling pre-training;
#         if the ``mask`` is ``None`` or ``False``, the forward function will
#         call ``super().forward()``, which extract features from images without
#         mask.


#         Args:
#             x (torch.Tensor): Input images, which is of shape B x C x H x W.
#             mask (bool, optional): To indicate whether the forward function
#                 generating ``mask`` or not.

#         Returns:
#             Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Hidden features,
#             mask and the ids to restore original image.

#             - ``x`` (torch.Tensor): hidden features, which is of shape
#               B x (L * mask_ratio) x C.
#             - ``mask`` (torch.Tensor): mask used to mask image.
#             - ``ids_restore`` (torch.Tensor): ids to restore original image.
#         """
#         if mask is None or False:
#             return super().forward(x)

#         else:
#             B = x.shape[0]
#             x, outsize = self.patch_embed(x)
#             # add pos embed w/o cls token
#             x = x + self.pos_embed[:, 1:, :]

#             # masking: length -> length * mask_ratio
#             if mask_type == 'random':
#                 x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
#             elif mask_type =='block':
#                 x, mask, ids_restore = self.block_masking(x, self.mask_ratio)
#             elif mask_type =='mulblock':
#                 x, mask, ids_restore = self.mulblock_masking(x, self.mask_ratio)
#             # append cls token
#             cls_token = self.cls_token + self.pos_embed[:, :1, :]
#             cls_tokens = cls_token.expand(B, -1, -1)
#             x = torch.cat((cls_tokens, x), dim=1)

#             for _, layer in enumerate(self.layers):
#                 x = layer(x)
#             # Use final norm
#             x = self.norm1(x)

#             return (x, mask, ids_restore)
