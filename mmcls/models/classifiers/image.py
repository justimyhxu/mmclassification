import torch.nn as nn

from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from .base import BaseClassifier


@CLASSIFIERS.register_module()
class ImageClassifier(BaseClassifier):

    def __init__(self, backbone, neck=None, head=None, pretrained=None, frozen_backbone=True):
        super(ImageClassifier, self).__init__()
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)

        self.init_weights(pretrained=pretrained, frozen_backbone=frozen_backbone)

    def init_weights(self, pretrained=None, frozen_backbone=False):
        super(ImageClassifier, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained, frozen=frozen_backbone)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_head:
            self.head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            gt_label (Tensor): of shape (N, 1) encoding the ground-truth label
                of input images.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # import ipdb
        # ipdb.set_trace()
        ncrop = img.shape[2] // 224
        img_crop = img.reshape(img.shape[0], 3, ncrop, 224, 224)
        img_crop = img_crop.permute(0,2, 1,3,4).reshape(-1,3,224,224)
        x = self.extract_feat(img_crop)

        losses = dict()
        loss = self.head.forward_train(x, gt_label)
        losses.update(loss)

        return losses

    def simple_test(self, img):
        """Test without augmentation."""
        x = self.extract_feat(img)
        return self.head.simple_test(x)
