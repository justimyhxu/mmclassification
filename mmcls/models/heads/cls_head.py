from mmcls.models.losses import Accuracy
from ..builder import HEADS, build_loss
from .base_head import BaseHead


@HEADS.register_module()
class ClsHead(BaseHead):
    """classification head.

    Args:
        loss (dict): Config of classification loss.
        topk (int | tuple): Top-k accuracy.
    """  # noqa: W605

    def __init__(self,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, )):
        super(ClsHead, self).__init__()

        assert isinstance(loss, dict)
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk

        self.compute_loss = build_loss(loss)
        self.compute_accuracy = Accuracy(topk=self.topk)

    def loss(self, cls_score, gt_label):
        if cls_score.shape[0] % 144 == 0 and False:
            nc = 144
        elif cls_score.shape[0] % 36 == 0 or True:
            nc = 36
        elif cls_score.shape[0] % 10 == 0 or True:
            nc = 10
        elif cls_score.shape[0] % 6 == 0:
            nc = 6
        cls_score = cls_score.reshape(-1, nc, 1000)
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        loss = self.compute_loss(cls_score.mean(dim=1), gt_label, avg_factor=num_samples)
        # compute accuracy
        cls_score = cls_score.softmax(dim=-1)
        cls_score = cls_score.mean(dim=1)
        acc = self.compute_accuracy(cls_score, gt_label)
        assert len(acc) == len(self.topk)
        losses['loss'] = loss
        losses['accuracy'] = {f'top-{k}': a for k, a in zip(self.topk, acc)}
        losses['num_samples'] = loss.new(1).fill_(num_samples)
        losses['cls_score'] = cls_score
        losses['label'] = gt_label
        return losses

    def forward_train(self, cls_score, gt_label):
        losses = self.loss(cls_score, gt_label)
        return losses
