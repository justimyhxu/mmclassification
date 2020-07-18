import os
import os.path as osp
import mmcv
from mmcv.runner.utils import obj_from_dict
from mmcv.runner.checkpoint import save_checkpoint
import torch
from .parameters import parameters


class Runner(mmcv.runner.Runner):
    """A training helper for PyTorch.

        Custom version of mmcv runner, overwrite init_optimizer method
    """

    def init_optimizer(self, optimizer):
        """Init the optimizer.

        Args:
            optimizer (dict or :obj:`~torch.optim.Optimizer`): Either an
                optimizer object or a dict used for constructing the optimizer.

        Returns:
            :obj:`~torch.optim.Optimizer`: An optimizer object.

        Examples:
            >>> optimizer = dict(type='SGD', lr=0.01, momentum=0.9)
            >>> type(runner.init_optimizer(optimizer))
            <class 'torch.optim.sgd.SGD'>
        """
        if isinstance(optimizer, dict):
            optimizer = obj_from_dict(
                optimizer, torch.optim,
                dict(params=parameters(self.model, optimizer.lr)))
        elif not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                'optimizer must be either an Optimizer object or a dict, '
                'but got {}'.format(type(optimizer)))
        return optimizer

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None):
        """remove symlink to avoid error in windows file system"""
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        linkpath = osp.join(out_dir, 'latest.pth')
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # use relative symlink
        try:
            mmcv.symlink(filename, linkpath)
        except:
            print('Failed to symlink from {} to {}.'.format(filename, linkpath))

    def resume(self, checkpoint, resume_optimizer=True,
               map_location='default', iter_ratio=1):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter'] * iter_ratio
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)

    def auto_resume(self, iter_ratio=1):
        linkname = osp.join(self.work_dir, 'latest.pth')
        if osp.exists(linkname):
            self.logger.info('latest checkpoint found')
            self.resume(linkname, iter_ratio=iter_ratio)
        else:
            # TODO: a more flexible way to auto_resume
            latest_epoch = 0
            latest_name = None
            for root, dirs, files in os.walk(self.work_dir, topdown=True):
                for name in files:
                    if 'epoch_' in name:
                        epoch_num = int(name[name.find('_') + 1:name.find('.pth')])
                        latest_name = name if epoch_num > latest_epoch else latest_name
                        latest_epoch = epoch_num if epoch_num > latest_epoch else latest_epoch

            if latest_name is not None and latest_epoch > 0:
                filename = osp.join(self.work_dir, latest_name)
                assert osp.exists(filename), '{} does not exist!'.format(filename)
                self.logger.info('latest checkpoint {} found'.format(latest_name))
                self.resume(filename, iter_ratio=iter_ratio)
