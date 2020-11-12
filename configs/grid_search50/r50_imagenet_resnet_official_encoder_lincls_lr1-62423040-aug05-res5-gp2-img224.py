_base_ = [
    './_base_/datasets/imagenet_bs32.py',
    './_base_/schedules/imagenet_bs256.py', './_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    pretrained='data/resnet50-encoder-pretrained-62423040.pth',
    frozen_backbone=True,
    backbone=dict(
        type='ResNetOfficial',
        depth = 50,
        filter_max=2048,
        with_fpn=False,
        with_ds_fuse=False,
        multi_level=False,
        norm_eval=True,
        ),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512*4*4,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
# dataset settings
dataset_type = 'ImageNet'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_norm_cfg = dict(
    mean=[255/2., 255/2., 255/2.,], std=[255/2., 255/2., 255/2.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224, scale=(0.5, 1.0), ratio=(3.00/4, 4.00/3)),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=256),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
data = dict(
    samples_per_gpu=64, # 8*64 512
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_prefix='data/imagenet/train',
        ann_file='data/imagenet/meta/train_map.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val_map.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val_map.txt',
        pipeline=test_pipeline))


# checkpoint saving
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

evaluation = dict(interval=1, metric='accuracy')
# optimizer
optimizer = dict(type='SGD', lr=1, momentum=0.9, weight_decay=0)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[60, 80])
total_epochs = 100
