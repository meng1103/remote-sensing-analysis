# 划分训练集/验证集/测试集，并生成文件名列表
# 注意，作为演示，本项目仅使用原数据集的训练集，即用来测试的数据也来自原数据集的训练集

import random
import os.path as osp
from os import listdir
import random
import os.path as osp

import cv2
import numpy as np
import paddle
import paddlers as pdrs
from paddlers import transforms as T
from matplotlib import pyplot as plt
from PIL import Image
import cv2


# 随机数生成器种子
RNG_SEED = 77571
# 调节此参数控制训练集数据的占比
TRAIN_RATIO = 0.9
# 调节此参数控制验证集数据的占比
VAL_RATIO = 0.05
# 使用的样本个数（选取排序靠前的样本）
NUM_SAMPLES_TO_USE = 10000
# 数据集路径
DATA_DIR = '/home/aistudio/data/data77571/dataset/'

# 分割类别
CLASSES = (
    'cls0',
    'cls1',
    'cls2',
    'cls3',
    'bg'
)


def reset_pixels(name):
    path = osp.join(DATA_DIR, 'lab_train', name)
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    im[im==255] = CLASSES.index('bg')
    cv2.imwrite(path, im)


def write_rel_paths(phase, names, out_dir):
    """将文件相对路径存储在txt格式文件中"""
    with open(osp.join(out_dir, phase+'.txt'), 'w') as f:
        for name in names:
            f.write(
                ' '.join([
                    osp.join('img_train', name.replace('.png', '.jpg')),
                    osp.join('lab_train', name)
                ])
            )
            f.write('\n')


random.seed(RNG_SEED)

names = listdir(osp.join(DATA_DIR, 'lab_train'))
# 将值为255的无效像素重设为背景类
for name in names:
    reset_pixels(name)
# 对文件名进行排序，以确保多次运行结果一致
names.sort()
if NUM_SAMPLES_TO_USE is not None:
    names = names[:NUM_SAMPLES_TO_USE]
random.shuffle(names)
len_train = int(len(names)*TRAIN_RATIO)
len_val = int(len(names)*VAL_RATIO)
write_rel_paths('train', names[:len_train], DATA_DIR)
write_rel_paths('val', names[len_train:len_train+len_val], DATA_DIR)
write_rel_paths('test', names[len_train+len_val:], DATA_DIR)

# 写入类别信息
with open(osp.join(DATA_DIR, 'labels.txt'), 'w') as f:
    for cls in CLASSES:
        f.write(cls+'\n')

print("数据集划分已完成。")

# 随机种子
SEED = 77571
# 数据集存放目录
DATA_DIR = '/home/aistudio/data/data77571/dataset/'
# 训练集`file_list`文件路径
TRAIN_FILE_LIST_PATH = '/home/aistudio/data/data77571/dataset/train.txt'
# 验证集`file_list`文件路径
VAL_FILE_LIST_PATH = '/home/aistudio/data/data77571/dataset/val.txt'
# 测试集`file_list`文件路径
TEST_FILE_LIST_PATH = '/home/aistudio/data/data77571/dataset/test.txt'
# 数据集类别信息文件路径
LABEL_LIST_PATH = '/home/aistudio/data/data77571/dataset/labels.txt'
# 实验目录，保存输出的模型权重和结果
EXP_DIR =  '/home/aistudio/exp/'

# 固定随机种子，尽可能使实验结果可复现

random.seed(SEED)
np.random.seed(SEED)
paddle.seed(SEED)


# 构建数据集

# 定义训练和验证时使用的数据变换（数据增强、预处理等）
train_transforms = T.Compose([
    # 将影像缩放到256x256大小
    T.Resize(target_size=256),
    # 以50%的概率实施随机水平翻转
    T.RandomHorizontalFlip(prob=0.5),
    # 将数据归一化到[-1,1]
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

eval_transforms = T.Compose([
    T.Resize(target_size=256),
    # 验证阶段与训练阶段的数据归一化方式必须相同
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# 分别构建训练和验证所用的数据集
train_dataset = pdrs.datasets.SegDataset(
    data_dir=DATA_DIR,
    file_list=TRAIN_FILE_LIST_PATH,
    label_list=LABEL_LIST_PATH,
    transforms=train_transforms,
    num_workers=4,
    shuffle=True
)

val_dataset = pdrs.datasets.SegDataset(
    data_dir=DATA_DIR,
    file_list=VAL_FILE_LIST_PATH,
    label_list=LABEL_LIST_PATH,
    transforms=eval_transforms,
    num_workers=0,
    shuffle=False
)

# 构建DeepLab V3+模型，使用ResNet-50作为backbone
model = pdrs.tasks.DeepLabV3P(
    input_channel=3,
    num_classes=len(train_dataset.labels),
    backbone='ResNet50_vd'
)
model.net_initialize(
    pretrain_weights='CITYSCAPES',
    save_dir=osp.join(EXP_DIR, 'pretrain'),
    resume_checkpoint=None,
    is_backbone_weights=False
)

# 使用focal loss作为损失函数
model.losses = dict(
    types=[pdrs.models.ppseg.models.FocalLoss()],
    coef=[1.0]
)

# 制定定步长学习率衰减策略
lr_scheduler = paddle.optimizer.lr.StepDecay(
    0.001,
    step_size=8000,
    gamma=0.5
)
# 构造Adam优化器
optimizer = paddle.optimizer.Adam(
    learning_rate=lr_scheduler,
    parameters=model.net.parameters()
)


# 执行模型训练
model.train(
    num_epochs=60,
    train_dataset=train_dataset,
    train_batch_size=16,
    eval_dataset=val_dataset,
    optimizer=optimizer,
    save_interval_epochs=3,
    # 每多少次迭代记录一次日志
    log_interval_steps=100,
    save_dir=EXP_DIR,
    # 是否使用early stopping策略，当精度不再改善时提前终止训练
    early_stop=False,
    # 是否启用VisualDL日志功能
    use_vdl=True,
    # 指定从某个检查点继续训练
    resume_checkpoint=None
)