# 划分训练集/验证集/测试集，并生成文件名列表
# 导入需要用到的库

import random
import os.path as osp

import cv2
import numpy as np
import paddle
import paddlers as pdrs
from paddlers import transforms as T
from matplotlib import pyplot as plt
from PIL import Image
import random
import os.path as osp
from os import listdir

import cv2


# 随机数生成器种子
RNG_SEED = 56961
# 调节此参数控制训练集数据的占比
TRAIN_RATIO = 0.9
# 数据集路径
DATA_DIR = '/home/aistudio/data/data56961/massroad'

# 分割类别
CLASSES = (
    'background',
    'road',
)

def write_rel_paths(phase, names, out_dir, prefix):
    """将文件相对路径存储在txt格式文件中"""
    with open(osp.join(out_dir, phase+'.txt'), 'w') as f:
        for name in names:
            f.write(
                ' '.join([
                    osp.join(prefix, 'input', name),
                    osp.join(prefix, 'output', name)
                ])
            )
            f.write('\n')


random.seed(RNG_SEED)

train_prefix = osp.join('road_segmentation_ideal', 'training')
test_prefix = osp.join('road_segmentation_ideal', 'testing')
train_names = listdir(osp.join(DATA_DIR, train_prefix, 'output'))
train_names = list(filter(lambda n: n.endswith('.png'), train_names))
test_names = listdir(osp.join(DATA_DIR, test_prefix, 'output'))
test_names = list(filter(lambda n: n.endswith('.png'), test_names))
# 对文件名进行排序，以确保多次运行结果一致
train_names.sort()
test_names.sort()
random.shuffle(train_names)
len_train = int(len(train_names)*TRAIN_RATIO)
write_rel_paths('train', train_names[:len_train], DATA_DIR, train_prefix)
write_rel_paths('val', train_names[len_train:], DATA_DIR, train_prefix)
write_rel_paths('test', test_names, DATA_DIR, test_prefix)

# 写入类别信息
with open(osp.join(DATA_DIR, 'labels.txt'), 'w') as f:
    for cls in CLASSES:
        f.write(cls+'\n')

print("数据集划分已完成。")

# 将GT中的255改写为1，便于训练

import os.path as osp
from glob import glob

import cv2
from tqdm import tqdm


# 数据集路径
DATA_DIR = '/home/aistudio/data/data56961/massroad'

train_prefix = osp.join('road_segmentation_ideal', 'training')
test_prefix = osp.join('road_segmentation_ideal', 'testing')

train_paths = glob(osp.join(DATA_DIR, train_prefix, 'output', '*.png'))
test_paths = glob(osp.join(DATA_DIR, test_prefix, 'output', '*.png'))
for path in tqdm(train_paths+test_paths):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    im[im>0] = 1
    # 原地改写
    cv2.imwrite(path, im)

# 随机种子
SEED = 56961
# 数据集存放目录
DATA_DIR = '/home/aistudio/data/data56961/massroad/'
# 训练集`file_list`文件路径
TRAIN_FILE_LIST_PATH = '/home/aistudio/data/data56961/massroad/train.txt'
# 验证集`file_list`文件路径
VAL_FILE_LIST_PATH = '/home/aistudio/data/data56961/massroad/val.txt'
# 测试集`file_list`文件路径
TEST_FILE_LIST_PATH = '/home/aistudio/data/data56961/massroad/test.txt'
# 数据集类别信息文件路径
LABEL_LIST_PATH = '/home/aistudio/data/data56961/massroad/labels.txt'
# 实验目录，保存输出的模型权重和结果
EXP_DIR =  '/home/aistudio/exp/'

# 固定随机种子，尽可能使实验结果可复现

random.seed(SEED)
np.random.seed(SEED)
paddle.seed(SEED)

# 构建数据集

# 定义训练和验证时使用的数据变换（数据增强、预处理等）
train_transforms = T.Compose([
    # 随机裁剪
    T.RandomCrop(crop_size=512),
    # 以50%的概率实施随机水平翻转
    T.RandomHorizontalFlip(prob=0.5),
    # 以50%的概率实施随机垂直翻转
    T.RandomVerticalFlip(prob=0.5),
    # 将数据归一化到[-1,1]
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

eval_transforms = T.Compose([
    T.Resize(target_size=1488),
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

# 构建优化器
optimizer = paddle.optimizer.Adam(
    learning_rate=0.001,
    parameters=model.net.parameters()
)

# 执行模型训练
model.train(
    num_epochs=200,
    train_dataset=train_dataset,
    train_batch_size=8,
    eval_dataset=val_dataset,
    optimizer=optimizer,
    save_interval_epochs=1,
    # 每多少次迭代记录一次日志
    log_interval_steps=30,
    save_dir=EXP_DIR,
    # 是否使用early stopping策略，当精度不再改善时提前终止训练
    early_stop=False,
    # 是否启用VisualDL日志功能
    use_vdl=True,
    # 指定从某个检查点继续训练
    resume_checkpoint=None
)