# 划分训练集/验证集/测试集，并生成文件名列表
# 所有样本从RSOD数据集的playground子集中选取
# 导入需要用到的库

import random
import os.path as osp

import cv2
import numpy as np
import paddle
import paddlers as pdrs
from paddlers import transforms as T
from paddlers.tasks.utils.visualize import visualize_detection
from matplotlib import pyplot as plt
from PIL import Image
import random
import os.path as osp
from os import listdir


# 随机数生成器种子
RNG_SEED = 52980
# 调节此参数控制训练集数据的占比
TRAIN_RATIO = 0.9
# 调节此参数控制验证集数据的占比
VAL_RATIO = 0.05
# 数据集路径
DATA_DIR = '/home/aistudio/data/data52980/rsod/'

# 目标类别
CLASS = 'playground'


def write_rel_paths(phase, names, out_dir):
    """将文件相对路径存储在txt格式文件中"""
    with open(osp.join(out_dir, phase+'.txt'), 'w') as f:
        for name in names:
            f.write(
                ' '.join([
                    osp.join(CLASS, 'JPEGImages', name),
                    osp.join(CLASS, 'Annotation', 'xml', name.replace('.jpg', '.xml'))
                ])
            )
            f.write('\n')


random.seed(RNG_SEED)

names = listdir(osp.join(DATA_DIR, CLASS, 'JPEGImages'))
# 对文件名进行排序，以确保多次运行结果一致
names.sort()
random.shuffle(names)
len_train = int(len(names)*TRAIN_RATIO)
len_val = int(len(names)*VAL_RATIO)
write_rel_paths('train', names[:len_train], DATA_DIR)
write_rel_paths('val', names[len_train:len_train+len_val], DATA_DIR)
write_rel_paths('test', names[len_train+len_val:], DATA_DIR)

# 写入类别信息
with open(osp.join(DATA_DIR, 'labels.txt'), 'w') as f:
    f.write(CLASS+'\n')

print("数据集划分已完成。")

# 定义全局变量

# 随机种子
SEED = 52980
# 数据集存放目录
DATA_DIR = '/home/aistudio/data/data52980/rsod/'
# 训练集`file_list`文件路径
TRAIN_FILE_LIST_PATH = '/home/aistudio/data/data52980/rsod/train.txt'
# 验证集`file_list`文件路径
VAL_FILE_LIST_PATH = '/home/aistudio/data/data52980/rsod/val.txt'
# 测试集`file_list`文件路径
TEST_FILE_LIST_PATH = '/home/aistudio/data/data52980/rsod/test.txt'
# 数据集类别信息文件路径
LABEL_LIST_PATH = '/home/aistudio/data/data52980/rsod/labels.txt'
# 实验目录，保存输出的模型权重和结果
EXP_DIR =  '/home/aistudio/exp/'
# 目标类别
CLASS = 'playground'
# 模型验证阶段输入影像尺寸
INPUT_SIZE = 608
# 固定随机种子，尽可能使实验结果可复现

random.seed(SEED)
np.random.seed(SEED)
paddle.seed(SEED)
# 构建数据集

train_transforms = T.Compose([
    # 对输入影像施加随机色彩扰动
    T.RandomDistort(),
    # 在影像边界进行随机padding
    T.RandomExpand(),
    # 随机裁剪，裁块大小在一定范围内变动
    T.RandomCrop(),
    # 随机水平翻转
    T.RandomHorizontalFlip(),
    # 对batch进行随机缩放，随机选择插值方式
    T.BatchRandomResize(
        target_sizes=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608],
        interp='RANDOM'
    ),
    # 影像归一化
    T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
])

eval_transforms = T.Compose([
    # 使用双三次插值将输入影像缩放到固定大小
    T.Resize(
        target_size=INPUT_SIZE, interp='CUBIC'
    ),
    # 验证阶段与训练阶段的归一化方式必须相同
    T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
])

# 分别构建训练和验证所用的数据集
train_dataset = pdrs.datasets.VOCDetection(
    data_dir=DATA_DIR,
    file_list=TRAIN_FILE_LIST_PATH,
    label_list=LABEL_LIST_PATH,
    transforms=train_transforms,
    shuffle=True
)

val_dataset = pdrs.datasets.VOCDetection(
    data_dir=DATA_DIR,
    file_list=VAL_FILE_LIST_PATH,
    label_list=LABEL_LIST_PATH,
    transforms=eval_transforms,
    shuffle=False
)

# 构建PP-YOLO模型
model = pdrs.tasks.PPYOLO(num_classes=len(train_dataset.labels))
model.net_initialize(
    pretrain_weights='COCO',
    save_dir=osp.join(EXP_DIR, 'pretrain'),
    resume_checkpoint=None,
    is_backbone_weights=False
)

# 执行模型训练
model.train(
    num_epochs=200,
    train_dataset=train_dataset,
    train_batch_size=8,
    eval_dataset=val_dataset,
    # 每多少个epoch存储一次检查点
    save_interval_epochs=10,
    # 每多少次迭代记录一次日志
    log_interval_steps=10,
    save_dir=EXP_DIR,
    # 指定预训练权重
    pretrain_weights='COCO',
    # 初始学习率大小
    learning_rate=0.0001,
    # 学习率预热（learning rate warm-up）步数与初始值
    warmup_steps=0,
    warmup_start_lr=0.0,
    # 是否启用VisualDL日志功能
    use_vdl=True
)