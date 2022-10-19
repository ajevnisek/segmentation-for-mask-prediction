from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.DIR = "ckpt/Hday2night-DeepLabV3-resnet34"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.name = 'HCOCO'
_C.DATASET.root_dataset = ".../data/Image_Harmonization_Dataset/"
_C.DATASET.list_train = "./data/Hday2night-training.odgt"
_C.DATASET.list_val = "./data/Hday2night-validation.odgt"
_C.DATASET.num_class = 2
# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# architecture of net_encoder
_C.MODEL.arch = "DeepLabV3"
_C.MODEL.encoder_name = "resnet34"

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.batch_size_per_gpu = 2
# epochs to train for
_C.TRAIN.num_epochs = 20
# epoch to start training. useful if continue from a checkpoint
_C.TRAIN.optim = 'Adam'
_C.TRAIN.lr = 1e-4

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()
# currently only supports 1
_C.VAL.batch_size = 16
# output visualization during validation
_C.VAL.visualize = False
# the checkpoint to evaluate on
_C.VAL.checkpoint = "epoch_20.pth"

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
_C.TEST = CN()
# currently only supports 1
_C.TEST.batch_size = 16
# the checkpoint to test on
_C.TEST.checkpoint = "epoch_20.pth"
# folder to output visualization results
_C.TEST.result = "./"
