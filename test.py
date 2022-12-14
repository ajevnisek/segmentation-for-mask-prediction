import os
import argparse

import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from pprint import pprint
from torch.utils.data import DataLoader

from config import cfg
from model import MaskModel
from dataset import ImageHarmonizationMaskDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        default="0-3",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    return cfg


cfg = parse_args()
head, tail = os.path.split(cfg.DIR)
dataset = tail.split('-')[0]
model_name = cfg.MODEL.arch
encoder_name = cfg.MODEL.encoder_name

versions_root = os.path.join('ckpt',
    f"{dataset}-{model_name}-{encoder_name}",
    'lightning_logs',)
all_versions = os.listdir(versions_root)
all_versions.sort(key=lambda x: int(x.split("_")[-1]))
logs_root = os.path.join(versions_root, all_versions[-1])

model_ckpt_dir = os.path.join(logs_root,
                               'checkpoints',)
model_ckpt_path = os.path.join(model_ckpt_dir,
                               os.listdir(model_ckpt_dir)[-1])


os.makedirs(os.path.join(logs_root, 'images'), exist_ok=True)
# download data
root = "../data/Image_Harmonization_Dataset/"

try:
    os.makedirs(cfg.DIR, exist_ok=True)
except:
    if not os.path.exists(cfg.DIR):
        head, tail = os.path.split(cfg.DIR)
        cfg.DIR = os.path.join('ckpt', tail)


# download data
dataset = cfg.DATASET.name
train_odgt = f'./data/{dataset}-training.odgt'
test_odgt = f'./data/{dataset}-validation.odgt'

# init train, val, test sets

if not os.path.exists(cfg.DATASET.root_dataset):
    cfg.DATASET.root_dataset = '../data/Image_Harmonization_Dataset/'


# init train, val, test sets
train_dataset = ImageHarmonizationMaskDataset(cfg.DATASET.root_dataset,
                                              train_odgt, "train")
valid_dataset = ImageHarmonizationMaskDataset(cfg.DATASET.root_dataset,
                                              test_odgt, "valid")
test_dataset = ImageHarmonizationMaskDataset(cfg.DATASET.root_dataset,
                                             test_odgt, "test")

print(f"Train size: {len(train_dataset)}")
print(f"Valid size: {len(valid_dataset)}")
print(f"Test size: {len(test_dataset)}")

n_cpu = os.cpu_count()
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=n_cpu)
valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)

# lets look at some samples

sample = train_dataset[0]
plt.subplot(1,2,1)
plt.imshow(sample["image"].transpose(1, 2, 0)) # for visualization we have to transpose back to HWC
plt.subplot(1,2,2)
plt.imshow(sample["mask"].squeeze())  # for visualization we have to remove 3rd dimension of mask
plt.savefig(os.path.join(logs_root, 'images', 'train.png'))

sample = valid_dataset[0]
plt.subplot(1,2,1)
plt.imshow(sample["image"].transpose(1, 2, 0)) # for visualization we have to transpose back to HWC
plt.subplot(1,2,2)
plt.imshow(sample["mask"].squeeze())  # for visualization we have to remove 3rd dimension of mask
plt.savefig(os.path.join(logs_root, 'images', 'validation.png'))

sample = test_dataset[0]
plt.subplot(1,2,1)
plt.imshow(sample["image"].transpose(1, 2, 0)) # for visualization we have to transpose back to HWC
plt.subplot(1,2,2)
plt.imshow(sample["mask"].squeeze())  # for visualization we have to remove 3rd dimension of mask
plt.savefig(os.path.join(logs_root, 'images', 'test.png'))



model = MaskModel(arch=model_name, encoder_name=encoder_name, in_channels=3,
                  out_classes=1, optimizer_type='SGD', learning_rate=1e-4)
model.load_state_dict(torch.load(model_ckpt_path)['state_dict'])
trainer = pl.Trainer(
    gpus=1,
    max_epochs=5,
)
test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
pprint(test_metrics)

for batch in test_dataloader:
    with torch.no_grad():
        model.eval()
        logits = model(batch["image"])
    pr_masks = logits.sigmoid()

    os.makedirs(os.path.join(logs_root, 'test_images'), exist_ok=True)

    for idx, (image, gt_mask, pr_mask, name) in enumerate(zip(
            batch["image"], batch["mask"], pr_masks, batch["name"])):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(image.numpy().transpose(1, 2, 0))  # convert CHW -> HWC
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask.numpy().squeeze()) # just squeeze classes dim, because we have only one class
        plt.title("Ground truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pr_mask.numpy().squeeze()) # just squeeze classes dim, because we have only one class
        plt.title("Prediction")
        plt.axis("off")

        plt.savefig(os.path.join(logs_root, 'test_images',
                                 f'prediction{idx:03d}_{name}.png'))

