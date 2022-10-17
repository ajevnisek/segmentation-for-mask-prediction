import os
import torch
import argparse
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from pprint import pprint
from torch.utils.data import DataLoader

from config import cfg
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

# download data
root = "../data/Image_Harmonization_Dataset/"
dataset = 'Hday2night'
train_odgt = f'./data/{dataset}-training.odgt'
test_odgt = f'./data/{dataset}-validation.odgt'

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
train_dataloader = DataLoader(train_dataset,
                              batch_size=cfg.TRAIN.batch_size_per_gpu,
                              shuffle=True, num_workers=n_cpu)
valid_dataloader = DataLoader(valid_dataset,
                              batch_size=cfg.VAL.batch_size,
                              shuffle=False, num_workers=n_cpu)
test_dataloader = DataLoader(test_dataset, batch_size=cfg.VAL.batch_size,
                             shuffle=False, num_workers=n_cpu)


from model import MaskModel

model = MaskModel(arch=cfg.MODEL.arch, encoder_name=cfg.MODEL.encoder_name,
                  in_channels=3, out_classes=1,
                  optimizer_type=cfg.TRAIN.optim, learning_rate=cfg.TRAIN.lr)
os.makedirs(cfg.DIR, exist_ok=True)
trainer = pl.Trainer(
    gpus=1,
    max_epochs=cfg.TRAIN.num_epochs,
    default_root_dir=cfg.DIR,
)

trainer.fit(
    model,
    train_dataloaders=train_dataloader,
    val_dataloaders=valid_dataloader,
)

# run validation dataset
valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
pprint(valid_metrics)

# run test dataset
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False,
                             num_workers=n_cpu)

test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
pprint(test_metrics)

batch = next(iter(test_dataloader))
with torch.no_grad():
    model.eval()
    logits = model(batch["image"])
pr_masks = logits.sigmoid()

images_folder =os.path.join(cfg.DIR, 'images')
os.makedirs(images_folder, exist_ok=True)

for idx, (image, gt_mask, pr_mask) in enumerate(zip(batch["image"], batch["mask"], pr_masks)):
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
    image_path = os.path.join(images_folder, f"prediction{idx:03d}.png")
    plt.savefig(image_path)
