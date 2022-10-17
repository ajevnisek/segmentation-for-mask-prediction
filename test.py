import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from pprint import pprint
from torch.utils.data import DataLoader

from model import MaskModel
from dataset import ImageHarmonizationMaskDataset


model_ckpt_path = 'lightning_logs/version_4/checkpoints/epoch=4-step=99.ckpt'
os.makedirs('images', exist_ok=True)
# download data
root = "../data/Image_Harmonization_Dataset/"
dataset = 'Hday2night'
train_odgt = f'./data/{dataset}-training.odgt'
test_odgt = f'./data/{dataset}-validation.odgt'

# init train, val, test sets
train_dataset = ImageHarmonizationMaskDataset(root, train_odgt, "train")
valid_dataset = ImageHarmonizationMaskDataset(root, test_odgt, "valid")
test_dataset = ImageHarmonizationMaskDataset(root, test_odgt, "test")

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
plt.savefig(os.path.join('images', 'train.png'))

sample = valid_dataset[0]
plt.subplot(1,2,1)
plt.imshow(sample["image"].transpose(1, 2, 0)) # for visualization we have to transpose back to HWC
plt.subplot(1,2,2)
plt.imshow(sample["mask"].squeeze())  # for visualization we have to remove 3rd dimension of mask
plt.savefig(os.path.join('images', 'validation.png'))

sample = test_dataset[0]
plt.subplot(1,2,1)
plt.imshow(sample["image"].transpose(1, 2, 0)) # for visualization we have to transpose back to HWC
plt.subplot(1,2,2)
plt.imshow(sample["mask"].squeeze())  # for visualization we have to remove 3rd dimension of mask
plt.savefig(os.path.join('images', 'test.png'))



model = MaskModel("DeepLabV3", "resnet34", in_channels=3, out_classes=1)
model.load_state_dict(torch.load(model_ckpt_path)['state_dict'])
trainer = pl.Trainer(
    gpus=1,
    max_epochs=5,
)
test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
pprint(test_metrics)

batch = next(iter(test_dataloader))
with torch.no_grad():
    model.eval()
    logits = model(batch["image"])
pr_masks = logits.sigmoid()

for idx, (image, gt_mask, pr_mask) in enumerate(zip(batch["image"],
                                                    batch["mask"], pr_masks)):
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

    plt.savefig(os.path.join('images', f'prediction{idx:03d}.png'))

