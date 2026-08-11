from torch.utils.data import DataLoader
from segmentation.losses import *
from segmentation.datasets.sdd_dataset import *
from segmentation.models.UNetFormer import UNetFormer
from catalyst.contrib.nn import Lookahead
from catalyst import utils

# training hparam
max_epoch = 400
ignore_index = 255
train_batch_size = 2
val_batch_size = 8
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "unetformer"
weights_path = "model_weights/sdd/{}".format(weights_name)
test_weights_name = "unetformer"
log_name = 'sdd/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None # the path for the pretrained model weight
gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None


#  define the network
net = UNetFormer(num_classes=num_classes)
# define the loss
loss = UnetFormerLoss(ignore_index=ignore_index)

use_aux_loss = True

# define the dataloader

train_dataset = SDDDataset(data_root='/home/srikanth/Interns/Gokul/Final_Model/patches/sdd_dataset/train', img_dir='images', mask_dir='masks',
                             mode='train', mosaic_ratio=0.25, transform=train_aug, img_size=(1024, 1024))

val_dataset = SDDDataset(data_root='/home/srikanth/Interns/Gokul/Final_Model/patches/sdd_dataset/val', img_dir='images', mask_dir='masks', mode='val',
                           mosaic_ratio=0.0, transform=val_aug, img_size=(1024, 1024))

test_dataset = SDDDataset(data_root='/home/srikanth/Interns/Gokul/Final_Model/patches/sdd_dataset/test', img_dir='images', mask_dir='masks', mode='val',
                           mosaic_ratio=0.0, transform=val_aug, img_size=(1024, 1024))


print("\n=== Dataset Summary ===")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print("=====================\n")

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)