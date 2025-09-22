# Revise the original TSLANet_classification.py to test the pretrained checkpoint
import argparse
import time
import datetime
import os

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from timm.loss import LabelSmoothingCrossEntropy
from timm.layers import DropPath
from timm.layers import trunc_normal_
from torchmetrics.classification import MulticlassF1Score

from dataloader import get_datasets
from utils import get_clf_report, save_copy_of_files, str2bool, random_masking_3D


class ICB(L.LightningModule):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, 1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x


class PatchEmbed(L.LightningModule):
    def __init__(self, seq_len, patch_size=8, in_chans=3, embed_dim=384):
        super().__init__()
        stride = patch_size // 2
        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        x_out = self.proj(x).flatten(2).transpose(1, 2)
        return x_out


class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)

        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1)) # * 0.5)

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

        # Normalize energy
        epsilon = 1e-6  # Small constant to avoid division by zero
        normalized_energy = energy / (median_energy + epsilon)

        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)

        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if args.adaptive_filter:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high

            x_weighted += x_weighted2

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape

        return x


class TSLANet_layer(L.LightningModule):
    def __init__(self, dim, mlp_ratio=3., drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.asb = Adaptive_Spectral_Block(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.icb = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        # Check if both ASB and ICB are true
        if args.ICB and args.ASB:
            x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        # If only ICB is true
        elif args.ICB:
            x = x + self.drop_path(self.icb(self.norm2(x)))
        # If only ASB is true
        elif args.ASB:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        # If neither is true, just pass x through
        return x


class TSLANet(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed(
            seq_len=args.seq_len, patch_size=args.patch_size,
            in_chans=args.num_channels, embed_dim=args.emb_dim
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, args.emb_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=args.dropout_rate)

        self.input_layer = nn.Linear(args.patch_size, args.emb_dim)

        dpr = [x.item() for x in torch.linspace(0, args.dropout_rate, args.depth)]  # stochastic depth decay rule

        self.tsla_blocks = nn.ModuleList([
            TSLANet_layer(dim=args.emb_dim, drop=args.dropout_rate, drop_path=dpr[i], mlp_ratio=args.mlp_ratio)
            for i in range(args.depth)]
        )

        # Classifier head
        self.head = nn.Linear(args.emb_dim, args.num_classes)

        # init weights
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def pretrain(self, x_in):
        x = self.patch_embed(x_in)
        x = x + self.pos_embed
        x_patched = self.pos_drop(x)

        x_masked, _, self.mask, _ = random_masking_3D(x, mask_ratio=args.masking_ratio)
        self.mask = self.mask.bool()  # mask: [bs x num_patch x n_vars]

        for tsla_blk in self.tsla_blocks:
            x_masked = tsla_blk(x_masked)

        return x_masked, x_patched

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for tsla_blk in self.tsla_blocks:
            x = tsla_blk(x)

        x = x.mean(1)
        x = self.head(x)
        return x


class model_training(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = TSLANet()
        self.f1 = MulticlassF1Score(num_classes=args.num_classes)
        self.criterion = LabelSmoothingCrossEntropy()
        self.start_time = None
        self.duration = 0

    def forward(self, x):
        return self.model(x)

    # def configure_optimizers(self):
    #     optimizer = optim.AdamW(self.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
    #     return optimizer

    def _calculate_loss(self, batch, mode="train"):
        data = batch[0]
        labels = batch[1].to(torch.int64)

        preds = self.model.forward(data)
        loss = self.criterion(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        f1 = self.f1(preds, labels)

        # Logging for both step and epoch
        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # def training_step(self, batch, batch_idx):
    #     loss = self._calculate_loss(batch, mode="train")
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")

    # def on_train_batch_start(self, batch, batch_idx):
    #     self.start_time = time.time()
    
    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     self.duration += time.time() - self.start_time
    
    # def on_validation_epoch_start(self):
    #     self.log("train_time", self.duration, prog_bar=False, logger=True)
    #     self.start_time, self.duration = time.time(), 0
    
    # def on_validation_epoch_end(self):
    #     # log timer information
    #     self.log("val_time", time.time() - self.start_time, prog_bar=False, logger=True)
    #     self.start_time = None


def test_model(model_path):
    trainer = L.Trainer(
        default_root_dir=None,
        accelerator="auto",
        devices=[args.gpu], # original code: devices=1,
        num_sanity_val_steps=0,
        max_epochs=1,
    )

    L.seed_everything(args.seed)  # To be reproducible
    model = model_training.load_from_checkpoint(model_path)

    # Test best model on validation and test set
    test_result = trainer.test(model, dataloaders=test_loader, verbose=True)

    acc_result = {"test": test_result[0]["test_acc"]}
    f1_result = {"test": test_result[0]["test_f1"]}

    return model, acc_result, f1_result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='TEST')
    parser.add_argument('--data_path', type=str, default=r'data/hhar')
    parser.add_argument('--data_type', type=str, default=r'other', choices=['ucr', 'uea', 'other'])
    parser.add_argument('--data_name', type=str, default=r'toy_dataset')
    parser.add_argument('--ckpt_path', type=str, default=r'checkpoint path')
    parser.add_argument('--ckpt_time', type=str, default=r'checkpoint time (HH_MM_SS)')

    # Training parameters:
    parser.add_argument('--gpu', type=int, default=1, help='gpu')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)

    # Model parameters:
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--mlp_ratio', type=float, default=3.)
    parser.add_argument('--masking_ratio', type=float, default=0.4)
    parser.add_argument('--dropout_rate', type=float, default=0.15)
    parser.add_argument('--patch_size', type=int, default=8)

    # TSLANet components:
    parser.add_argument('--load_from_pretrained', type=str2bool, default=True, help='False: without pretraining')
    parser.add_argument('--ICB', type=str2bool, default=True)
    parser.add_argument('--ASB', type=str2bool, default=True)
    parser.add_argument('--adaptive_filter', type=str2bool, default=True)

    args = parser.parse_args()
    DATASET_PATH = args.data_path

    # load from checkpoint
    run_description = f"{os.path.basename(args.data_path)}_dim{args.emb_dim}_depth{args.depth}_mask{args.masking_ratio}_patch{args.patch_size}_mlp{args.mlp_ratio}___"
    run_description += f"ASB_{args.ASB}__AF_{args.adaptive_filter}__ICB_{args.ICB}__preTr_{args.load_from_pretrained}_"
    run_description += f"{args.ckpt_time}"
    print(f"========== {run_description} ===========")

    import glob
    CHECKPOINT_PATH = os.path.join(args.ckpt_path, run_description, run_description)
    print("Checkpoint path:", CHECKPOINT_PATH)
    if len(glob.glob(os.path.join(CHECKPOINT_PATH, "epoch=*.ckpt"))) > 0:
        CHECKPOINT_PATH = glob.glob(os.path.join(CHECKPOINT_PATH, "epoch=*.ckpt"))[0]
    else:
        print("No checkpoint found!")
        exit(0)
    

    # load datasets ...
    _, _, test_loader, class_names, seq_len, num_channels = get_datasets(DATASET_PATH, args)
    print("Dataset loaded ...")

    # Get dataset characteristics ...
    # args.num_classes = len(np.unique(train_loader.dataset.y_data))
    # args.class_names = [str(i) for i in range(args.num_classes)]
    # args.seq_len = train_loader.dataset.x_data.shape[-1]
    # args.num_channels = train_loader.dataset.x_data.shape[1]
    args.num_classes = len(class_names)
    args.class_names = class_names.values.tolist()
    args.seq_len = seq_len
    args.num_channels = num_channels

    model, acc_results, f1_results = test_model(CHECKPOINT_PATH)

    # # append result to a text file...
    # text_save_dir = "textFiles"
    # os.makedirs(text_save_dir, exist_ok=True)
    # f = open(f"{text_save_dir}/{args.model_id}.txt", 'a')
    # f.write(run_description + "  \n")
    # f.write(f"TSLANet_{os.path.basename(args.data_path)}_l_{args.depth}" + "  \n")
    # f.write('acc:{}, mf1:{}'.format(acc_results, f1_results))
    # f.write('\n')
    # f.write('\n')
    # f.close()
