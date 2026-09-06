from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from model.TasiCNN import DMC1, WaferAutoEncoder
from utils.trainer import Trainer
from utils.dataset import fixed_resizing
from utils.utils import EarlyStopping


BASE_SEED = 42
DATA_SEED=4 ##
WMB_PATH = './data/wm811k-wafer-map/with_label/wafer-map-with-label.pkl'
BATCH_SIZE = 128
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_COUNTS = {0:6970, 1:6970, 2:6970, 4:6970, 5:6970, 6:6970, 7:6970}


# model
AE_EPOCH = 50
EPOCHS = 200
LR = 1e-4
WEIGHT_DECAY = 1e-3
STOPPING_RULE = 10

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class WaferAEDataset(Dataset):
    def __init__(self, wmaps):
        """
        wmaps: numpy array or list, shape [N, H, W], values in {0, 1, 2}
        """
        self.wmaps = np.asarray(wmaps)

    def __len__(self):
        return len(self.wmaps)

    def __getitem__(self, idx):
        img = self.wmaps[idx]  # [H, W]

        img = torch.tensor(img, dtype=torch.long)
        x = F.one_hot(img, num_classes=3).permute(2, 0, 1).float() # one-hot: [H, W, 3] -> [3, H, W]
        return x

class WafertDataset(Dataset):
    def __init__(self, wmaps, labels):
        self.wmaps = np.asarray(wmaps)
        self.labels = np.asarray(labels)

    def __len__(self):
        return len(self.wmaps)

    def __getitem__(self, idx):
        img = torch.tensor(self.wmaps[idx], dtype=torch.long)
        x = F.one_hot(img, num_classes=3).permute(2, 0, 1).float()
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def train_autoencoder(ae_model, train_wmaps, device, batch_size=128, epochs=1, lr=1e-3, weight_decay=1e-5):
    ae_dataset = WaferAEDataset(train_wmaps)
    ae_loader = DataLoader(ae_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    ae_model = ae_model.to(device)
    ae_model.train()

    optimizer = optim.Adam(ae_model.parameters(),lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        pbar = tqdm(ae_loader, desc=f"[AE Epoch {epoch}/{epochs}]")

        for x in pbar:
            x = x.to(device)  # [B, 3, H, W]

            # target: [B, H, W], values in {0,1,2}
            target = torch.argmax(x, dim=1).long()
            logits, _ = ae_model(x)
            loss = criterion(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(ae_dataset)
        print(f"[AE Epoch {epoch}] Reconstruction Loss: {avg_loss:.6f}")

    return ae_model

def rotate_wmap_90(wmap, k):
    return np.rot90(wmap, k=k).copy()

def build_augmentation(ae_model, tr_data, tr_labels, target_counts, device, batch_size=128, noise_std=0.10, rotate_ks=(1, 2, 3)):

    tr_data = np.asarray(tr_data)
    tr_labels = np.asarray(tr_labels)

    new_data = []
    new_labels = []

    for cls in sorted(np.unique(tr_labels)):
        cls_wmaps = tr_data[tr_labels == cls]
        cls_count = len(cls_wmaps)


        if cls not in target_counts:
            final_wmaps = cls_wmaps

        else:
            target_count = target_counts[cls]

            if cls_count >= target_count:
                idx = np.random.choice(cls_count, size=target_count, replace=False)
                final_wmaps = cls_wmaps[idx]

            else:
                need_count = target_count - cls_count
                candidate_aug = []

                # 1. Rotation of original samples

                for k in rotate_ks:
                    rotated = np.asarray([rotate_wmap_90(wm, k=k) for wm in cls_wmaps])
                    candidate_aug.append(rotated)

                # 2. Autoencoder-generated samples
                ae_generated = generate_ae_samples(ae_model=ae_model, class_wmaps=cls_wmaps, num_generate=need_count, device=device, batch_size=batch_size, noise_std=noise_std)
                candidate_aug.append(ae_generated)

                
                # 3. Rotation of AE-generated samples
                for k in rotate_ks:
                    rotated_ae = np.asarray([rotate_wmap_90(wm, k=k) for wm in ae_generated])
                    candidate_aug.append(rotated_ae)

                # Candidate pool
                candidate_aug = np.concatenate(candidate_aug, axis=0)

                # Shuffle candidate pool and select only needed samples
                idx = np.random.permutation(len(candidate_aug))
                selected_aug = candidate_aug[idx[:need_count]]
                final_wmaps = np.concatenate([cls_wmaps, selected_aug], axis=0)

        final_labels = np.full(len(final_wmaps), cls)
        new_data.append(final_wmaps)
        new_labels.append(final_labels)

        print(f"Class {cls}: original={cls_count}, "f"final={len(final_wmaps)}, "f"target={target_counts.get(cls, cls_count)}")

    new_data = np.concatenate(new_data, axis=0)
    new_labels = np.concatenate(new_labels, axis=0)

    # Final shuffle
    idx = np.random.permutation(len(new_data))
    new_data = new_data[idx]
    new_labels = new_labels[idx]
    return new_data, new_labels

@torch.no_grad()
def generate_ae_samples(ae_model, class_wmaps, num_generate, device, batch_size=128, noise_std=0.10):
    class_wmaps = np.asarray(class_wmaps)

    dataset = WaferAEDataset(class_wmaps)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    ae_model.eval()
    ae_model.to(device)

    generated = []

    while len(generated) < num_generate:
        for x in loader:
            x = x.to(device)

            gen_map, _ = ae_model.generate(x, noise_std=noise_std)
            gen_map = gen_map.cpu().numpy()

            for wm in gen_map:
                generated.append(wm)
                if len(generated) >= num_generate:
                    break

            if len(generated) >= num_generate:
                break
    return np.asarray(generated[:num_generate])



if __name__ == "__main__":
    set_seed(seed=BASE_SEED)

    wmap_df = pd.read_pickle(WMB_PATH)
    wmaps, labels = wmap_df['waferMap'].copy(), wmap_df['failureNum'].copy()
    resized_wmaps = np.array([fixed_resizing(w, size=(64, 64)) for w in tqdm(wmaps, desc="Resizing wafer bin maps")])  # (N, 64, 64)

    
    tr_val_data, test_data, tr_val_labels, test_labels = train_test_split(
        resized_wmaps,
        labels,
        test_size=0.1,
        random_state=DATA_SEED,
        stratify=labels,
    )
    tr_data, val_data, tr_labels, val_labels = train_test_split(
        tr_val_data,
        tr_val_labels,
        test_size=0.2,
        random_state=DATA_SEED,
        stratify=tr_val_labels,
    )

    ae_model = WaferAutoEncoder(latent_channels=64)
    ae_model = train_autoencoder(ae_model=ae_model, train_wmaps=tr_data, device=DEVICE, batch_size=BATCH_SIZE, epochs=AE_EPOCH, lr=1e-3, weight_decay=1e-5)
    
    ## Types of Augmented Samples 
    # 1. rotated original samples
    # 2. autoencoder-generated samples
    # 3. rotated autoencoder-generated samples
    
    ae_rot_aug_data, ae_rot_aug_labels = build_augmentation(ae_model=ae_model, tr_data=tr_data, tr_labels=tr_labels,
        target_counts=TARGET_COUNTS, device=DEVICE, batch_size=BATCH_SIZE,
        noise_std=0.10, rotate_ks=(1, 2, 3))

    train_dataset = WafertDataset(ae_rot_aug_data, ae_rot_aug_labels)
    val_dataset = WafertDataset(val_data, val_labels)
    test_dataset = WafertDataset(test_data, test_labels)
    #distribution = distribution_info("Augmented Train wafer distribution after augmentation", train_dataset.augmented_labels)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    ## Model
    model = DMC1(emb_dim=128)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    checkpoint_path = Path("./checkpoints/seed_experiment/revision/Tasi/v2/cnn-mult-classification_seed{}.pt".format(DATA_SEED))
    early_stopping = EarlyStopping(patience=STOPPING_RULE, verbose=True, path=str(checkpoint_path))


    ## Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=EPOCHS,
        criterion=criterion,
        optimizer=optimizer,
        early_stopping=early_stopping,
        device=DEVICE 
    )
    trainer.training()
    print('Tasi_CNN-data_seed: {}'.format(DATA_SEED))
