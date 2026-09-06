import numpy as np
import pandas as pd
import time
import torch
from torch import optim

import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.model import CNN_Classifier
from utils.trainer import Trainer
from utils.loss import FocalLoss, imbalance_weight
from utils.dataset import AugmentedWaferDataset, WaferDataset, felexible_resizing, fixed_resizing
from utils.utils import EarlyStopping, set_seed, distribution_info


BASE_SEED = 42
DATA_SEED=1
WMB_PATH = './data/wm811k-wafer-map/with_label/wafer-map-with-label.pkl'
BATCH_SIZE = 128
TEST_BATCH_SIZE = [8, 16, 32, 64, 128, 256, 512, 1024]
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_COUNTS = {0:6970, 1:6970, 2:6970, 4:6970, 5:6970, 6:6970, 7:6970}


# model
EPOCHS = 100
ALPHA = 0.05
GAMMA = 2.5
LR = 1e-4
WEIGHT_DECAY = 1e-3
STOPPING_RULE = 10
BEST_PATH = './checkpoints/seed_experiment/revision/proposed_model/alpha{}/gamma{}/cnn-mult-classification_seed{}.pt'.format(ALPHA, GAMMA, DATA_SEED)


def build_augment_transform():
    return transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomVerticalFlip(p=0.5),
         transforms.RandomRotation((-180, 180)),
        ]
    )



if __name__ == "__main__":
    set_seed(seed=BASE_SEED)

    wmap_df = pd.read_pickle(WMB_PATH)
    wmaps, labels = wmap_df['waferMap'].copy(), wmap_df['failureNum'].copy()
    resized_wmaps = np.array([felexible_resizing(w) for w in tqdm(wmaps, desc="Resizing wafer bin maps")])  # (N, 128, 128)

    
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
    print('Size of Test data: ', test_data.shape)
    augment_transform = build_augment_transform()
    train_dataset = AugmentedWaferDataset(tr_data, tr_labels, target_counts=TARGET_COUNTS, augment_transform=augment_transform)
    val_dataset = WaferDataset(val_data, val_labels)
    test_dataset = WaferDataset(test_data, test_labels)
    distribution = distribution_info("Augmented Train wafer distribution after augmentation", train_dataset.augmented_labels)


    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)


    ## Model
    model = CNN_Classifier(emb_dim=512, h1_dim=256, h2_dim=128, out_dim=9, drop_p=0.1)
    alpha_weight = imbalance_weight(distribution, none_alpha = ALPHA)
    criterion = FocalLoss(alpha = alpha_weight, gamma=GAMMA, reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    early_stopping = EarlyStopping(patience=STOPPING_RULE, verbose=True, path=str(BEST_PATH))
    early_stopping.load_best_model(model)



    for test_batch_size in TEST_BATCH_SIZE:
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=test_batch_size,
            shuffle=False
        )

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

        _, test_metrics = trainer.evaludate(data_loader=test_loader, loader_name=f"test_loader_batch_{test_batch_size}")
        latency = trainer.measure_inference_latency(data_loader=test_loader, repeat=10)
        
        print(f"[Batch Size: {test_batch_size}]")
        print(f"Inference Time: {latency['mean_time']:.3f} ± {latency['std_time']:.3f} sec")
        print(f"Time per Sample: {latency['ms_per_sample']:.3f} ms/sample")
        print(f"Throughput: {latency['throughput']:.3f} samples/sec")
        print(f"[Test] Acc: {test_metrics['accuracy']:.3f}, "
              f"Prec: {test_metrics['precision']:.3f}, "
              f"Rec: {test_metrics['recall']:.3f}, "
              f"F1: {test_metrics['f1']:.3f}")
        print("\n")