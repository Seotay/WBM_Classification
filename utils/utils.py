import random
import numpy as np
import torch
from collections import Counter


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def distribution_info(name, multi_labels):
    counter = Counter(multi_labels)
    total = sum(counter.values())

    print(f"{name} label distribution:")

    distribution = {}

    for label, count in sorted(counter.items()):
        ratio = count / total
        print(f"  Label {label}: {count} ({ratio:.2%})")

        distribution[label] = {
            "count": count,
            "ratio": ratio
        }

    print()
    return distribution


# Early stopping
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        
        self.counter = 0
        self.best_val_f1 = None
        self.early_stop = False

    def __call__(self, current_val_f1, model):
        if self.best_val_f1 is None:
            self.best_val_f1 = current_val_f1
            self.save_checkpoint(current_val_f1, model)
        
        elif current_val_f1 >= self.best_val_f1 - self.delta:
            self.save_checkpoint(current_val_f1, model)
            self.counter = 0
            self.best_val_f1 = current_val_f1
        else:
            self.counter += 1
            if self.verbose:
                print(f"\tEarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, current_val_f1, model):
        if self.verbose:
            print(f"\tValidation F1-score increased ({self.best_val_f1:.4f} → {current_val_f1:.4f}). Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.best_val_f1 = current_val_f1

    def load_best_model(self, model):
        model.load_state_dict(torch.load(self.path, weights_only=True))