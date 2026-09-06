from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import time
import numpy as np

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 epochs,
                 criterion, optimizer, early_stopping, device):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.epochs = epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.early_stopping = early_stopping
    

    def training(self):

        start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        for epoch in range(self.epochs):
            train_loss, train_metrics = self.train_one_epoch()
            val_loss, val_metrics = self.evaludate(data_loader=self.val_loader, loader_name = "Validation Evaluating...")
            
            print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, " f"Acc: {train_metrics['accuracy']:.4f}, " f"Prec: {train_metrics['precision']:.4f}, " f"Rec: {train_metrics['recall']:.4f}, "f"F1: {train_metrics['f1']:.4f}")
            print(f"[Epoch {epoch+1}] Val Loss: {val_loss:.4f}, " f"Acc: {val_metrics['accuracy']:.4f}, "f"Prec: {val_metrics['precision']:.4f}, " f"Rec: {val_metrics['recall']:.4f}, "f"F1: {val_metrics['f1']:.4f}")

            # Early stopping uses validation F1
            self.early_stopping(val_metrics['f1'], self.model)

            if self.early_stopping.early_stop:
                print("Early stopping triggered...")
                break
        

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end = time.perf_counter()

        # load best model
        self.early_stopping.load_best_model(self.model)
        print("Loaded best model.")


        _, best_train_metrics = self.evaludate(data_loader=self.train_loader, loader_name="Training Evaluating...")
        print(f"[Train] Acc: {best_train_metrics['accuracy']:.4f}, "f"Prec: {best_train_metrics['precision']:.4f}, "f"Rec: {best_train_metrics['recall']:.4f}, "f"F1: {best_train_metrics['f1']:.4f}")
        print("[Best Model Train] Classification Report:\n", best_train_metrics["classification_report"])
        print("[Best Model Train] Confusion Matrix:\n", best_train_metrics["confusion_matrix"])
        print('\n')

        _, best_val_metrics = self.evaludate(data_loader=self.val_loader, loader_name = "Validation Evaluating...")
        print(f"[Validation] Acc: {best_val_metrics['accuracy']:.4f}, "f"Prec: {best_val_metrics['precision']:.4f}, "f"Rec: {best_val_metrics['recall']:.4f}, "f"F1: {best_val_metrics['f1']:.4f}")
        print("[Best Model Validation] Classification Report:\n", best_val_metrics["classification_report"])
        print("[Best Model Validation] Confusion Matrix:\n", best_val_metrics["confusion_matrix"])
        print('\n')

        _, test_metrics = self.evaludate(data_loader=self.test_loader, loader_name="Test Evaluating...")
        print(f"[Test] Acc: {test_metrics['accuracy']:.4f}, "f"Prec: {test_metrics['precision']:.4f}, "f"Rec: {test_metrics['recall']:.4f}, "f"F1: {test_metrics['f1']:.4f}")
        print("Classification Report: \n", test_metrics["classification_report"])
        print("Confusion Matrix:\n", test_metrics["confusion_matrix"])
        print('\n')

        total_min = (end - start) / 60
        print(f"Total training time (min): {total_min:.2f} (epochs_run={epoch})")

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        all_labels = []
        all_preds = []

        for wmap, labels in tqdm(self.train_loader, desc="Training", leave=False):
            x = wmap.to(self.device)
            labels = labels.to(self.device).long()

            outputs = self.model(x)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.train_loader)
        metrics = self.compute_metrics(all_labels, all_preds)
        return avg_loss, metrics


    def evaludate(self, data_loader, loader_name):
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for wmap, labels in tqdm(data_loader, desc=loader_name, leave=False):
                x = wmap.to(self.device)
                labels = labels.to(self.device).long()

                outputs = self.model(x)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        metrics = self.compute_metrics(all_labels, all_preds)

        return avg_loss, metrics


    def compute_metrics(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        conf_matrix = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, digits=4, zero_division=0)
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "classification_report": report, "confusion_matrix": conf_matrix}
    
    @staticmethod
    @torch.no_grad()
    def extract_cnn_features(model, data_loader, device):
        model.eval()

        feature_list = []
        label_list = []

        for x, y in tqdm(data_loader, desc="Extracting CNN features", leave=False):
            x = x.to(device)

            features = model.extract_features(x)

            feature_list.append(features.cpu().numpy())
            label_list.append(y.cpu().numpy())

        features = np.concatenate(feature_list, axis=0)
        labels = np.concatenate(label_list, axis=0)

        return features, labels


    def measure_inference_latency(self, data_loader, repeat=10):
        self.model.eval()
        times = []

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Measurement
        for _ in tqdm(range(repeat), leave=False):
            # start time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            with torch.inference_mode():
                for wmap, labels in data_loader:
                    x = wmap.to(self.device)
                    _ = self.model(x)
            
            # end time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        mean_time = sum(times) / len(times)
        std_time = torch.tensor(times).std().item() if len(times) > 1 else 0.0

        total_samples = len(data_loader.dataset)
        ms_per_sample = (mean_time / total_samples) * 1000
        throughput = total_samples / mean_time

        return {"mean_time": mean_time, "std_time": std_time, "ms_per_sample": ms_per_sample, "throughput": throughput}