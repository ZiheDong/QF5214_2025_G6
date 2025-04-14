# Name: Yang Tan
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from Data_Loader import load_test_pipeline,Power_Dataset_Pipeline
from collections import defaultdict

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

from Data_Loader import Test, Mysql_Data_Source


class MultiTaskEnergyModel(nn.Module):
    def __init__(self, input_dim=1, lstm_hidden=64, lstm_layers=1,
                 region_vocab_size=1600, region_embed_dim=32,
                 indicator_vocab_size=90, indicator_embed_dim=16):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, lstm_hidden, lstm_layers, batch_first=True)

        self.region_embedding = nn.Embedding(region_vocab_size, region_embed_dim)
        self.indicator_embedding = nn.Embedding(indicator_vocab_size, indicator_embed_dim)

        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden + region_embed_dim + indicator_embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        #self.task_weights = nn.Parameter(torch.ones(indicator_vocab_size))
        self.raw_task_weights = nn.Parameter(torch.ones(indicator_vocab_size))

    @property
    def task_weights(self):
        return F.softplus(self.raw_task_weights) + 1e-3

    def forward(self, x_seq, region_id, indicator_id):
        lstm_out, _ = self.lstm(x_seq)  # [B, T, H]
        lstm_feat = lstm_out[:, -1, :]  # [B, H]

        region_feat = self.region_embedding(region_id)        # [B, region_embed_dim]
        indicator_feat = self.indicator_embedding(indicator_id)  # [B, indicator_embed_dim]

        features = torch.cat([lstm_feat, region_feat, indicator_feat], dim=-1)
        out = self.regressor(features).squeeze(-1)
        return out


### ------------------------------
### 验证函数：计算 MAPE
### ------------------------------
def evaluate_model(model, dataloader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            x = batch['input_seq'].to(device)
            y = batch['target'].to(device)
            region_id = batch['region_id'].to(device)
            indicator_id = batch['indicator_id'].to(device)

            pred = model(x, region_id, indicator_id)
            base_loss = loss_fn(pred, y)  # shape: [B]

            task_weights = model.task_weights[indicator_id]  # shape [B]
            normed_weights = task_weights / task_weights.sum()
            weighted_loss = (base_loss * normed_weights).sum()


            total_loss += weighted_loss.item() * x.size(0)
            total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    return avg_loss


### ------------------------------
### 训练函数（含 Early Stopping）
### ------------------------------
def train_model(model,  optimizer, loss_fn, device, epochs=5, patience=3,dataloader=None):
    start_time = time.time()
    model.to(device)

    val_loader = None
    is_model_save = False

    for epoch in range(epochs):
        model.train()
        best_val_score = float('inf')
        patience_counter = 0
        val_loss_threshold = 0.01
        train_loss_threshold = 0.01
        total_loss = 0
        total_samples = 0
        print("Loading Data...")
        data_source = Mysql_Data_Source()
        pipeline = Power_Dataset_Pipeline(data_source=data_source, batch_size=128, seq_len=30)
        pipeline_iter = iter(pipeline)
        flat_dataloader = chain.from_iterable(pipeline_iter)
        progress_bar = tqdm(flat_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch", dynamic_ncols=True)
        for batch in progress_bar:
            if len(batch) == 0:
                continue
            x = batch['input_seq'].to(device)
            y = batch['target'].to(device)
            region_id = batch['region_id'].to(device)
            indicator_id = batch['indicator_id'].to(device)

            pred = model(x, region_id, indicator_id)
            loss = loss_fn(pred, y)


            task_weights = model.task_weights[indicator_id]  # shape [B]
            normed_weights = task_weights / task_weights.sum()
            weighted_loss = (loss * normed_weights).sum()


            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            total_loss += weighted_loss.item() * x.size(0)
            total_samples += x.size(0)



        avg_loss = total_loss / total_samples
        print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")
        if is_model_save:
            pass
        else:
            print("Back up the best model.")
            torch.save(model.state_dict(), "best_model.pth")
            is_model_save = True

        if val_loader is not None:
            val_loss = evaluate_model(model, val_loader, device, loss_fn)
            print(f"Epoch {epoch + 1} - Val Loss: {val_loss:.4f}")

            if val_loss < best_val_score:
                best_val_score = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "best_model.pth")
                is_model_save = True
                print("✅ New best model saved.")

            # 🎯 提前终止：val_loss 低于绝对阈值
            if val_loss <= val_loss_threshold:
                print(f"🎯 Val loss 已达到阈值 {val_loss_threshold}，提前终止训练")
                torch.save(model.state_dict(), "best_model.pth")
                is_model_save = True
                break

            # ⏳ Early stop by patience（放最后！）
            patience_counter += 1
            print(f"⚠️  No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("🛑 Early stopping triggered (patience limit)")
                torch.save(model.state_dict(), "best_model.pth")
                is_model_save = True
                break

        else:
            if avg_loss <= train_loss_threshold:
                print(f"🛑 Train loss trigger train_loss_threshold: {train_loss_threshold}，提前停止训练")
                torch.save(model.state_dict(), "best_model.pth")
                is_model_save = True
                print("✅ New best model saved.")
                break

            print(f"⚠️  No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                torch.save(model.state_dict(), "best_model.pth")
                print("🛑 Early stopping triggered (patience limit)")
                torch.save(model.state_dict(), "best_model.pth")
                is_model_save = True
                break
            patience_counter += 1






    end_time = time.time()

    print(f"Total training time: {(end_time-start_time)/60} MINS ")

def test_train(data_dir='./data/train_chunks', indicator_vocab_size=90, device=device, epochs=3, patience=3):

    print("Start Training...")
    model = MultiTaskEnergyModel(indicator_vocab_size=indicator_vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ##loss_fn = nn.MSELoss()
    loss_fn = nn.MSELoss(reduction='none')

    train_model(
        model=model,
        #dataloader=pipeline_iter,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        epochs=epochs,
        #val_loader=val_loader,
        patience=patience
    )

    print("\n✅ Training Done，load best-model...")
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    model.eval()
    return model

def selectiive_fine_tune(base_model_path: str,validation_sets: dict,device,loss_fn,fine_tune_epochs: int = 3,
                         fine_tune_threshold: float = 0.025,save_dir: str = "fine_tuned_models"):
    os.makedirs(save_dir, exist_ok=True)
    results = {}

    validation_iter = tqdm(validation_sets.items(), desc="Validation Sets", unit="set", dynamic_ncols=True)

    for val_id, val_loader in validation_iter:
        print(f"Evaluating on Validation Set {val_id} ...")

        model = MultiTaskEnergyModel().to(device)
        model.load_state_dict(torch.load(base_model_path, map_location=device))
        model.eval()

        before_loss = evaluate_test_model(
            model_class=MultiTaskEnergyModel,
            model_path=base_model_path,
            loss_fn=loss_fn,
            dataloader=val_loader,
            device=device,
        )
        print(f"Loss Before Fine-tuning: {before_loss:.6f}")

        if before_loss <= fine_tune_threshold:
            print(f"Loss within threshold ({fine_tune_threshold}), skipping fine-tuning.")
            results[val_id] = {"before": before_loss, "after": before_loss, "improved": False}
            continue

        model = MultiTaskEnergyModel().to(device)
        model.load_state_dict(torch.load(base_model_path, map_location=device))

        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("regressor")

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3
        )

        print("Fine-tuning ...")
        train_model(
            model=model,
            dataloader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epochs=fine_tune_epochs,
            val_loader=None
        )

        after_loss = evaluate_test_model(
            model_class=MultiTaskEnergyModel,
            model_path=None,
            loss_fn=loss_fn,
            dataloader=val_loader,
            device=device,
            model=model
        )
        print(f"Loss After Fine-tuning: {after_loss:.6f}")

        if after_loss < before_loss:
            save_path = os.path.join(save_dir, f"model_finetuned_{val_id}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model improved and saved to {save_path}")
            results[val_id] = {"before": before_loss, "after": after_loss, "improved": True}
        else:
            print("Fine-tuning did not improve performance. Skipping save.")
            results[val_id] = {"before": before_loss, "after": after_loss, "improved": False}

    return results

def evaluate_test_model(model_class, model_path, loss_fn, device=device, batch_size=64):
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))
    model.eval()
    print("Loading Model...")

    data_source = Mysql_Data_Source()
    pipeline = Power_Dataset_Pipeline(data_source=data_source, batch_size=128, seq_len=30)
    pipeline_iter = iter(pipeline)
    print("Loading Data ...")

    total_loss, total_samples = 0.0, 0
    indicator_loss_sum = defaultdict(float)
    indicator_sample_count = defaultdict(int)

    with torch.no_grad():
        for chunk_id, dataloader in enumerate(pipeline_iter):
            progress_bar = tqdm(dataloader, desc=f"🔄 Chunk {chunk_id}", unit="batch", dynamic_ncols=True)

            for batch in progress_bar:
                    x = batch['input_seq'].to(device)
                    y = batch['target'].to(device)
                    region_id = batch['region_id'].to(device)
                    indicator_id = batch['indicator_id'].to(device)

                    pred = model(x, region_id, indicator_id)

                    base_loss = loss_fn(pred, y)  # [B]
                    task_weights = model.task_weights[indicator_id]  # [B]
                    normed_weights = task_weights / task_weights.sum()
                    weighted_loss = (base_loss * normed_weights).sum()

                    total_loss += weighted_loss.item()
                    total_samples += x.size(0)
                    for i in range(len(indicator_id)):
                        gid = int(indicator_id[i].item())
                        indicator_loss_sum[gid] += base_loss[i].item()
                        indicator_sample_count[gid] += 1

    avg_loss = total_loss / total_samples
    print(f"Test Avg Loss: {avg_loss:.6f}")

    print("Per-indicator MSE:")
    for gid in sorted(indicator_loss_sum.keys()):
        mse = indicator_loss_sum[gid] / indicator_sample_count[gid]
        print(f"Indicator {gid:>3}: MSE = {mse:.6f}")
    return avg_loss


if __name__ == '__main__':
    pass

    #test_train()
