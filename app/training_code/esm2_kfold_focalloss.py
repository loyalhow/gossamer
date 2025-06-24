import argparse
import math
import pickle
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F 
from tqdm import tqdm
from omegaconf import OmegaConf
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from transformers import AutoTokenizer, EsmModel
from sklearn.model_selection import KFold

class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):

        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        
        pt = torch.exp(-ce_loss)
        
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            active_loss = focal_loss[targets != self.ignore_index]
            return active_loss.mean()
        elif self.reduction == 'sum':
            active_loss = focal_loss[targets != self.ignore_index]
            return active_loss.sum()
        else:
            return focal_loss

class DisProtDataset(Dataset):
    def __init__(self, dict_data, tokenizer, max_len=1024):
        self.sequences = [d['sequence'] for d in dict_data]
        self.labels = [d['label'] for d in dict_data]
        self.tokenizer = tokenizer
        self.max_len = max_len
        assert len(self.sequences) == len(self.labels)
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        sequence_str = self.sequences[idx]
        label_str = self.labels[idx]
        tokenized_output = self.tokenizer(
            sequence_str, return_tensors='pt', padding='max_length',
            truncation=True, max_length=self.max_len)
        input_ids = tokenized_output.input_ids.squeeze(0)
        attention_mask = tokenized_output.attention_mask.squeeze(0)
        label_list = [int(c) for c in label_str]
        aligned_labels = torch.full_like(input_ids, -100)
        true_seq_len = attention_mask.sum().item() - 2
        len_to_copy = min(len(label_list), true_seq_len)
        if len_to_copy > 0:
            aligned_labels[1:1 + len_to_copy] = torch.tensor(label_list[:len_to_copy], dtype=torch.long)
        return input_ids, attention_mask, aligned_labels

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=4096):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class DisProtModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.d_model = model_config.d_model
        self.n_head = model_config.n_head
        self.n_layer = model_config.n_layer
        self.embedding_projection = nn.Linear(model_config.esm_embedding_dim, self.d_model)
        self.position_embed = PositionalEncoding(self.d_model, max_len=model_config.max_len + 2)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.dropout_in = nn.Dropout(p=0.1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.n_head, activation='gelu', batch_first=True,
            dim_feedforward=self.d_model * 4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layer)
        self.output_layer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.d_model, model_config.o_dim))
    def forward(self, esm_embeddings):
        x = self.embedding_projection(esm_embeddings)
        x = self.position_embed(x)
        x = self.input_norm(x)
        x = self.dropout_in(x)
        x = self.transformer(x)
        x = self.output_layer(x)
        return x

def metric_fn(pred, gt):
    active_indices = gt.view(-1) != -100
    if not active_indices.any():
        return 0.0
    active_logits = pred.view(-1, pred.shape[-1])[active_indices]
    active_labels = gt.view(-1)[active_indices]
    pred_labels = torch.argmax(active_logits, dim=-1)
    score = f1_score(
        y_true=active_labels.cpu(), 
        y_pred=pred_labels.cpu(), 
        average='micro')
    return score

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser('IDRs prediction - K-Fold with Focal Loss')
    parser.add_argument('--config_path', default='./config.yaml')
    parser.add_argument('--output_dir', default='./outputs_focal_loss', help='Directory to save model checkpoints')
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading ESM model: {config.model.esm_model_name}")
    esm_tokenizer = AutoTokenizer.from_pretrained(config.model.esm_model_name)
    esm_model = EsmModel.from_pretrained(config.model.esm_model_name).to(device)
    esm_model.eval()
    for param in esm_model.parameters():
        param.requires_grad = False

    print("Loading full dataset...")
    with open(config.data.data_path, 'rb') as f:
        full_data_dicts = pickle.load(f)
    full_dataset = DisProtDataset(full_data_dicts, esm_tokenizer, max_len=config.model.max_len)
    print(f"Full dataset loaded with {len(full_dataset)} samples.")

    kfold = KFold(n_splits=config.train.n_splits, shuffle=True, random_state=config.train.seed)
    fold_best_f1_scores = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
        print(f"\n{'='*25} FOLD {fold+1}/{config.train.n_splits} {'='*25}")

        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)
        train_dataloader = DataLoader(full_dataset, sampler=train_subsampler, **config.train.dataloader)
        valid_dataloader = DataLoader(full_dataset, sampler=val_subsampler, batch_size=config.train.dataloader.batch_size)

        print("Initializing model and optimizer for this fold...")
        model = DisProtModel(config.model).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.optimizer.lr, weight_decay=config.train.optimizer.weight_decay)
        
        print(f"Using Focal Loss with gamma={config.train.focal_loss.gamma}, alpha={config.train.focal_loss.alpha}")
        loss_fn = FocalLoss(
            gamma=config.train.focal_loss.gamma,
            alpha=config.train.focal_loss.alpha
        )

        best_f1_in_fold = 0.0
        best_model_fold_path = os.path.join(args.output_dir, f"best_model_fold_{fold+1}.pth")
        patience = config.train.patience
        epochs_no_improve = 0
        
        for epoch in range(config.train.epochs):
            model.train()
            total_loss = 0.
            progress_bar = tqdm(train_dataloader, desc=f"Fold {fold+1} Epoch {epoch:03d}", leave=False)
            for input_ids, attention_mask, labels in progress_bar:
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                with torch.no_grad():
                    embeddings = esm_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                pred = model(embeddings)
                loss = loss_fn(pred.permute(0, 2, 1), labels)
                progress_bar.set_postfix(loss=loss.item())
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            avg_loss = total_loss / len(train_dataloader)

            model.eval()
            total_f1 = 0.
            with torch.no_grad():
                for input_ids, attention_mask, labels in valid_dataloader:
                    input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                    embeddings = esm_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                    pred = model(embeddings)
                    total_f1 += metric_fn(pred, labels)
            avg_f1 = total_f1 / len(valid_dataloader)
            
            print(f"Fold {fold+1} Epoch {epoch:03d} | Avg Loss: {avg_loss:.4f} | Val F1 (micro): {avg_f1:.4f}")

            if avg_f1 > best_f1_in_fold:
                best_f1_in_fold = avg_f1
                epochs_no_improve = 0
                print(f"** New best F1 in Fold {fold+1}: {best_f1_in_fold:.4f}. Saving model to {best_model_fold_path} **")
                torch.save(model.state_dict(), best_model_fold_path)
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered for Fold {fold+1}.")
                break
        
        print(f"--- Fold {fold+1} finished. Best F1 score in this fold: {best_f1_in_fold:.4f} ---")
        fold_best_f1_scores.append(best_f1_in_fold)

    mean_f1 = np.mean(fold_best_f1_scores)
    std_f1 = np.std(fold_best_f1_scores)
    print(f"\n{'='*20} K-Fold Cross-Validation Summary {'='*20}")
    print(f"F1 (micro) scores for each fold: {[round(f, 4) for f in fold_best_f1_scores]}")
    print(f"Average F1 Score: {mean_f1:.4f}")
    print(f"Standard Deviation: {std_f1:.4f}")
