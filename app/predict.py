import argparse
import math
import pickle
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from omegaconf import OmegaConf
from transformers import AutoTokenizer, EsmModel

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
            dim_feedforward=self.d_model * 4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layer)
        self.output_layer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.d_model, model_config.o_dim)
        )
    def forward(self, esm_embeddings):
        x = self.embedding_projection(esm_embeddings)
        x = self.position_embed(x)
        x = self.input_norm(x)
        x = self.dropout_in(x)
        x = self.transformer(x)
        x = self.output_layer(x)
        return x

# ------------------------------------------------------------------------------------

def predict_and_save_submission(config_path, model_dir, input_path, output_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    config = OmegaConf.load(config_path)
    
    esm_model_local_path = "./esm2_model_local"
    print(f"Loading feature extractor from local path: {esm_model_local_path}")
    esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_local_path)
    esm_model = EsmModel.from_pretrained(esm_model_local_path).to(device)
    esm_model.eval()

    models = []
    num_folds = 5 
    print(f"Loading {num_folds} trained models for ensembling from '{model_dir}'...")
    
    for fold in range(1, num_folds + 1):
        model_path = os.path.join(model_dir, f"best_model_fold_{fold}.pth")
        if not os.path.exists(model_path):
            print(f"!!! WARNING: Model weight not found at {model_path}. Skipping this model. !!!")
            continue
        
        model = DisProtModel(config.model).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)
    
    if not models:
        raise RuntimeError(f"Fatal: No models were loaded from '{model_dir}'. Please check the path and file names.")
    print(f"Successfully loaded {len(models)} models.")

    print(f"Loading test data from {input_path}...")
    with open(input_path, 'rb') as f:
        test_data_dicts = pickle.load(f)

    results = []
    print(f"Starting prediction for {len(test_data_dicts)} samples...")
    for item in tqdm(test_data_dicts, desc="Predicting"):
        protein_id = item.get('id', 'unknown_id')
        sequence = item.get('sequence', '')
        if not sequence:
            print(f"Warning: Found sample with empty sequence for ID {protein_id}. Skipping.")
            continue

        tokenized_output = esm_tokenizer(
            sequence, return_tensors='pt', padding='max_length',
            truncation=True, max_length=config.model.max_len
        )
        input_ids = tokenized_output.input_ids.to(device)
        attention_mask = tokenized_output.attention_mask.to(device)

        all_logits = []
        with torch.no_grad():
            esm_embeddings = esm_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            for model in models:
                logits = model(esm_embeddings)
                all_logits.append(logits)
            
            stacked_logits = torch.stack(all_logits, dim=0)
            avg_logits = torch.mean(stacked_logits, dim=0)
            final_predictions = torch.argmax(avg_logits, dim=-1).squeeze(0)

        true_seq_len = len(sequence)
        len_to_show = min(true_seq_len, config.model.max_len - 2)
        final_labels_array = final_predictions[1 : 1 + len_to_show].cpu().numpy()
        prediction_str = ''.join(map(str, final_labels_array))
        
        results.append({
            "proteinID": protein_id,
            "sequence": sequence,
            "LIPs": prediction_str
        })

    print("Generating submission file...")
    submission_df = pd.DataFrame(results)
    submission_df = submission_df[["proteinID", "sequence", "LIPs"]]
    submission_df.to_csv(output_path, index=False)
    print(f"Submission file saved successfully to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensemble Prediction for Protein Functional Regions')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the config.yaml file')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing the k-fold model weight files')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input test data file (.pkl)')
    parser.add_argument('--output_path', type=str, required=True, help='Path to write the output submission.csv')
    
    args = parser.parse_args()

    try:
        predict_and_save_submission(
            config_path=args.config_path,
            model_dir=args.model_dir,
            input_path=args.input_path,
            output_path=args.output_path
        )
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        pd.DataFrame(columns=["proteinID", "sequence", "LIPs"]).to_csv(args.output_path, index=False)
        print(f"An empty submission file has been created at {args.output_path} due to an error.")
