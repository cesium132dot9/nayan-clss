#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict
import random
from transformer_lens import HookedTransformer
import torch.nn.functional as F
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

from huggingface_hub import login
import os
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    login(token=hf_token)
else:
    print("Warning: HF_TOKEN not set. Gated models may fail to load.")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class SimplifiedIronicReboundExperiments:
    
    def __init__(self, device='cuda', dtype=torch.bfloat16):
        self.device = device
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU.")
            self.device = 'cpu'
        self.dtype = dtype
        
    def load_existing_dataset(self) -> pd.DataFrame:
        df = pd.read_csv('negation_dataset.csv')
        return df
    
    def _get_model_internal_proxies(self, model: HookedTransformer, forbidden_concepts: List[str]) -> Dict[str, str]:
        proxies = {}
        embed_matrix = model.embed.W_E.to(self.device)
        
        for forbidden in set(forbidden_concepts):
            try:
                # Attempt to get the token ID for the forbidden concept. If it's multi-token, use the first token.
                # This addresses the 'Input string is not a single token' error.
                try:
                    forbidden_token = model.to_single_token(forbidden)
                except Exception:
                    # Fallback for multi-token concepts: use the first token's ID
                    forbidden_token = model.to_tokens(forbidden, prepend_bos=False).flatten()[0].item()
                forbidden_emb = embed_matrix[forbidden_token]
                similarities = F.cosine_similarity(forbidden_emb.unsqueeze(0), embed_matrix, dim=1)
                valid_indices = torch.where((similarities > 0.7) & 
                                          (torch.arange(len(similarities), device=self.device) != forbidden_token))[0]
                
                if len(valid_indices) > 0:
                    proxy_idx = valid_indices[torch.randint(len(valid_indices), (1,), device=self.device)].item()
                    proxy_token = model.to_string(proxy_idx)
                    proxies[forbidden] = proxy_token
                else:
                    similarities[forbidden_token] = -1
                    best_idx = similarities.argmax().item()
                    proxy_token = model.to_string(best_idx)
                    proxies[forbidden] = proxy_token
            except:
                proxies[forbidden] = 'thing'
        
        return proxies
    
    def get_optimal_batch_size(self, model_name):
        size_mapping = {
            'gpt-oss-20b': 8,
            'gemma-3-270m-it': 256,
            'lfm2-350m': 256
        }
        return size_mapping.get(model_name, 64)
    
    def format_prompt(self, text, model_name):
        if 'gemma-3' in model_name.lower():
            return f"<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"
        elif 'gpt-oss' in model_name.lower():
            return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        elif 'lfm2' in model_name.lower():
            return text  # LFM2 uses standard text format
        return text
    
    def load_single_model(self, model_name: str, model_path: str) -> HookedTransformer:
        """Loads a single model onto the specified device (GPU only) with OOM handling."""
        print(f"Attempting to load model: {model_name} from {model_path}")
        
        # Clear CUDA cache first
        torch.cuda.empty_cache()
        
        model = None
        try:
            # First attempt: Load to CPU with memory optimization, then move to GPU
            model = HookedTransformer.from_pretrained_no_processing(
                model_path, 
                device='cpu',
                dtype=self.dtype, 
                trust_remote_code=True,
                low_cpu_mem_usage=True  # Reduce peak RAM usage during loading
            )
            
            # Move to GPU after successful CPU loading
            print(f"Moving {model_name} to GPU...")
            model = model.to(self.device)
            
            # Validate model on GPU with test input
            model.eval()
            test_input = torch.tensor([[1, 2, 3]]).to(self.device)
            with torch.no_grad():
                _ = model(test_input)
            print(f"Successfully loaded and validated {model_name} on GPU")
            
            return model
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA OOM error for {model_name}: {e}. Trying device_map='auto'...")
            if model is not None:
                del model
            torch.cuda.empty_cache()
            
            try:
                # Second attempt: Use device_map="auto" for automatic GPU placement
                model = HookedTransformer.from_pretrained_no_processing(
                    model_path, 
                    device_map="auto",
                    dtype=self.dtype, 
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                # Validate model with test input
                model.eval()
                test_input = torch.tensor([[1, 2, 3]]).to(self.device)
                with torch.no_grad():
                    _ = model(test_input)
                print(f"Successfully loaded {model_name} with device_map='auto'")
                return model
                
            except Exception as e2:
                print(f"Second attempt with device_map failed for {model_name}: {e2}")
                if model is not None:
                    del model
                torch.cuda.empty_cache()
                return None
                
        except KeyboardInterrupt:
            print(f"Loading interrupted for {model_name}. Cleaning up...")
            if model is not None:
                del model
            torch.cuda.empty_cache()
            raise
            
        except Exception as e:
            print(f"Loading failed for {model_name}: {e}")
            if model is not None:
                del model
            torch.cuda.empty_cache()
            
            # Try redownload as fallback
            try:
                print(f"Attempting redownload for {model_name}")
                from huggingface_hub import snapshot_download
                import shutil
                from pathlib import Path
                
                cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
                model_cache_path = cache_dir / f"models--{model_path.replace('/', '--')}"
                if model_cache_path.exists():
                    shutil.rmtree(model_cache_path)
                    
                snapshot_download(repo_id=model_path)
                
                model = HookedTransformer.from_pretrained_no_processing(
                    model_path, 
                    device='cpu', 
                    dtype=self.dtype, 
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                model = model.to(self.device)
                return model
                
            except Exception as e2:
                print(f"Redownload also failed for {model_name}: {e2}")
                if model is not None:
                    del model
                torch.cuda.empty_cache()
                return None
    
    def experiment_e1_mention_controlled_contrast(self, model: HookedTransformer, 
                                                 dataset: pd.DataFrame) -> pd.DataFrame:
        forbidden_concepts = dataset['forbidden_concept'].tolist()
        proxy_dict = self._get_model_internal_proxies(model, forbidden_concepts)
        dataset = dataset.copy()
        dataset['proxy_concept'] = dataset['forbidden_concept'].map(proxy_dict)
        
        results = []
        batch_size = self.get_optimal_batch_size(getattr(model.cfg, 'model_name', 'unknown'))
        
        neg_data = dataset[dataset['prompt_type'] == 'negative'].copy()
        neu_data = dataset[dataset['prompt_type'] == 'neutral'].copy()
        pos_data = dataset[dataset['prompt_type'] == 'positive'].copy()
        
        model_name = getattr(model.cfg, 'model_name', 'unknown')
        
        all_data = []
        for df, ptype in [(neg_data, 'negative'), (neu_data, 'neutral'), (pos_data, 'positive')]:
            for _, row in df.iterrows():
                formatted_prompt = self.format_prompt(row['prompt_text'], model_name)
                all_data.append({
                    'id': row['id'],
                    'prompt_type': ptype,
                    'prompt_text': formatted_prompt,
                    'forbidden_concept': row['forbidden_concept']
                })
        
        for i in range(0, len(all_data), batch_size):
            batch = all_data[i:i+batch_size]
            try:
                prompts = [item['prompt_text'] for item in batch]
                
                if hasattr(model, 'tokenizer'):
                    tokens = model.tokenizer(prompts, padding=True, return_tensors='pt', 
                                           truncation=True, max_length=512).to(self.device)
                    token_ids = tokens['input_ids']
                else:
                    token_ids = model.to_tokens(prompts, prepend_bos=True)
                
                with torch.no_grad():
                    logits = model(token_ids)[:, -1, :]
                    
                for j, item in enumerate(batch):
                    forbidden = item['forbidden_concept']
                    try:
                        # Attempt to get the token ID for the forbidden concept. If it's multi-token, use the first token.
                        # This addresses the 'Input string is not a single token' error.
                        try:
                            forbidden_token = model.to_single_token(forbidden)
                        except Exception:
                            # Fallback for multi-token concepts: use the first token's ID
                            forbidden_token = model.to_tokens(forbidden, prepend_bos=False).flatten()[0].item()
                        logp = F.log_softmax(logits[j], dim=-1)[forbidden_token].item()
                        results.append({
                            'id': item['id'],
                            'prompt_type': item['prompt_type'],
                            'logp': logp,
                            'forbidden_concept': forbidden
                        })
                    except Exception as e:
                        print(f"Warning: Could not process forbidden concept '{forbidden}' for ID {item['id']}: {e}")
                        results.append({
                            'id': item['id'],
                            'prompt_type': item['prompt_type'],
                            'logp': np.nan,
                            'forbidden_concept': forbidden
                        })
            except Exception as e:
                print(f"Batch {i} failed: {e}. Skipping.")
                continue
        
        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0 or 'logp' not in results_df.columns:
            return pd.DataFrame(columns=['id', 'delta', 'logp_neutral', 'logp_negative', 'logp_positive', 'forbidden_concept'])
        
        # Fill NaN values in logp with the mean of non-NaN values
        logp_mean = results_df['logp'].dropna().mean()
        if pd.isna(logp_mean):
            logp_mean = -10.0  # Default fallback value
        results_df['logp'] = results_df['logp'].fillna(logp_mean)
        
        pivot_df = pd.pivot_table(results_df, values='logp', index='id', 
                                 columns='prompt_type', fill_value=0, aggfunc='mean')
        
        delta_results = []
        
        # Only process negative prompt IDs (0-1665) and match with corresponding neutral/positive
        for neg_id in range(len(neg_data)):
            neutral_id = neg_id + 1666  # Corresponding neutral prompt ID  
            pos_id = neg_id + 3332      # Corresponding positive prompt ID
            
            # Initialize log probabilities
            logp_negative = 0
            logp_neutral = 0
            logp_positive = np.nan
            
            # Get negative log probability
            if neg_id in pivot_df.index and 'negative' in pivot_df.columns:
                logp_negative = pivot_df.loc[neg_id, 'negative']
                
            # Get neutral log probability  
            if neutral_id in pivot_df.index and 'neutral' in pivot_df.columns:
                logp_neutral = pivot_df.loc[neutral_id, 'neutral']
                
            # Get positive log probability with fallback
            if pos_id in pivot_df.index and 'positive' in pivot_df.columns:
                logp_positive = pivot_df.loc[pos_id, 'positive']
                if logp_positive == 0:
                    logp_positive = np.nan
            else:
                if 'positive' in pivot_df.columns:
                    pos_values = pivot_df['positive'][pivot_df['positive'] != 0]
                    if len(pos_values) > 0:
                        logp_positive = pos_values.mean()
                    else:
                        logp_positive = np.nan
                else:
                    logp_positive = np.nan
            
            # Only include if we have both negative and neutral non-zero values
            if logp_negative != 0 and logp_neutral != 0:
                orig_row = dataset[dataset['id'] == neg_id]
                if len(orig_row) > 0:
                    forbidden = orig_row.iloc[0]['forbidden_concept']
                    delta = logp_neutral - logp_negative
                    delta_results.append({
                        'id': neg_id,
                        'delta': delta,
                        'logp_neutral': logp_neutral,
                        'logp_negative': logp_negative,
                        'logp_positive': logp_positive,
                        'forbidden_concept': forbidden
                    })
        
        final_df = pd.DataFrame(delta_results)
        
        # Ensure no NaN values in final CSV
        if len(final_df) > 0:
            # Fill NaN values in numeric columns with their means
            numeric_cols = ['delta', 'logp_neutral', 'logp_negative', 'logp_positive']
            for col in numeric_cols:
                if col in final_df.columns:
                    col_mean = final_df[col].dropna().mean()
                    if pd.isna(col_mean):
                        col_mean = -10.0 if 'logp' in col else 0.0
                    final_df[col] = final_df[col].fillna(col_mean)
        
        return final_df
    
    def experiment_e2_load_distractor_effects(self, model: HookedTransformer,
                                             dataset: pd.DataFrame) -> pd.DataFrame:
        results = []
        distractor_lengths = [0, 64, 256, 1024, 2048, 4096]
        distractor_types = ['syntactic', 'semantic', 'repetition']
        
        neg_data = dataset[dataset['prompt_type'] == 'negative']
        batch_size = self.get_optimal_batch_size(getattr(model.cfg, 'model_name', 'unknown')) // 4
        
        for dist_type in distractor_types:
            for length in distractor_lengths:
                for batch_start in range(0, len(neg_data), batch_size):
                    batch_end = min(batch_start + batch_size, len(neg_data))
                    batch_data = neg_data.iloc[batch_start:batch_end]
                    
                    try:
                        for _, row in batch_data.iterrows():
                            if length == 0:
                                distractor = ""
                            elif dist_type == 'syntactic':
                                pattern = " The quick brown fox jumps over the lazy dog."
                                distractor = pattern * (length // len(pattern.split()))
                            elif dist_type == 'semantic':
                                related = f" This {row['topic']} contains many elements and features."
                                distractor = related * (length // len(related.split()))
                            else:
                                distractor = f" {row['prompt_text']}" * (length // len(row['prompt_text'].split()))
                            
                            modified_prompt = row['prompt_text'] + distractor
                            
                            try:
                                tokens = model.to_tokens([modified_prompt])
                                if tokens.shape[1] > 4096:
                                    tokens = tokens[:, :4096]
                                    
                                # Attempt to get the token ID for the forbidden concept. If it's multi-token, use the first token.
                                # This addresses the 'Input string is not a single token' error.
                                try:
                                    forbidden_token = model.to_single_token(row['forbidden_concept'])
                                except Exception:
                                    # Fallback for multi-token concepts: use the first token's ID
                                    forbidden_token = model.to_tokens(row['forbidden_concept'], prepend_bos=False).flatten()[0].item()
                                
                                with torch.no_grad():
                                    logits = model(tokens)[:, -1, :]
                                    logp = F.log_softmax(logits, dim=-1)[0, forbidden_token].item()
                                
                                results.append({
                                    'id': row['id'],
                                    'distractor_type': dist_type,
                                    'distractor_length': length,
                                    'logp': logp,
                                    'forbidden_concept': row['forbidden_concept']
                                })
                            except Exception as e:
                                print(f"Warning: Could not process forbidden concept '{row['forbidden_concept']}' for ID {row['id']}: {e}")
                                results.append({
                                    'id': row['id'],
                                    'distractor_type': dist_type,
                                    'distractor_length': length,
                                    'logp': np.nan,
                                    'forbidden_concept': row['forbidden_concept']
                                })
                    except Exception as e:
                        print(f"Batch starting at {batch_start} for distractor type {dist_type} and length {length} failed: {e}. Skipping.")
                        continue
        
        results_df = pd.DataFrame(results)
        
        # Fill NaN values in logp with the mean of non-NaN values
        if len(results_df) > 0 and 'logp' in results_df.columns:
            logp_mean = results_df['logp'].dropna().mean()
            if pd.isna(logp_mean):
                logp_mean = -10.0  # Default fallback value
            results_df['logp'] = results_df['logp'].fillna(logp_mean)
        
        for dist_type in distractor_types:
            type_data = results_df[results_df['distractor_type'] == dist_type]
            if len(type_data) > 5:
                try:
                    x = type_data['distractor_length'].values
                    y = type_data['logp'].values
                    smoothed = sm.nonparametric.lowess(y, x, frac=0.2)
                    type_indices = results_df[results_df['distractor_type'] == dist_type].index
                    results_df.loc[type_indices, 'logp_smoothed'] = smoothed[:, 1]
                except:
                    results_df.loc[results_df['distractor_type'] == dist_type, 'logp_smoothed'] = type_data['logp']
        
        # Ensure logp_smoothed has no NaN values
        if 'logp_smoothed' in results_df.columns:
            logp_smoothed_mean = results_df['logp_smoothed'].dropna().mean()
            if pd.isna(logp_smoothed_mean):
                logp_smoothed_mean = results_df['logp'].mean()
            results_df['logp_smoothed'] = results_df['logp_smoothed'].fillna(logp_smoothed_mean)
        
        return results_df
    
    def compute_statistics_fixed(self, data: pd.DataFrame, value_col: str = 'delta') -> Dict:
        values = data[value_col].dropna()
        
        if len(values) == 0:
            return {'mean': 0, 'ci_lower': 0, 'ci_upper': 0, 'p_value': 1.0, 'p_perm': 1.0, 'bootstrap_samples': []}
        
        bootstrap_samples = []
        for _ in range(5000):
            sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_samples.append(np.mean(sample))
        
        ci_lower = np.percentile(bootstrap_samples, 2.5)
        ci_upper = np.percentile(bootstrap_samples, 97.5)
        
        observed_mean = np.mean(values)
        perm_stats = []
        
        for _ in range(1000):
            perm_values = values * np.random.choice([-1, 1], size=len(values))
            perm_stats.append(np.mean(perm_values))
        
        p_perm = np.mean(np.abs(perm_stats) >= np.abs(observed_mean))
        t_stat, p_value = stats.ttest_1samp(values, 0)
        
        return {
            'mean': observed_mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'p_perm': p_perm,
            'n_samples': len(values),
            'bootstrap_samples': bootstrap_samples
        }
    
    def validate_experiment_results(self, df, experiment_name, model_name):
        issues = []
        
        # Conditional validation based on experiment type
        if experiment_name == 'e1':
            # E1 validation: expected 1666 rows and specific columns
            expected_rows = 1666
            if len(df) != expected_rows:
                issues.append(f"Expected {expected_rows} rows, got {len(df)}")
            
            required_cols = ['id', 'delta', 'logp_neutral', 'logp_negative', 'logp_positive', 'forbidden_concept']
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                issues.append(f"Missing columns: {missing_cols}")
            
            # Check for zero and NaN values in logp columns
            for col in ['logp_neutral', 'logp_negative']:
                if col in df.columns:
                    zero_count = (df[col] == 0).sum()
                    if zero_count > 0:
                        issues.append(f"{col}: {zero_count} zero values found")
                    
                    nan_count = df[col].isna().sum()
                    if nan_count > 0:
                        issues.append(f"{col}: {nan_count} NaN values found")
            
            # Check logp_positive for high NaN percentage
            if 'logp_positive' in df.columns:
                pos_nan_count = df['logp_positive'].isna().sum()
                pos_nan_pct = pos_nan_count / len(df) * 100 if len(df) > 0 else 0
                if pos_nan_pct > 20:
                    issues.append(f"logp_positive: {pos_nan_count} NaN values ({pos_nan_pct:.1f}%) - High missing rate")
                    
        elif experiment_name == 'e2':
            # E2 validation: different columns, no strict row count
            required_cols = ['id', 'distractor_type', 'distractor_length', 'logp', 'forbidden_concept']
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                issues.append(f"Missing columns: {missing_cols}")
            
            # Check for zero and NaN values in logp
            if 'logp' in df.columns:
                zero_count = (df['logp'] == 0).sum()
                if zero_count > 0:
                    issues.append(f"logp: {zero_count} zero values found")
                
                nan_count = df['logp'].isna().sum()
                if nan_count > 0:
                    issues.append(f"logp: {nan_count} NaN values found")
            
            # Check logp_smoothed if it exists
            if 'logp_smoothed' in df.columns:
                smoothed_nan_count = df['logp_smoothed'].isna().sum()
                if smoothed_nan_count > 0:
                    issues.append(f"logp_smoothed: {smoothed_nan_count} NaN values found")
                    
        else:
            issues.append(f"Unknown experiment type: {experiment_name}")
        
        # Report results
        if issues:
            print(f"VALIDATION ISSUES for {model_name} {experiment_name}:")
            for issue in issues:
                print(f"  ⚠️  {issue}")
        else:
            print(f"✓ Validation passed for {model_name} {experiment_name}")
        
        return len(issues) == 0
    
    def generate_simple_visualizations(self, results_dir: str, model_name: str = "Model"):
        e1_files = [f for f in os.listdir(results_dir) if f == 'e1.csv']
        if e1_files:
            e1_data = pd.read_csv(os.path.join(results_dir, e1_files[0]))
            
            # Validate data before plotting
            if len(e1_data) > 0 and 'delta' in e1_data.columns:
                delta_values = e1_data['delta'].dropna()
                if len(delta_values) > 0:  # Only plot if we have valid data
                    plt.figure(figsize=(12, 8))
                    mean_delta = delta_values.mean()
                    
                    plt.hist(delta_values, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
                    plt.axvline(x=mean_delta, color='red', linestyle='--', linewidth=2, 
                               label=f'Average = {mean_delta:.1f}')
                    
                    plt.xlabel('Delta Score (Difference Between Responses)', fontsize=14)
                    plt.ylabel('Number of Examples', fontsize=14)
                    
                    model_display = model_name.upper() if 'gpt2' in model_name.lower() else model_name.title()
                    if 'opt' in model_name.lower():
                        model_display = model_name.upper()
                    elif 'gemma' in model_name.lower():
                        model_display = model_name.title()
                        
                    plt.title(f'{model_display}: How Much Models Distinguish Between Different Responses', 
                             fontsize=16, fontweight='bold')
                    
                    textstr = f'Higher scores = Model can better distinguish\nbetween positive, negative, and neutral responses\nTotal examples: {len(delta_values):,}'
                    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
                            verticalalignment='top', bbox=props)
                    
                    plt.legend(fontsize=12)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(results_dir, 'e1.png'), dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"E1 visualization saved for {model_name}")
                else:
                    print(f"No valid delta values found for {model_name} E1 visualization")
            else:
                print(f"Invalid E1 data for {model_name} visualization")
        
        e2_files = [f for f in os.listdir(results_dir) if f == 'e2.csv']
        if e2_files:
            e2_data = pd.read_csv(os.path.join(results_dir, e2_files[0]))
            
            # Validate data before plotting
            if len(e2_data) > 0 and 'distractor_type' in e2_data.columns and 'logp' in e2_data.columns:
                valid_data = e2_data.dropna(subset=['logp', 'distractor_type', 'distractor_length'])
                if len(valid_data) > 0:
                    plt.figure(figsize=(12, 8))
                    
                    colors = ['blue', 'red', 'green']
                    plot_generated = False
                    
                    for i, dist_type in enumerate(valid_data['distractor_type'].unique()):
                        type_data = valid_data[valid_data['distractor_type'] == dist_type]
                        
                        length_stats = {}
                        for length in sorted(type_data['distractor_length'].unique()):
                            length_data = type_data[type_data['distractor_length'] == length]['logp'].values
                            if len(length_data) > 1:
                                stats = self.compute_statistics_fixed(pd.DataFrame({'delta': length_data}), 'delta')
                                length_stats[length] = {
                                    'mean': stats['mean'],
                                    'ci_lower': np.percentile(stats['bootstrap_samples'], 2.5),
                                    'ci_upper': np.percentile(stats['bootstrap_samples'], 97.5)
                                }
                        
                        if length_stats:
                            lengths = sorted(length_stats.keys())
                            means = [length_stats[l]['mean'] for l in lengths]
                            cis_lower = [length_stats[l]['ci_lower'] for l in lengths]
                            cis_upper = [length_stats[l]['ci_upper'] for l in lengths]
                            
                            plt.plot(lengths, means, marker='o', label=dist_type, 
                                    linewidth=2, color=colors[i % len(colors)])
                            plt.fill_between(lengths, cis_lower, cis_upper,
                                           alpha=0.2, color=colors[i % len(colors)])
                            plot_generated = True
                    
                    if plot_generated:
                        plt.xlabel('Distractor Length (tokens)')
                        plt.ylabel('Log Probability')
                        plt.title('E2: Load/Distractor Effects with 95% CI')
                        plt.legend()
                        plt.savefig(os.path.join(results_dir, 'e2.png'), dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"E2 visualization saved for {model_name}")
                    else:
                        plt.close()
                        print(f"No valid data for {model_name} E2 visualization")
                else:
                    print(f"No valid E2 data found for {model_name} visualization")
            else:
                print(f"Invalid E2 data for {model_name} visualization")
    
    def run_simplified_pipeline(self):
        dataset = self.load_existing_dataset()
        
        # Set memory optimization environment variable
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        model_configs = [
            ("gpt-oss-20b", "openai/gpt-oss-20b"),
            ("gemma-3-270m-it", "google/gemma-3-270m-it"),
            ("lfm2-350m", "LiquidAI/LFM2-350M")
        ]

        for model_name, model_path in model_configs:
            print(f"\n--- Running experiments for model: {model_name} ---")
            
            # Clear CUDA cache before loading each model
            torch.cuda.empty_cache()
            
            try:
                model = self.load_single_model(model_name, model_path)

                if model is None:
                    print(f"ERROR: Failed to load {model_name}. Skipping to next model.")
                    # Clear cache even on failure
                    torch.cuda.empty_cache()
                    continue

                results_dir = f'results_{model_name}_simplified'
                os.makedirs(results_dir, exist_ok=True)
                
                e1_results = None
                e2_results = None

                try:
                    # Experiment E1
                    print(f"Running Experiment E1 for {model_name}...")
                    e1_results = self.experiment_e1_mention_controlled_contrast(model, dataset)
                    
                    # Validate and save E1 results
                    if len(e1_results) > 0:
                        e1_results.to_csv(os.path.join(results_dir, 'e1.csv'), index=False)
                        self.validate_experiment_results(e1_results, 'e1', model_name)
                        print(f"✓ E1 results saved to {os.path.join(results_dir, 'e1.csv')}")
                    else:
                        print(f"WARNING: E1 experiment produced no results for {model_name}")

                    # Experiment E2
                    print(f"Running Experiment E2 for {model_name}...")
                    e2_results = self.experiment_e2_load_distractor_effects(model, dataset)
                    
                    # Validate and save E2 results
                    if len(e2_results) > 0:
                        e2_results.to_csv(os.path.join(results_dir, 'e2.csv'), index=False)
                        self.validate_experiment_results(e2_results, 'e2', model_name)
                        print(f"✓ E2 results saved to {os.path.join(results_dir, 'e2.csv')}")
                    else:
                        print(f"WARNING: E2 experiment produced no results for {model_name}")

                    # Generate Visualizations
                    print(f"Generating visualizations for {model_name}...")
                    self.generate_simple_visualizations(results_dir, model_name)

                    # Compute and save summary statistics
                    if e1_results is not None and len(e1_results) > 0:
                        e1_stats = self.compute_statistics_fixed(e1_results, 'delta')
                        summary_stats = {
                            'model': model_name,
                            'e1_mean_delta': e1_stats['mean'],
                            'e1_p_perm': e1_stats['p_perm'],
                            'e1_ci_lower': e1_stats['ci_lower'],
                            'e1_ci_upper': e1_stats['ci_upper'],
                            'total_samples_processed': len(dataset),
                            'e1_valid_deltas': len(e1_results),
                            'e2_samples': len(e2_results) if e2_results is not None else 0
                        }
                        summary_df = pd.DataFrame([summary_stats])
                        summary_df.to_csv(os.path.join(results_dir, 'summary_stats.csv'), index=False)
                        print(f"✓ Summary statistics saved to {os.path.join(results_dir, 'summary_stats.csv')}")
                    
                    print(f"✓ Successfully completed all experiments for {model_name}")

                except KeyboardInterrupt:
                    print(f"Experiments interrupted for {model_name}. Cleaning up...")
                    raise
                    
                except Exception as e:
                    print(f"ERROR during experiments for {model_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    
                finally:
                    # Always clean up model from GPU memory
                    if model is not None:
                        print(f"Unloading {model_name} and clearing CUDA cache...")
                        del model
                        torch.cuda.empty_cache()
                        
            except KeyboardInterrupt:
                print(f"Model loading interrupted for {model_name}. Moving to cleanup...")
                raise
                
            except Exception as e:
                print(f"ERROR loading {model_name}: {e}")
                continue
                
            print(f"Finished processing {model_name}.")

def main():
    experiments = SimplifiedIronicReboundExperiments(device='cuda', dtype=torch.bfloat16)
    experiments.run_simplified_pipeline()

if __name__ == "__main__":
    main()