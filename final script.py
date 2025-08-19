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

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class SimplifiedIronicReboundExperiments:
    
    def __init__(self, device='cuda', dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.models = {}
        
    def load_existing_dataset(self) -> pd.DataFrame:
        df = pd.read_csv('negation_dataset.csv')
        return df
    
    def _get_model_internal_proxies(self, model: HookedTransformer, forbidden_concepts: List[str]) -> Dict[str, str]:
        proxies = {}
        embed_matrix = model.embed.W_E.to(self.device)
        
        for forbidden in set(forbidden_concepts):
            try:
                forbidden_token = model.to_single_token(forbidden)
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
            'gemma-3-270m-it': 256,
            'gpt2-small': 256,
            'opt-2.7b': 128,
            'gemma-7b-it': 64,
            'llama-3-8b-instruct': 32,
            'qwen3-14b': 16,
            'gpt-oss-20b': 8
        }
        return size_mapping.get(model_name, 64)
    
    def format_prompt(self, text, model_name):
        if 'gemma-3' in model_name.lower():
            return f"<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"
        elif 'gemma' in model_name.lower():
            return f"[INST] {text} [/INST]"
        elif 'llama-3' in model_name.lower():
            return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        elif 'qwen3' in model_name.lower():
            return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        elif 'gpt-oss' in model_name.lower():
            return f"### Instruction:\n{text}\n\n### Response:\n"
        return text
    
    def load_models(self) -> Dict[str, HookedTransformer]:
        model_configs = [
            ('gpt2-small', 'gpt2'),
            ('opt-2.7b', 'facebook/opt-2.7b'),
            ('gemma-7b-it', 'google/gemma-7b-it'),
            ('gemma-3-270m-it', 'unsloth/gemma-3-270m-it'),
            ('llama-3-8b-instruct', 'meta-llama/Meta-Llama-3-8B-Instruct'),
            ('qwen3-14b', 'Qwen/Qwen3-14B'),
            ('gpt-oss-20b', 'unsloth/gpt-oss-20b')
        ]
        
        for name, model_path in model_configs:
            try:
                if name in ['gemma-3-270m-it', 'gpt2-small']:
                    model = HookedTransformer.from_pretrained_no_processing(
                        model_path, device=self.device, dtype=self.dtype, trust_remote_code=True)
                elif name in ['opt-2.7b', 'gemma-7b-it']:
                    model = HookedTransformer.from_pretrained_no_processing(
                        model_path, device=self.device, dtype=self.dtype, trust_remote_code=True)
                else:
                    model = HookedTransformer.from_pretrained_no_processing(
                        model_path, device_map='auto', dtype=self.dtype, trust_remote_code=True, low_cpu_mem_usage=True)
                
                model = model.to(self.device)
                self.models[name] = model
            except Exception:
                continue
                
        return self.models
    
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
            prompts = [item['prompt_text'] for item in batch]
            
            try:
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
                        forbidden_token = model.to_single_token(forbidden)
                        logp = F.log_softmax(logits[j], dim=-1)[forbidden_token].item()
                        results.append({
                            'id': item['id'],
                            'prompt_type': item['prompt_type'],
                            'logp': logp,
                            'forbidden_concept': forbidden
                        })
                    except:
                        results.append({
                            'id': item['id'],
                            'prompt_type': item['prompt_type'],
                            'logp': np.nan,
                            'forbidden_concept': forbidden
                        })
            except:
                continue
        
        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0 or 'logp' not in results_df.columns:
            return pd.DataFrame(columns=['id', 'delta', 'logp_neutral', 'logp_negative', 'logp_positive', 'forbidden_concept'])
        
        pivot_df = pd.pivot_table(results_df, values='logp', index='id', 
                                 columns='prompt_type', fill_value=0, aggfunc='mean')
        
        delta_results = []
        for idx, row in pivot_df.iterrows():
            if 'neutral' in row and 'negative' in row:
                orig_row = dataset[dataset['id'] == idx]
                if len(orig_row) > 0:
                    forbidden = orig_row.iloc[0]['forbidden_concept']
                    delta = row['neutral'] - row['negative']
                    delta_results.append({
                        'id': idx,
                        'delta': delta,
                        'logp_neutral': row['neutral'],
                        'logp_negative': row['negative'],
                        'logp_positive': row.get('positive', np.nan),
                        'forbidden_concept': forbidden
                    })
        
        final_df = pd.DataFrame(delta_results)
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
                                
                            forbidden_token = model.to_single_token(row['forbidden_concept'])
                            
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
                        except:
                            continue
        
        results_df = pd.DataFrame(results)
        
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
    
    def generate_simple_visualizations(self, results_dir: str, model_name: str = "Model"):
        e1_files = [f for f in os.listdir(results_dir) if 'e1_results' in f and 'reformatted' not in f]
        if e1_files:
            e1_data = pd.read_csv(os.path.join(results_dir, e1_files[0]))
            
            if len(e1_data) > 0 and 'delta' in e1_data.columns:
                e1_data.to_csv(os.path.join(results_dir, 'e1_results_reformatted.csv'), index=False)
                
                plt.figure(figsize=(12, 8))
                delta_values = e1_data['delta'].dropna()
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
                plt.savefig(os.path.join(results_dir, 'simple_graph.png'), dpi=150, bbox_inches='tight')
                plt.close()
        
        e2_files = [f for f in os.listdir(results_dir) if 'e2_results' in f]
        if e2_files:
            e2_data = pd.read_csv(os.path.join(results_dir, e2_files[0]))
            
            if len(e2_data) > 0:
                plt.figure(figsize=(12, 8))
                
                colors = ['blue', 'red', 'green']
                for i, dist_type in enumerate(e2_data['distractor_type'].unique()):
                    type_data = e2_data[e2_data['distractor_type'] == dist_type]
                    
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
                
                plt.xlabel('Distractor Length (tokens)')
                plt.ylabel('Log Probability')
                plt.title('E2: Load/Distractor Effects with 95% CI')
                plt.legend()
                plt.savefig(os.path.join(results_dir, 'e2_distractor_effects.png'), dpi=150, bbox_inches='tight')
                plt.close()
    
    def run_simplified_pipeline(self):
        dataset = self.load_existing_dataset()
        models = self.load_models()
        
        if not models:
            return
        
        for model_name, model in models.items():
            results_dir = f'results_{model_name}_simplified'
            os.makedirs(results_dir, exist_ok=True)
            
            try:
                e1_results = self.experiment_e1_mention_controlled_contrast(model, dataset)
                e1_results.to_csv(os.path.join(results_dir, 'e1_results.csv'), index=False)
                
                e2_results = self.experiment_e2_load_distractor_effects(model, dataset)
                e2_results.to_csv(os.path.join(results_dir, 'e2_results.csv'), index=False)
                
                self.generate_simple_visualizations(results_dir, model_name)
                
                e1_stats = self.compute_statistics_fixed(e1_results, 'delta')
                summary_stats = {
                    'model': model_name,
                    'e1_mean_delta': e1_stats['mean'],
                    'e1_p_perm': e1_stats['p_perm'],
                    'e1_ci_lower': e1_stats['ci_lower'],
                    'e1_ci_upper': e1_stats['ci_upper'],
                    'total_samples_processed': len(dataset),
                    'e1_valid_deltas': len(e1_results),
                    'e2_samples': len(e2_results)
                }
                
                summary_df = pd.DataFrame([summary_stats])
                summary_df.to_csv(os.path.join(results_dir, 'summary_stats.csv'), index=False)
                
            except Exception:
                continue

def main():
    experiments = SimplifiedIronicReboundExperiments(device='cuda', dtype=torch.bfloat16)
    experiments.run_simplified_pipeline()

if __name__ == "__main__":
    main()