#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import random
from datasets import load_dataset
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
        self.results = {}
        
    def generate_dataset(self, size: int = 5000) -> pd.DataFrame:
        print("Generating simplified 5k-sample dataset...")
        
        seed_pairs = []
        try:
            seed_data = load_dataset('tum-nlp/cannot-dataset', split='train[:1000]')
            print(f"Loaded {len(seed_data)} seed samples")
            
            for item in seed_data:
                if isinstance(item, dict):
                    forbidden = item.get('forbidden', item.get('target', item.get('concept', 'thing')))
                    topic = item.get('topic', item.get('context', item.get('domain', 'place')))
                    seed_pairs.append((topic, forbidden))
                    
        except Exception as e:
            print(f"Failed to load seed dataset: {e}")
            topics_fallback = ['farm', 'ocean', 'city', 'forest', 'mountain', 'garden']
            forbidden_fallback = ['cat', 'dog', 'tree', 'car', 'book', 'water', 'fire', 'bird', 'house', 'food']
            for t in topics_fallback:
                for f in forbidden_fallback:
                    seed_pairs.append((t, f))
                    if len(seed_pairs) >= 1000:
                        break
                if len(seed_pairs) >= 1000:
                    break
        
        print(f"Extracted {len(seed_pairs)} seed pairs")
        
        topics = [
            'farm', 'ocean', 'forest', 'city', 'mountain', 'desert', 'garden', 
            'kitchen', 'library', 'hospital', 'school', 'park', 'beach', 'house',
            'restaurant', 'museum', 'zoo', 'theater', 'market', 'workshop',
            'space', 'technology', 'laboratory', 'factory', 'office', 'stadium',
            'airport', 'station', 'mall', 'church', 'castle', 'cave', 'valley',
            'island', 'river', 'lake', 'pond', 'field', 'meadow', 'village',
            'town', 'suburb', 'countryside', 'jungle', 'savanna', 'tundra',
            'arctic', 'tropical', 'urban', 'rural'
        ]
        
        templates = [
            "Describe a {topic} but do not mention {forbidden}.",
            "Write about a {topic} without referring to {forbidden}.", 
            "Explain a {topic} but avoid discussing {forbidden}.",
            "Talk about a {topic} while not mentioning {forbidden}.",
            "Describe a {topic} excluding any reference to {forbidden}."
        ]
        
        temp_model = HookedTransformer.from_pretrained_no_processing('gpt2', device='cpu')
        
        data = []
        samples_per_type = size // 3
        
        while len(seed_pairs) < size:
            topic = random.choice(topics)
            forbidden = random.choice(['cat', 'dog', 'tree', 'car', 'book', 'water', 'fire'])
            seed_pairs.append((topic, forbidden))
        
        for i in range(size):
            topic, forbidden = seed_pairs[i % len(seed_pairs)]
            
            is_single_token = True
            try:
                temp_model.to_single_token(forbidden)
            except:
                is_single_token = False
                forbidden = random.choice(['cat', 'dog', 'tree', 'car', 'book'])
                
            if i < samples_per_type:
                prompt_type = 'negative'
                template = random.choice(templates)
                prompt_text = template.format(topic=topic, forbidden=forbidden)
            elif i < 2 * samples_per_type:
                prompt_type = 'neutral'
                prompt_text = f"Describe a {topic}."
            else:
                prompt_type = 'positive'
                prompt_text = f"Describe a {topic} and mention {forbidden}."
            
            proxy = forbidden + '_proxy'
            
            data.append({
                'id': i,
                'prompt_type': prompt_type,
                'topic': topic,
                'forbidden_concept': forbidden,
                'proxy_concept': proxy,
                'prompt_text': prompt_text,
                'is_single_token': is_single_token
            })
        
        df = pd.DataFrame(data)
        
        type_counts = df['prompt_type'].value_counts()
        print(f"Exact balance: negative={type_counts.get('negative', 0)}, "
              f"neutral={type_counts.get('neutral', 0)}, positive={type_counts.get('positive', 0)}")
        
        df.to_csv('negation_dataset_5k.csv', index=False)
        print(f"Generated simplified dataset with {len(df)} samples")
        return df
    
    def _get_model_internal_proxies(self, model: HookedTransformer, forbidden_concepts: List[str]) -> Dict[str, str]:
        print("Computing model-internal proxies...")
        
        proxies = {}
        
        embed_matrix = model.embed.W_E.to(self.device)
        
        for forbidden in set(forbidden_concepts):
            try:
                forbidden_token = model.to_single_token(forbidden)
                forbidden_emb = embed_matrix[forbidden_token]
                
                similarities = F.cosine_similarity(
                    forbidden_emb.unsqueeze(0), 
                    embed_matrix, 
                    dim=1
                )
                
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
                    
            except Exception as e:
                print(f"Failed to compute proxy for {forbidden}: {e}")
                proxies[forbidden] = 'thing'
        
        return proxies
    
    def load_models(self) -> Dict[str, HookedTransformer]:
        print("Loading models with simplified configuration...")
        
        model_configs = [
            ('gpt2-small', 'gpt2'),
            ('opt-2.7b', 'facebook/opt-2.7b'), 
            ('gemma-7b-it', 'google/gemma-7b-it')
        ]
        
        for name, model_path in model_configs:
            try:
                print(f"Loading {name}...")
                
                if 'gemma' in name:
                    try:
                        model = HookedTransformer.from_pretrained_no_processing(
                            model_path,
                            dtype=self.dtype,
                            trust_remote_code=True
                        )
                        print(f"  Loaded {name}")
                    except Exception as gemma_error:
                        print(f"Gemma failed, checking memory and skipping: {gemma_error}")
                        continue
                else:
                    model = HookedTransformer.from_pretrained_no_processing(
                        model_path,
                        device=self.device,
                        dtype=self.dtype,
                        trust_remote_code=True
                    )
                
                model = model.to(self.device)
                
                self.models[name] = model
                print(f"✓ Loaded {name}")
                
            except Exception as e:
                print(f"✗ Failed to load {name}: {e}")
                continue
                
        return self.models
    
    def experiment_e1_mention_controlled_contrast(self, model: HookedTransformer, 
                                                 dataset: pd.DataFrame) -> pd.DataFrame:
        print("Running E1: Mention-Controlled Contrast (Simplified - Full 5k)...")
        
        forbidden_concepts = dataset['forbidden_concept'].tolist()
        proxy_dict = self._get_model_internal_proxies(model, forbidden_concepts)
        
        dataset = dataset.copy()
        dataset['proxy_concept'] = dataset['forbidden_concept'].map(proxy_dict)
        
        results = []
        batch_size = 128
        
        neg_data = dataset[dataset['prompt_type'] == 'negative'].copy()
        neu_data = dataset[dataset['prompt_type'] == 'neutral'].copy()
        pos_data = dataset[dataset['prompt_type'] == 'positive'].copy()
        
        print(f"Processing full dataset: {len(neg_data)} neg, {len(neu_data)} neu, {len(pos_data)} pos")
        
        def format_prompt(text, model_name):
            if 'gemma' in model_name.lower():
                return f"[INST] {text} [/INST]"
            return text
        
        model_name = getattr(model.cfg, 'model_name', 'unknown')
        
        all_data = []
        for df, ptype in [(neg_data, 'negative'), (neu_data, 'neutral'), (pos_data, 'positive')]:
            for _, row in df.iterrows():
                formatted_prompt = format_prompt(row['prompt_text'], model_name)
                all_data.append({
                    'id': row['id'],
                    'prompt_type': ptype,
                    'prompt_text': formatted_prompt,
                    'forbidden_concept': row['forbidden_concept']
                })
        
        print(f"Processing {len(all_data)} total prompts...")
        
        for i in range(0, len(all_data), batch_size):
            batch = all_data[i:i+batch_size]
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
                    forbidden_token = model.to_single_token(forbidden)
                    logp = F.log_softmax(logits[j], dim=-1)[forbidden_token].item()
                    
                    if logp == 0:
                        prob = F.softmax(logits[j], dim=-1)[forbidden_token].item()
                        print(f"Debug: logp=0, prob={prob:.6f} for id={item['id']}, prompt_type={item['prompt_type']}")
                    
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
        
        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0 or 'logp' not in results_df.columns:
            print("WARNING: No valid results collected, returning empty DF")
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
                    
                    if delta <= 0:
                        print(f"Debug: delta={delta:.6f} <= 0 for id={idx}, neutral={row['neutral']:.6f}, negative={row['negative']:.6f}")
                    
                    delta_results.append({
                        'id': idx,
                        'delta': delta,
                        'logp_neutral': row['neutral'],
                        'logp_negative': row['negative'],
                        'logp_positive': row.get('positive', np.nan),
                        'forbidden_concept': forbidden
                    })
        
        final_df = pd.DataFrame(delta_results)
        print(f"E1 completed: {len(final_df)} valid deltas from full 5k")
        return final_df
    
    def experiment_e4_load_distractor_effects(self, model: HookedTransformer,
                                             dataset: pd.DataFrame) -> pd.DataFrame:
        print("Running E4: Load/Distractor Effects (Simplified)...")
        
        results = []
        distractor_lengths = [0, 64, 256, 1024, 2048, 4096]
        distractor_types = ['syntactic', 'semantic', 'repetition']
        
        neg_data = dataset[dataset['prompt_type'] == 'negative']
        print(f"Processing FULL negative dataset for E4: {len(neg_data)} samples")
        
        batch_size = 32
        
        for dist_type in distractor_types:
            for length in distractor_lengths:
                print(f"  Processing {dist_type} distractors, length {length}...")
                
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
                        except Exception as e:
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
        
        print(f"E4 completed: {len(results_df)} samples from FULL negative dataset")
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
        print("Generating simplified visualizations...")
        
        e1_files = [f for f in os.listdir(results_dir) if 'e1_results' in f and 'reformatted' not in f]
        if e1_files:
            e1_data = pd.read_csv(os.path.join(results_dir, e1_files[0]))
            
            if len(e1_data) > 0 and 'delta' in e1_data.columns:
                # Save reformatted CSV (same as original but with explicit reformatted name)
                e1_data.to_csv(os.path.join(results_dir, 'e1_results_reformatted.csv'), index=False)
                print("  ✓ E1 reformatted CSV saved")
                
                # Generate histogram like simple_graph.png
                plt.figure(figsize=(12, 8))
                
                # Create histogram of delta scores
                delta_values = e1_data['delta'].dropna()
                mean_delta = delta_values.mean()
                
                # Create histogram
                plt.hist(delta_values, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
                
                # Add vertical line for average
                plt.axvline(x=mean_delta, color='red', linestyle='--', linewidth=2, 
                           label=f'Average = {mean_delta:.1f}')
                
                # Add styling and labels
                plt.xlabel('Delta Score (Difference Between Responses)', fontsize=14)
                plt.ylabel('Number of Examples', fontsize=14)
                
                # Format model name for title
                model_display = model_name.upper() if 'gpt2' in model_name.lower() else model_name.title()
                if 'opt' in model_name.lower():
                    model_display = model_name.upper()
                elif 'gemma' in model_name.lower():
                    model_display = model_name.title()
                    
                plt.title(f'{model_display}: How Much Models Distinguish Between Different Responses', 
                         fontsize=16, fontweight='bold')
                
                # Add text box with explanation
                textstr = f'Higher scores = Model can better distinguish\nbetween positive, negative, and neutral responses\nTotal examples: {len(delta_values):,}'
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
                        verticalalignment='top', bbox=props)
                
                # Add legend
                plt.legend(fontsize=12)
                
                # Add grid
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, 'simple_graph.png'), dpi=150, bbox_inches='tight')
                plt.close()
                print("  ✓ E1 histogram (simple_graph.png) saved")
                
                # Keep the old bar chart too
                plt.figure(figsize=(8, 6))
                
                stats = self.compute_statistics_fixed(e1_data, 'delta')
                
                plt.bar(['Delta'], [stats['mean']], 
                       yerr=[[stats['mean'] - stats['ci_lower']], [stats['ci_upper'] - stats['mean']]],
                       capsize=5, alpha=0.7, color='steelblue')
                plt.ylabel('Mean Delta (Log Probability)')
                plt.title('E1: Mention-Controlled Contrast - Delta Mean with 95% CI')
                plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                
                plt.text(0, stats['mean'] + (stats['ci_upper'] - stats['mean']) * 1.1, 
                        f"p_perm = {stats['p_perm']:.3f}", ha='center', fontsize=10)
                
                plt.savefig(os.path.join(results_dir, 'e1_simple_bar.png'), dpi=150, bbox_inches='tight')
                plt.close()
                print("  ✓ E1 bar chart saved")
            else:
                print("  ! E1 data empty or missing delta column")
        else:
            print("  ! No E1 files found")
        
        e4_files = [f for f in os.listdir(results_dir) if 'e4_results' in f]
        if e4_files:
            e4_data = pd.read_csv(os.path.join(results_dir, e4_files[0]))
            
            if len(e4_data) > 0:
                plt.figure(figsize=(12, 8))
                
                colors = ['blue', 'red', 'green']
                for i, dist_type in enumerate(e4_data['distractor_type'].unique()):
                    type_data = e4_data[e4_data['distractor_type'] == dist_type]
                    
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
                plt.title('E4: Load/Distractor Effects with 95% CI')
                plt.legend()
                plt.savefig(os.path.join(results_dir, 'e4_simple_line.png'), dpi=150, bbox_inches='tight')
                plt.close()
                print("  ✓ E4 line chart saved")
            else:
                print("  ! E4 data empty")
        else:
            print("  ! No E4 files found")
    
    def run_simplified_pipeline(self):
        print("=== Starting Simplified Ironic Rebound Experiments ===")
        print("Focus: E1 (mention contrast) + E4 (load effects) on full 5k samples")
        
        dataset = self.generate_dataset(5000)
        
        models = self.load_models()
        
        if not models:
            print("No models loaded successfully. Exiting.")
            return
        
        for model_name, model in models.items():
            print(f"\n=== Running simplified experiments for {model_name} ===")
            
            results_dir = f'results_{model_name}_simplified'
            os.makedirs(results_dir, exist_ok=True)
            
            try:
                print("Running E1 (Simplified)...")
                e1_results = self.experiment_e1_mention_controlled_contrast(model, dataset)
                e1_results.to_csv(os.path.join(results_dir, 'e1_results.csv'), index=False)
                
                e1_stats = self.compute_statistics_fixed(e1_results, 'delta')
                print(f"E1 Stats: mean={e1_stats['mean']:.4f}, p_perm={e1_stats['p_perm']:.4f}")
                
                print("Running E4 (Simplified)...")
                e4_results = self.experiment_e4_load_distractor_effects(model, dataset)
                e4_results.to_csv(os.path.join(results_dir, 'e4_results.csv'), index=False)
                
                self.generate_simple_visualizations(results_dir, model_name)
                
                summary_stats = {
                    'model': model_name,
                    'e1_mean_delta': e1_stats['mean'],
                    'e1_p_perm': e1_stats['p_perm'],
                    'e1_ci_lower': e1_stats['ci_lower'],
                    'e1_ci_upper': e1_stats['ci_upper'],
                    'total_samples_processed': len(dataset),
                    'e1_valid_deltas': len(e1_results),
                    'e4_samples': len(e4_results)
                }
                
                summary_df = pd.DataFrame([summary_stats])
                summary_df.to_csv(os.path.join(results_dir, 'summary_stats.csv'), index=False)
                
                print(f"✓ All simplified experiments completed for {model_name}")
                
            except Exception as e:
                print(f"✗ Error running simplified experiments for {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print("\n=== Simplified Pipeline completed ===")
        print("Results: E1 delta analysis + E4 load effects for all available models")

def main():
    experiments = SimplifiedIronicReboundExperiments(device='cuda', dtype=torch.bfloat16)
    experiments.run_simplified_pipeline()

if __name__ == "__main__":
    main()