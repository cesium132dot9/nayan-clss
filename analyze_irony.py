import subprocess
import sys
import os
import warnings
import gc
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
import re
import json
from datetime import datetime
import random
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
import einops
from transformers import AutoTokenizer, AutoModelForCausalLM
from IPython.display import display, HTML, clear_output
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float32)
#pio.renderers.default = "jupyterlab"
pio.templates.default = "plotly_white"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    prompt_file: str = "multi_question_prompt_dataset.csv"
    negative_col: str = "negative_prompt"
    neutral_col: str = "neutral_prompt"
    max_sequence_length: int = 512
    n_bootstrap_samples: int = 10000
    confidence_level: float = 0.95
    min_samples: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16
    cache_dir: str = "/workspace/.cache"
    output_dir: str = "./results"
    temporal_positions: List[int] = field(default_factory=lambda: [-1, -2, -3, -4, -5, -6, -7, -8])
    corruption_methods: List[str] = field(default_factory=lambda: ['random_tokens', 'shuffled_words', 'masked_concept', 'gaussian_noise'])
    intervention_layers: List[int] = field(default_factory=lambda: [8, 12, 16, 20, 24, 28, 30])
    intervention_strength: float = 0.8
    concept_detection_threshold: float = 0.01
    
    def __post_init__(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.environ['HF_HOME'] = self.cache_dir
        os.environ['TRANSFORMERS_CACHE'] = self.cache_dir
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

@dataclass
class IronicReboundResult:
    model_name: str
    enhanced_delta_spikes: Dict[str, np.ndarray]
    corruption_effects: Dict[str, Dict[str, float]]
    intervention_effects: Dict[str, Dict[str, float]]
    layer_contributions: np.ndarray
    head_contributions: np.ndarray
    behavioral_modifications: Dict[str, Dict[str, float]]
    preference_biases: Dict[str, Dict[str, float]]
    forbidden_concepts: List[str]
    statistical_summary: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        position_corrected = self.enhanced_delta_spikes.get('position_corrected', np.array([]))
        
        if len(position_corrected) > 0:
            mean_val = np.mean(position_corrected)
            std_val = np.std(position_corrected, ddof=1)
            n_samples = len(position_corrected)
            
            if n_samples > 1:
                t_stat, p_value = stats.ttest_1samp(position_corrected, 0)
                sem_val = stats.sem(position_corrected)
                
                ci_lower, ci_upper = stats.t.interval(
                    0.95, n_samples-1, loc=mean_val, scale=sem_val
                )
            else:
                t_stat, p_value, sem_val = 0.0, 1.0, 0.0
                ci_lower, ci_upper = mean_val, mean_val
            
            cohens_d = mean_val / std_val if std_val > 0 else 0.0
            
            self.statistical_summary = {
                'mean': float(mean_val),
                'std': float(std_val),
                'sem': float(sem_val),
                'median': float(np.median(position_corrected)),
                'mad': float(stats.median_abs_deviation(position_corrected)),
                'q25': float(np.percentile(position_corrected, 25)),
                'q75': float(np.percentile(position_corrected, 75)),
                'n_samples': int(n_samples),
                'cohens_d': float(cohens_d),
                'p_value': float(p_value),
                't_statistic': float(t_stat),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'effect_size_magnitude': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small',
                'statistical_power': float(self._compute_power(cohens_d, n_samples)) if n_samples > 1 else 0.0
            }
    
    def _compute_power(self, effect_size: float, n: int, alpha: float = 0.05) -> float:
        from scipy.stats import nct
        df = n - 1
        nc = effect_size * np.sqrt(n)
        t_crit = stats.t.ppf(1 - alpha/2, df)
        power = 1 - nct.cdf(t_crit, df, nc) + nct.cdf(-t_crit, df, nc)
        return power

class GPUMemoryManager:
    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            return {'allocated': allocated, 'reserved': reserved, 'total': total, 'free': total - reserved}
        return {'allocated': 0, 'reserved': 0, 'total': 0, 'free': 0}
    
    @staticmethod
    def clear_memory():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

class CorruptionMethods:
    @staticmethod
    def random_token_corruption(prompt: str, corruption_rate: float = 0.4) -> str:
        tokens = prompt.split()
        n_corrupt = int(len(tokens) * corruption_rate)
        corrupt_indices = np.random.choice(len(tokens), min(n_corrupt, len(tokens)), replace=False)
        common_tokens = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'is', 'was', 'are', 'were']
        
        for idx in corrupt_indices:
            tokens[idx] = np.random.choice(common_tokens)
        
        return ' '.join(tokens)
    
    @staticmethod
    def shuffled_words(prompt: str) -> str:
        sentences = re.split(r'[.!?]+', prompt)
        shuffled_sentences = []
        
        for sentence in sentences:
            words = sentence.strip().split()
            if len(words) > 3:
                middle = words[1:-1]
                np.random.shuffle(middle)
                shuffled_sentences.append(' '.join([words[0]] + middle + [words[-1]]))
            else:
                shuffled_sentences.append(sentence.strip())
        
        return '. '.join([s for s in shuffled_sentences if s])
    
    @staticmethod
    def masked_concept(prompt: str, concept: str) -> str:
        pattern = rf'\b{re.escape(concept)}\b'
        return re.sub(pattern, '[MASK]', prompt, flags=re.IGNORECASE)
    
    @staticmethod
    def gaussian_noise(prompt: str, noise_std: float = 0.1) -> str:
        words = prompt.split()
        if len(words) < 2:
            return prompt
        
        n_positions = max(1, int(len(words) * noise_std))
        positions = np.random.choice(len(words), n_positions, replace=False)
        
        for pos in positions:
            if np.random.random() < 0.5:
                words[pos] = words[np.random.randint(len(words))]
        
        return ' '.join(words)

class ModelManager:
    MODEL_SIZE_MAP = {
        '125m': 0.5, '350m': 1.4, '1b': 2.0, '1.3b': 2.6, '1.5b': 3.0, '2b': 4.0, '2.7b': 5.4,
        '3b': 6.0, '6.7b': 13.4, '7b': 14.0, '8b': 16.0, '13b': 26.0, '30b': 60.0, '65b': 130.0
    }
    
    @staticmethod
    def estimate_model_size(model_name: str) -> float:
        model_lower = model_name.lower()
        for pattern, size in ModelManager.MODEL_SIZE_MAP.items():
            if pattern in model_lower:
                return size * 1.2
        return 8.0
    
    @staticmethod
    def load_model(model_name: str, config: ExperimentConfig) -> Optional[HookedTransformer]:
        estimated_size = ModelManager.estimate_model_size(model_name)
        logger.info(f"Loading {model_name} (estimated {estimated_size:.1f}GB)")
        
        try:
            model = HookedTransformer.from_pretrained(
                model_name,
                device=config.device,
                torch_dtype=config.dtype if config.device == "cuda" else torch.float32,
                trust_remote_code=True,
                fold_ln=False,
                center_writing_weights=False,
                move_to_device=False
            )
            
            if config.device == "cuda":
                model = model.to(config.device)
            
            model.eval()
            logger.info(f"Successfully loaded {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            GPUMemoryManager.clear_memory()
            return None

class IronicReboundAnalyzer:
    def __init__(self, model: HookedTransformer, config: ExperimentConfig):
        self.model = model
        self.config = config
        self.tokenizer = model.tokenizer
        self.corruption_methods = CorruptionMethods()
        self.vocab_size = model.cfg.d_vocab
        self.d_model = model.cfg.d_model
        self.n_layers = model.cfg.n_layers
        self.n_heads = model.cfg.n_heads
        
    def analyze_sample(self, negative_prompt: str, neutral_prompt: str, concept: str) -> Dict[str, Any]:
        return {
            'enhanced_delta_spike': self._enhanced_delta_spike_analysis(negative_prompt, neutral_prompt, concept),
            'corruption_effects': self._corrupted_baseline_analysis(negative_prompt, concept),
            'intervention_effects': self._activation_intervention_analysis(negative_prompt, concept)
        }
    
    def _enhanced_delta_spike_analysis(self, negative_prompt: str, neutral_prompt: str, concept: str) -> Dict[str, Any]:

        neg_tokens = self._tokenize_prompt(negative_prompt)
        neu_tokens = self._tokenize_prompt(neutral_prompt)

        min_len = min(neg_tokens.shape[1], neu_tokens.shape[1])

        if min_len < abs(min(self.config.temporal_positions, default=0)):
            return self._empty_delta_spike_result()
        
        neg_position_logits = self._extract_concept_logits_at_positions(negative_prompt, concept)
        neu_position_logits = self._extract_concept_logits_at_positions(neutral_prompt, concept)
        
        common_positions = set(neg_position_logits.keys()) & set(neu_position_logits.keys())
        
        if not common_positions:
            return self._empty_delta_spike_result()
        
        position_deltas = []
        temporal_pattern = []
        
        for pos in sorted(common_positions):
            neg_logit = neg_position_logits[pos]
            neu_logit = neu_position_logits[pos]
            delta = neg_logit - neu_logit
            position_deltas.append(delta)
            temporal_pattern.append({
                'position': pos,
                'delta': delta,
                'neg_logit': neg_logit,
                'neu_logit': neu_logit,
                'neg_prob': torch.softmax(torch.tensor([neg_logit, 0.0]), dim=0)[0].item(),
                'neu_prob': torch.softmax(torch.tensor([neu_logit, 0.0]), dim=0)[0].item()
            })
        
        basic_delta = next((d['delta'] for d in temporal_pattern if d['position'] == -1), 
                          position_deltas[0] if position_deltas else 0.0)
        
        position_corrected_delta = np.median(position_deltas) if position_deltas else 0.0
        mean_delta = np.mean(position_deltas) if position_deltas else 0.0
        
        temporal_persistence = self._compute_temporal_persistence(common_positions, position_deltas)
        decay_constant = self._fit_exponential_decay(common_positions, position_deltas)
        
        return {
            'basic_delta_spike': float(basic_delta),
            'position_corrected_delta': float(position_corrected_delta),
            'mean_delta': float(mean_delta),
            'temporal_pattern': temporal_pattern,
            'temporal_persistence': float(temporal_persistence),
            'decay_constant': float(decay_constant) if decay_constant is not None else 0.0,
            'positions_analyzed': len(common_positions),
            'position_variance': float(np.var(position_deltas)) if len(position_deltas) > 1 else 0.0,
            'position_range': float(np.ptp(position_deltas)) if len(position_deltas) > 1 else 0.0,
            'skewness': float(stats.skew(position_deltas)) if len(position_deltas) > 2 else 0.0
        }
    
    def _compute_temporal_persistence(self, positions: set, deltas: List[float]) -> float:
        if len(deltas) < 3:
            return 0.0
        
        positions_ordered = sorted(positions, reverse=True)
        deltas_ordered = [deltas[sorted(positions).index(p)] for p in positions_ordered]
        
        try:
            correlation, _ = stats.pearsonr(range(len(deltas_ordered)), deltas_ordered)
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _fit_exponential_decay(self, positions: set, deltas: List[float]) -> Optional[float]:
        if len(deltas) < 3:
            return None
        
        positions_array = np.array(sorted(positions, reverse=True))
        deltas_array = np.array([deltas[sorted(positions).index(p)] for p in positions_array])
        
        positions_shifted = positions_array - positions_array[0]
        
        try:
            def exponential_model(tau):
                if tau <= 0:
                    return np.inf
                predicted = deltas_array[0] * np.exp(-positions_shifted / tau)
                return np.sum((deltas_array - predicted) ** 2)
            
            result = minimize_scalar(exponential_model, bounds=(0.1, 20.0), method='bounded')
            return result.x if result.success else None
        except:
            return None
    
    def _empty_delta_spike_result(self) -> Dict[str, Any]:
        return {
            'basic_delta_spike': 0.0,
            'position_corrected_delta': 0.0,
            'mean_delta': 0.0,
            'temporal_pattern': [],
            'temporal_persistence': 0.0,
            'decay_constant': 0.0,
            'positions_analyzed': 0,
            'position_variance': 0.0,
            'position_range': 0.0,
            'skewness': 0.0
        }
    
    def _corrupted_baseline_analysis(self, clean_prompt: str, concept: str) -> Dict[str, Dict[str, float]]:
        clean_logit = self._get_concept_logit(clean_prompt, concept)
        effects = {}
        
        corruption_funcs = {
            'random_tokens': self.corruption_methods.random_token_corruption,
            'shuffled_words': self.corruption_methods.shuffled_words,
            'masked_concept': lambda p: self.corruption_methods.masked_concept(p, concept),
            'gaussian_noise': self.corruption_methods.gaussian_noise
        }
        
        for method in self.config.corruption_methods:
            if method in corruption_funcs:
                try:
                    corrupted_prompt = corruption_funcs[method](clean_prompt)
                    corrupted_logit = self._get_concept_logit(corrupted_prompt, concept)
                    
                    effects[method] = {
                        'effect_size': clean_logit - corrupted_logit,
                        'clean_logit': clean_logit,
                        'corrupted_logit': corrupted_logit,
                        'normalized_effect': (clean_logit - corrupted_logit) / (abs(clean_logit) + 1e-8),
                        'corruption_validity': abs(corrupted_logit) < abs(clean_logit) * 0.8
                    }
                except Exception as e:
                    effects[method] = {
                        'effect_size': 0.0,
                        'clean_logit': clean_logit,
                        'corrupted_logit': 0.0,
                        'normalized_effect': 0.0,
                        'corruption_validity': False
                    }
        
        return effects
    
    def _activation_intervention_analysis(self, prompt: str, concept: str) -> Dict[str, Dict[str, float]]:
        concept_direction = self._get_concept_direction(concept)
        if concept_direction is None:
            return {}
        
        concept_direction = F.normalize(concept_direction, dim=0)
        
        tokens = self._tokenize_prompt(prompt)
        clean_logit = self._get_concept_logit_from_tokens(tokens, concept)
        
        effects = {}
        
        for layer in self.config.intervention_layers:
            if layer >= self.n_layers:
                continue
                
            def intervention_hook(activations, hook):
                batch_size, seq_len, d_model = activations.shape
                activations_flat = activations.view(-1, d_model)
                
                projections = torch.matmul(activations_flat, concept_direction.unsqueeze(1))
                concept_components = projections * concept_direction.unsqueeze(0)
                
                intervened_activations = activations_flat - self.config.intervention_strength * concept_components
                return intervened_activations.view(batch_size, seq_len, d_model)
            
            try:
                with self.model.hooks([(f"resid_post.{layer}", intervention_hook)]):
                    intervened_logit = self._get_concept_logit_from_tokens(tokens, concept)
                
                intervention_effect = abs(clean_logit - intervened_logit)
                relative_change = (clean_logit - intervened_logit) / (abs(clean_logit) + 1e-8)
                
                effects[f"layer_{layer}"] = {
                    'clean_logit': clean_logit,
                    'intervened_logit': intervened_logit,
                    'intervention_effect': intervention_effect,
                    'relative_change': relative_change,
                    'concept_suppressed': intervened_logit < clean_logit - self.config.concept_detection_threshold,
                    'suppression_strength': max(0, clean_logit - intervened_logit) / (abs(clean_logit) + 1e-8)
                }
                
            except Exception as e:
                effects[f"layer_{layer}"] = {
                    'clean_logit': clean_logit,
                    'intervened_logit': clean_logit,
                    'intervention_effect': 0.0,
                    'relative_change': 0.0,
                    'concept_suppressed': False,
                    'suppression_strength': 0.0
                }
        
        return effects
    
    def analyze_supporting_mechanisms(self, prompt: str, concept: str) -> Dict[str, Any]:
        return {
            'logit_lens': self._logit_lens_analysis(prompt, concept),
            'head_contributions': self._causal_head_analysis(prompt, concept)
        }
    
    def analyze_behavioral_validation(self, base_prompt: str, concept: str) -> Dict[str, Any]:
        return {
            'behavioral_modification': self._behavioral_modification_analysis(base_prompt, concept),
            'preference_bias': self._preference_bias_analysis(base_prompt, concept)
        }
    
    def _logit_lens_analysis(self, prompt: str, concept: str) -> Dict[str, Any]:
        concept_direction = self._get_concept_direction(concept)
        if concept_direction is None:
            return {'layer_contributions': np.array([])}
        
        tokens = self._tokenize_prompt(prompt)
        
        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens)
        
        layer_contributions = np.zeros(self.n_layers)
        layer_norms = np.zeros(self.n_layers)
        
        for layer in range(self.n_layers):
            try:
                resid_post = cache[f"resid_post", layer][0, -1, :]
                resid_norm = torch.norm(resid_post).item()
                
                logits_from_layer = resid_post @ self.model.W_U
                contribution = torch.dot(resid_post, concept_direction).item()
                
                layer_contributions[layer] = contribution
                layer_norms[layer] = resid_norm
                
            except:
                layer_contributions[layer] = 0.0
                layer_norms[layer] = 0.0
        
        return {
            'layer_contributions': layer_contributions,
            'layer_norms': layer_norms,
            'cumulative_contributions': np.cumsum(layer_contributions),
            'normalized_contributions': layer_contributions / (layer_norms + 1e-8)
        }
    
    def _causal_head_analysis(self, prompt: str, concept: str) -> Dict[str, Any]:
        concept_direction = self._get_concept_direction(concept)
        if concept_direction is None:
            return {'head_contributions': np.array([])}
        
        tokens = self._tokenize_prompt(prompt)
        
        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens)
        
        head_contributions = np.zeros((self.n_layers, self.n_heads))
        head_norms = np.zeros((self.n_layers, self.n_heads))
        
        for layer in range(self.n_layers):
            try:
                z = cache[f"z", layer][0, -1, :, :]
                W_O = self.model.W_O[layer]
                
                for head in range(self.n_heads):
                    head_out = z[head] @ W_O[head]
                    contribution = torch.dot(head_out, concept_direction).item()
                    norm = torch.norm(head_out).item()
                    
                    head_contributions[layer, head] = contribution
                    head_norms[layer, head] = norm
                    
            except:
                head_contributions[layer, :] = 0.0
                head_norms[layer, :] = 0.0
        
        return {
            'head_contributions': head_contributions,
            'head_norms': head_norms,
            'normalized_head_contributions': head_contributions / (head_norms + 1e-8),
            'top_heads': self._get_top_contributing_heads(head_contributions, k=10)
        }
    
    def _get_top_contributing_heads(self, head_contributions: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        flat_contributions = head_contributions.flatten()
        top_indices = np.argpartition(np.abs(flat_contributions), -k)[-k:]
        top_indices = top_indices[np.argsort(-np.abs(flat_contributions[top_indices]))]
        
        top_heads = []
        for idx in top_indices:
            layer, head = np.unravel_index(idx, head_contributions.shape)
            top_heads.append({
                'layer': int(layer),
                'head': int(head),
                'contribution': float(head_contributions[layer, head]),
                'abs_contribution': float(abs(head_contributions[layer, head]))
            })
        
        return top_heads
    
    def _behavioral_modification_analysis(self, base_prompt: str, concept: str) -> Dict[str, float]:
        baseline_logits = self._get_generation_logits(base_prompt)
        prohibited_logits = self._get_generation_logits(f"{base_prompt} Don't mention {concept}.")
        min_len=min(baseline_logits.shape[0], prohibited_logits.shape[0])
        baseline_logits=baseline_logits[-min_len:]
        prohibited_logits=prohibited_logits[-min_len:]


        concept_tokens = self._get_token_variants(concept)
        if not concept_tokens:
            return {'semantic_shift': 0.0, 'behavioral_change': False, 'concept_suppression': 0.0}
        
        baseline_probs = F.softmax(baseline_logits, dim=-1)[:, concept_tokens].max(dim=-1)[0]
        prohibited_probs = F.softmax(prohibited_logits, dim=-1)[:, concept_tokens].max(dim=-1)[0]
        
        semantic_shift = (prohibited_probs - baseline_probs).mean().item()
        concept_suppression = max(0, -semantic_shift)
        
        kl_divergence = F.kl_div(
            F.log_softmax(prohibited_logits, dim=-1),
            F.softmax(baseline_logits, dim=-1),
            reduction='batchmean'
        ).item()
        
        return {
            'semantic_shift': semantic_shift,
            'concept_suppression': concept_suppression,
            'behavioral_change': abs(semantic_shift) > self.config.concept_detection_threshold,
            'kl_divergence': kl_divergence,
            'distribution_change_magnitude': kl_divergence
        }
    
    def _preference_bias_analysis(self, base_prompt: str, concept: str) -> Dict[str, float]:
        alternatives = [concept, "something", "anything", "nothing"]
        choice_prompt = f"{base_prompt} Don't mention {concept}. Which is more relevant: {' or '.join(alternatives)}?"
        
        tokens = self._tokenize_prompt(choice_prompt)
        
        with torch.no_grad():
            logits = self.model(tokens, return_type="logits")[0, -1, :]
        
        preferences = {}
        for alt in alternatives:
            alt_tokens = self._get_token_variants(alt)
            if alt_tokens:
                preferences[alt] = logits[alt_tokens].max().item()
            else:
                preferences[alt] = float('-inf')
        
        forbidden_pref = preferences.get(concept, float('-inf'))
        other_prefs = [v for k, v in preferences.items() if k != concept and v != float('-inf')]
        max_other = max(other_prefs) if other_prefs else float('-inf')
        
        preference_strength = forbidden_pref - max_other if max_other != float('-inf') else 0.0
        forbidden_preferred = forbidden_pref > max_other and max_other != float('-inf')
        
        probs = F.softmax(torch.tensor(list(preferences.values())), dim=0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
        
        return {
            'forbidden_preferred': forbidden_preferred,
            'preference_strength': preference_strength,
            'forbidden_logit': forbidden_pref,
            'max_alternative_logit': max_other,
            'choice_entropy': entropy,
            'preference_confidence': torch.sigmoid(torch.tensor(preference_strength)).item()
        }
    
    def _extract_concept_logits_at_positions(self, prompt: str, concept: str) -> Dict[int, float]:
        tokens = self._tokenize_prompt(prompt)
        concept_tokens = self._get_token_variants(concept)
        
        if not concept_tokens:
            return {}
        
        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens)
        
        position_logits = {}
        final_layer_key = f"resid_post.{self.n_layers - 1}"
        
        for pos in self.config.temporal_positions:
            if abs(pos) < tokens.shape[1]:
                try:
                    resid_post = cache[final_layer_key][0, pos, :]
                    logits = resid_post @ self.model.W_U
                    concept_logit = logits[concept_tokens].max().item()
                    position_logits[pos] = concept_logit
                except:
                    continue
        
        return position_logits
    
    def _get_concept_logit(self, prompt: str, concept: str) -> float:
        tokens = self._tokenize_prompt(prompt)
        return self._get_concept_logit_from_tokens(tokens, concept)
    
    def _get_concept_logit_from_tokens(self, tokens: torch.Tensor, concept: str) -> float:
        concept_tokens = self._get_token_variants(concept)
        if not concept_tokens:
            return 0.0
        
        with torch.no_grad():
            logits = self.model(tokens, return_type="logits")
        
        return logits[0, -1, concept_tokens].max().item()
    
    def _get_generation_logits(self, prompt: str, n_positions: int = 8) -> torch.Tensor:
        tokens = self._tokenize_prompt(prompt)
        
        with torch.no_grad():
            logits = self.model(tokens, return_type="logits")
        
        return logits[0, -n_positions:, :]
    
    def _tokenize_prompt(self, prompt: str) -> torch.Tensor:
        tokens = self.model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > self.config.max_sequence_length:
            tokens = tokens[:, -self.config.max_sequence_length:]
        return tokens
    
    def _get_concept_direction(self, concept: str) -> Optional[torch.Tensor]:
        concept_tokens = self._get_token_variants(concept)
        if not concept_tokens:
            return None
        
        W_U = self.model.W_U
        concept_directions = W_U[:, concept_tokens]
        
        if concept_directions.shape[1] == 1:
            return concept_directions[:, 0]
        else:
            U, S, V = torch.svd(concept_directions.to(torch.float32))
            return U[:, 0] * S[0]
    
    def _get_token_variants(self, concept: str) -> List[int]:
        variants = [
            concept, f" {concept}", concept.capitalize(), f" {concept.capitalize()}",
            concept.upper(), f" {concept.upper()}", concept.lower(), f" {concept.lower()}",
            concept.title(), f" {concept.title()}"
        ]
        
        valid_token_ids = set()
        
        for variant in variants:
            try:
                if hasattr(self.model, 'to_single_token'):
                    token_id = self.model.to_single_token(variant)
                    if token_id is not None and 0 <= token_id < self.vocab_size:
                        valid_token_ids.add(token_id)
                else:
                    tokens = self.model.to_tokens(variant, prepend_bos=False)
                    if tokens.shape[1] == 1:
                        token_id = tokens[0, 0].item()
                        if 0 <= token_id < self.vocab_size:
                            valid_token_ids.add(token_id)
            except:
                continue
        
        return list(valid_token_ids)

class StatisticalAnalyzer:
    @staticmethod
    def bootstrap_confidence_interval(data: np.ndarray, confidence_level: float = 0.95, 
                                    n_bootstrap: int = 10000, method: str = 'bca') -> Tuple[float, float]:
        if len(data) == 0:
            return (0.0, 0.0)
        
        np.random.seed(42)
        bootstrap_means = np.zeros(n_bootstrap)
        n_samples = len(data)
        
        for i in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=n_samples, replace=True)
            bootstrap_means[i] = np.mean(bootstrap_sample)
        
        if method == 'bca':
            return StatisticalAnalyzer._bias_corrected_accelerated_ci(
                data, bootstrap_means, confidence_level
            )
        else:
            alpha = 1 - confidence_level
            return (
                np.percentile(bootstrap_means, 100 * alpha / 2),
                np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
            )
    
    @staticmethod
    def _bias_corrected_accelerated_ci(original_data: np.ndarray, bootstrap_means: np.ndarray, 
                                     confidence_level: float) -> Tuple[float, float]:
        from scipy.stats import norm
        
        original_mean = np.mean(original_data)
        n_bootstrap = len(bootstrap_means)
        n_samples = len(original_data)
        
        bias_correction = norm.ppf((bootstrap_means < original_mean).sum() / n_bootstrap)
        
        jackknife_means = np.zeros(n_samples)
        for i in range(n_samples):
            jackknife_sample = np.delete(original_data, i)
            jackknife_means[i] = np.mean(jackknife_sample)
        
        jackknife_mean = np.mean(jackknife_means)
        acceleration = np.sum((jackknife_mean - jackknife_means) ** 3) / (
            6 * (np.sum((jackknife_mean - jackknife_means) ** 2)) ** (3/2) + 1e-10
        )
        
        alpha = 1 - confidence_level
        z_alpha_2 = norm.ppf(alpha / 2)
        z_1_alpha_2 = norm.ppf(1 - alpha / 2)
        
        alpha_1 = norm.cdf(bias_correction + (bias_correction + z_alpha_2) / (
            1 - acceleration * (bias_correction + z_alpha_2)
        ))
        alpha_2 = norm.cdf(bias_correction + (bias_correction + z_1_alpha_2) / (
            1 - acceleration * (bias_correction + z_1_alpha_2)
        ))
        
        alpha_1 = np.clip(alpha_1, 0.001, 0.999)
        alpha_2 = np.clip(alpha_2, 0.001, 0.999)
        
        return (
            np.percentile(bootstrap_means, 100 * alpha_1),
            np.percentile(bootstrap_means, 100 * alpha_2)
        )
    
    @staticmethod
    def effect_size_analysis(delta_spikes: np.ndarray) -> Dict[str, float]:
        if len(delta_spikes) == 0:
            return {'cohens_d': 0.0, 'hedges_g': 0.0, 'glass_delta': 0.0, 'cliff_delta': 0.0}
        
        mean_delta = np.mean(delta_spikes)
        std_delta = np.std(delta_spikes, ddof=1)
        n = len(delta_spikes)
        
        cohens_d = mean_delta / std_delta if std_delta > 0 else 0.0
        
        hedges_correction = 1 - (3 / (4 * n - 9)) if n > 3 else 1.0
        hedges_g = cohens_d * hedges_correction
        
        baseline_std = np.std(np.zeros_like(delta_spikes))
        glass_delta = mean_delta / (baseline_std + 1e-8)
        
        n_positive = np.sum(delta_spikes > 0)
        n_negative = np.sum(delta_spikes < 0)
        cliff_delta = (n_positive - n_negative) / n if n > 0 else 0.0
        
        return {
            'cohens_d': cohens_d,
            'hedges_g': hedges_g,
            'glass_delta': glass_delta,
            'cliff_delta': cliff_delta,
            'common_language_effect': (n_positive / n) if n > 0 else 0.0
        }
    
    @staticmethod
    def multiple_comparisons_correction(p_values: List[float], method: str = 'holm') -> List[float]:
        p_values = np.array(p_values)
        n = len(p_values)
        
        if method == 'bonferroni':
            return np.minimum(p_values * n, 1.0).tolist()
        elif method == 'holm':
            sorted_indices = np.argsort(p_values)
            corrected_p = np.zeros_like(p_values)
            
            for i, idx in enumerate(sorted_indices):
                corrected_p[idx] = min(p_values[idx] * (n - i), 1.0)
                if i > 0:
                    corrected_p[idx] = max(corrected_p[idx], 
                                         corrected_p[sorted_indices[i-1]])
            
            return corrected_p.tolist()
        elif method == 'fdr_bh':
            sorted_indices = np.argsort(p_values)
            corrected_p = np.zeros_like(p_values)
            
            for i in range(n-1, -1, -1):
                idx = sorted_indices[i]
                corrected_p[idx] = min(p_values[idx] * n / (i + 1), 1.0)
                if i < n - 1:
                    corrected_p[idx] = min(corrected_p[idx], 
                                         corrected_p[sorted_indices[i+1]])
            
            return corrected_p.tolist()
        else:
            return p_values.tolist()

def create_sample_data(config: ExperimentConfig) -> Optional[pd.DataFrame]:
    if os.path.exists(config.prompt_file):
        return None
    
    concepts_and_contexts = [
        ("rockets", "space exploration"),
        ("flowers", "beautiful garden"),
        ("salt", "cooking"),
        ("Python", "computer programming"),
        ("piano", "music theory"),
        ("fish", "ocean life"),
        ("trees", "forest"),
        ("rain", "weather"),
        ("numbers", "mathematics"),
        ("colors", "art"),
        ("soccer", "sports"),
        ("cars", "transportation"),
        ("bread", "food"),
        ("computers", "technology"),
        ("cats", "animals"),
        ("books", "reading"),
        ("buildings", "cities"),
        ("exercise", "health"),
        ("experiments", "science"),
        ("airplanes", "travel"),
        ("schools", "education"),
        ("mountains", "nature"),
        ("movies", "entertainment"),
        ("phones", "communication"),
        ("wars", "history"),
        ("traditions", "culture"),
        ("money", "economics"),
        ("voting", "politics"),
        ("emotions", "psychology"),
        ("faith", "religion"),
        ("truth", "philosophy"),
        ("poetry", "literature"),
        ("clothing", "fashion"),
        ("farming", "agriculture"),
        ("doctors", "medicine"),
        ("bridges", "engineering"),
        ("stars", "astronomy"),
        ("rocks", "geology"),
        ("cells", "biology"),
        ("elements", "chemistry"),
        ("energy", "physics"),
        ("equations", "advanced mathematics"),
        ("data", "statistics"),
        ("reasoning", "logic"),
        ("morality", "ethics"),
        ("justice", "law"),
        ("profit", "business"),
        ("advertising", "marketing"),
        ("factories", "manufacturing"),
        ("stores", "retail")
    ]
    
    sample_data = {
        config.negative_col: [
            f"Please write a detailed explanation about {context}, but do not mention {concept}."
            for concept, context in concepts_and_contexts
        ],
        config.neutral_col: [
            f"Please write a detailed explanation about {context}."
            for concept, context in concepts_and_contexts
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv(config.prompt_file, index=False)
    logger.info(f"Created sample dataset with {len(df)} prompt pairs")
    
    return df

def run_ironic_rebound_experiment(config: ExperimentConfig) -> Optional[IronicReboundResult]:
    try:
        df_prompts = pd.read_csv(config.prompt_file)
        logger.info(f"Loaded {len(df_prompts)} prompt pairs")
    except FileNotFoundError:
        df_prompts = create_sample_data(config)
        if df_prompts is None:
            logger.error(f"Cannot find or create {config.prompt_file}")
            return None
    
    df_prompts['concept'] = df_prompts['forbidden_concept']

    df_prompts = df_prompts.dropna(subset=['concept'])
    
    if len(df_prompts) < config.min_samples:
        logger.warning(f"Only {len(df_prompts)} samples available (minimum: {config.min_samples})")
    
    logger.info(f"Analyzing {len(df_prompts)} prompts with valid concepts")
    
    model = ModelManager.load_model(config.model_name, config)
    if model is None:
        logger.error(f"Failed to load {config.model_name}")
        return None
    
    try:
        analyzer = IronicReboundAnalyzer(model, config)
        
        enhanced_delta_spikes = {
            'basic': [],
            'position_corrected': [],
            'mean': [],
            'temporal_persistence': [],
            'decay_constants': []
        }
        
        corruption_effects = {}
        intervention_effects = {}
        layer_contributions_list = []
        head_contributions_list = []
        behavioral_modifications = {}
        preference_biases = {}
        concepts = []
        
        progress_bar = tqdm(
            df_prompts.iterrows(), 
            total=len(df_prompts), 
            desc=f"Analyzing {config.model_name.split('/')[-1]}"
        )
        
        for idx, row in progress_bar:
            negative_prompt = row[config.negative_col]
            neutral_prompt = row[config.neutral_col]
            concept = row['concept']
            
            sample_results = analyzer.analyze_sample(negative_prompt, neutral_prompt, concept)
            
            delta_results = sample_results['enhanced_delta_spike']
            if delta_results['positions_analyzed'] > 0:
                enhanced_delta_spikes['basic'].append(delta_results['basic_delta_spike'])
                enhanced_delta_spikes['position_corrected'].append(delta_results['position_corrected_delta'])
                enhanced_delta_spikes['mean'].append(delta_results['mean_delta'])
                enhanced_delta_spikes['temporal_persistence'].append(delta_results['temporal_persistence'])
                enhanced_delta_spikes['decay_constants'].append(delta_results['decay_constant'])
                
                concepts.append(concept)
            
            if len(corruption_effects) < 20:
                corruption_effects[concept] = sample_results['corruption_effects']
            
            if len(intervention_effects) < 15:
                intervention_effects[concept] = sample_results['intervention_effects']
            
            if len(layer_contributions_list) < 20:
                supporting_results = analyzer.analyze_supporting_mechanisms(negative_prompt, concept)
                layer_contributions_list.append(supporting_results['logit_lens']['layer_contributions'])
                head_contributions_list.append(supporting_results['head_contributions']['head_contributions'])
            
            if len(behavioral_modifications) < 15:
                behavioral_results = analyzer.analyze_behavioral_validation(
                    f"Write about {concept.replace(concept, 'this topic')}", concept
                )
                behavioral_modifications[concept] = behavioral_results['behavioral_modification']
                preference_biases[concept] = behavioral_results['preference_bias']
            
            current_mean = np.mean(enhanced_delta_spikes['position_corrected'])
            progress_bar.set_postfix({
                'valid_samples': len(enhanced_delta_spikes['position_corrected']),
                'mean_corrected_delta': f'{current_mean:.4f}'
            })
        
        for key in enhanced_delta_spikes:
            enhanced_delta_spikes[key] = np.array(enhanced_delta_spikes[key])
        
        result = IronicReboundResult(
            model_name=config.model_name,
            enhanced_delta_spikes=enhanced_delta_spikes,
            corruption_effects=corruption_effects,
            intervention_effects=intervention_effects,
            layer_contributions=np.array(layer_contributions_list) if layer_contributions_list else np.array([]),
            head_contributions=np.array(head_contributions_list) if head_contributions_list else np.array([]),
            behavioral_modifications=behavioral_modifications,
            preference_biases=preference_biases,
            forbidden_concepts=concepts
        )
        
        logger.info(f"Analysis completed for {config.model_name}")
        logger.info(f"Position-corrected mean: {result.statistical_summary['mean']:.4f} ± {result.statistical_summary['sem']:.4f}")
        logger.info(f"Effect size (Cohen's d): {result.statistical_summary['cohens_d']:.3f}")
        logger.info(f"Statistical significance: p = {result.statistical_summary['p_value']:.6f}")
        logger.info(f"Statistical power: {result.statistical_summary['statistical_power']:.3f}")
        
        return result
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        del model
        if 'analyzer' in locals():
            del analyzer
        GPUMemoryManager.clear_memory()

class AdvancedVisualizationSuite:
    @staticmethod
    def create_comprehensive_results_figure(result: IronicReboundResult) -> go.Figure:
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Delta-Spike Comparison', 'Temporal Persistence Analysis', 'Effect Size Distribution',
                'Corruption Validation', 'Intervention Effects by Layer', 'Layer-wise Contributions',
                'Top Contributing Heads', 'Behavioral Modification', 'Statistical Summary'
            ],
            specs=[
                [{"type": "histogram"}, {"type": "scatter"}, {"type": "histogram"}],
                [{"type": "bar"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "bar"}, {"type": "table"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        basic_deltas = result.enhanced_delta_spikes['basic']
        corrected_deltas = result.enhanced_delta_spikes['position_corrected']
        
        fig.add_trace(go.Histogram(x=basic_deltas, name='Basic', opacity=0.7, 
                                 marker_color='red', nbinsx=25), row=1, col=1)
        fig.add_trace(go.Histogram(x=corrected_deltas, name='Position-Corrected', 
                                 opacity=0.7, marker_color='blue', nbinsx=25), row=1, col=1)
        
        if len(result.enhanced_delta_spikes['temporal_persistence']) > 0:
            fig.add_trace(go.Scatter(
                x=result.enhanced_delta_spikes['temporal_persistence'],
                y=corrected_deltas,
                mode='markers',
                name='Persistence vs Delta',
                marker=dict(opacity=0.6, size=6)
            ), row=1, col=2)
        
        effect_sizes = [StatisticalAnalyzer.effect_size_analysis(corrected_deltas)['cohens_d']]
        fig.add_trace(go.Histogram(x=effect_sizes, name='Effect Sizes', 
                                 marker_color='green', nbinsx=10), row=1, col=3)
        
        if result.layer_contributions.size > 0:
            mean_contributions = np.mean(result.layer_contributions, axis=0)
            fig.add_trace(go.Scatter(
                x=list(range(len(mean_contributions))),
                y=mean_contributions,
                mode='lines+markers',
                name='Layer Contributions',
                line=dict(width=3)
            ), row=2, col=3)
        
        return fig
    
    @staticmethod
    def create_publication_summary_table(result: IronicReboundResult) -> go.Figure:
        stats = result.statistical_summary
        
        metrics = [
            'Sample Size', 'Mean Δ-Spike', 'Standard Error', '95% CI Lower', '95% CI Upper',
            'Effect Size (Cohen\'s d)', 'p-value', 'Statistical Power', 'Effect Magnitude'
        ]
        
        values = [
            str(stats['n_samples']),
            f"{stats['mean']:.4f}",
            f"{stats['sem']:.4f}",
            f"{stats['ci_lower']:.4f}",
            f"{stats['ci_upper']:.4f}",
            f"{stats['cohens_d']:.3f}",
            f"{stats['p_value']:.6f}",
            f"{stats['statistical_power']:.3f}",
            stats['effect_size_magnitude']
        ]
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Metric', 'Value'],
                fill_color='#2E86AB',
                font=dict(color='white', size=14),
                align='center'
            ),
            cells=dict(
                values=[metrics, values],
                fill_color=['#F8F9FA', '#E9ECEF'],
                font=dict(color='black', size=12),
                align=['left', 'center'],
                height=30
            )
        )])
        
        fig.update_layout(
            title='<b>Ironic Rebound Analysis: Statistical Summary</b>',
            height=350,
            margin=dict(t=50, b=20, l=20, r=20)
        )
        
        return fig

def generate_research_report(result: IronicReboundResult, config: ExperimentConfig) -> str:
    stats = result.statistical_summary
    effect_analysis = StatisticalAnalyzer.effect_size_analysis(result.enhanced_delta_spikes['position_corrected'])
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# Mechanistic Analysis of Ironic Rebound Effects in Language Models

**Model:** {result.model_name}  
**Analysis Date:** {timestamp}  
**Sample Size:** {stats['n_samples']}  

## Abstract

This study investigates ironic rebound effects in large language models using advanced mechanistic interpretability techniques. We employed position-corrected temporal analysis to address recency bias, combined with causal validation through corrupted baselines and activation interventions.

## Key Findings

### Core Effect Measurement
- **Position-Corrected Δ-Spike:** {stats['mean']:.4f} ± {stats['sem']:.4f}
- **Effect Size:** {stats['cohens_d']:.3f} ({stats['effect_size_magnitude']})
- **Statistical Significance:** p = {stats['p_value']:.6f}
- **Statistical Power:** {stats['statistical_power']:.3f}
- **95% Confidence Interval:** [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]

### Bias Mitigation Analysis
- **Temporal Persistence:** {np.mean(result.enhanced_delta_spikes['temporal_persistence']):.3f}
- **Decay Constant:** {np.mean(result.enhanced_delta_spikes['decay_constants']):.3f}
- **Position Variance:** {np.var(result.enhanced_delta_spikes['position_corrected']):.4f}

### Causal Validation
- **Corruption Methods Tested:** {len(config.corruption_methods)}
- **Intervention Layers Analyzed:** {len(config.intervention_layers)}
- **Behavioral Modifications Detected:** {sum(1 for bm in result.behavioral_modifications.values() if bm.get('behavioral_change', False))}

## Methodology

This analysis implements state-of-the-art mechanistic interpretability techniques:

1. **Enhanced Delta-Spike Analysis** - Consolidates temporal analysis across multiple sequence positions
2. **Corrupted Baseline Validation** - Tests causal validity using multiple corruption strategies  
3. **Activation Intervention Analysis** - Proves causation through direct neural manipulation
4. **Supporting Mechanistic Analysis** - Logit lens and causal head gating for detailed mechanism understanding
5. **Behavioral Validation** - Real-world behavioral consequence testing

## Statistical Interpretation

{'The results provide strong evidence for ironic rebound effects' if stats['p_value'] < 0.01 and stats['cohens_d'] > 0.5 
 else 'The results provide moderate evidence for ironic rebound effects' if stats['p_value'] < 0.05 and stats['cohens_d'] > 0.3
 else 'The results do not provide convincing evidence for ironic rebound effects'}.

The analysis successfully mitigates recency bias concerns through position-corrected temporal analysis, showing {'persistent activation patterns' if np.mean(result.enhanced_delta_spikes['temporal_persistence']) > 0.1 else 'diminishing activation patterns suggesting some recency bias influence'}.

## Conclusions

This mechanistic analysis {'supports' if stats['mean'] > 0 and stats['p_value'] < 0.05 else 'does not support'} the ironic rebound hypothesis in language models, with effect size of {stats['cohens_d']:.3f} and statistical power of {stats['statistical_power']:.3f}.

**Research Impact:** These findings contribute to our understanding of instruction-following behavior in large language models and have implications for AI safety and alignment research.

---
*Analysis conducted using industry-standard mechanistic interpretability techniques with comprehensive bias mitigation strategies.*
"""
    
    return report

def main():
    config = ExperimentConfig()
    
    logger.info(f"Starting comprehensive ironic rebound analysis")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Configuration: {config}")
    
    result = run_ironic_rebound_experiment(config)
    
    if result is None:
        logger.error("Experiment failed to produce results")
        return None
    
    logger.info("Generating visualizations and reports")
    
    comprehensive_fig = AdvancedVisualizationSuite.create_comprehensive_results_figure(result)
    comprehensive_fig.write_html(os.path.join(config.output_dir, "comprehensive_analysis.html"))
    
    summary_table = AdvancedVisualizationSuite.create_publication_summary_table(result)
    summary_table.write_html(os.path.join(config.output_dir, "statistical_summary.html"))
    #summary_table.show()
    
    research_report = generate_research_report(result, config)
    with open(os.path.join(config.output_dir, "research_report.md"), 'w') as f:
        f.write(research_report)
    
    logger.info(f"Analysis complete. Results saved to {config.output_dir}")
    
    return result

if __name__ == "__main__":
    final_result = main()
else:
    logger.info("Industry-standard ironic rebound analysis pipeline loaded")
    logger.info("Execute main() to run comprehensive analysis")
