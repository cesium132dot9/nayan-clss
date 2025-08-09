# --- Core Imports ---
import torch
import numpy as np
import pandas as pd
import json
import os
import hashlib
import pickle
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

# --- ML & Mech Interp Imports ---
from transformer_lens import HookedTransformer
from openai import OpenAI

# --- Stats & NLP Imports ---
from scipy import stats
from nltk.corpus import wordnet as wn
from tqdm import tqdm

# --- Visualization Imports ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration Dataclass ---
@dataclass(frozen=True)
class Config:
    """Central configuration for all experimental parameters."""
    MODEL_NAME: str = "gpt2-small"
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED: int = 42
    N_STIMULI: int = 1000
    N_CONSTRAINTS_PER_STIMULUS: int = 7
    WORDNET_PATH_THRESHOLD: int = 3
    ANALYSIS_WINDOW_START: int = 10
    ANALYSIS_WINDOW_END: int = 50
    POSITION_WEIGHT_DECAY: float = 2.0
    SUPPRESSION_HEAD_LAYER: int = 10
    SUPPRESSION_HEAD_INDEX: int = 7
    ALPHA: float = 0.001
    N_PERMUTATIONS: int = 1000

# --- Data Structure Dataclasses ---
@dataclass
class ExperimentalStimulus:
    stimulus_id: int
    sentence: str
    constraint_set: List[str]

@dataclass
class ExperimentResult:
    stimulus_id: int
    n_constraints: int
    delta_logit: float
    l10h7_activation: float
    l10h7_zscore: float

class DatasetGenerator:
    """Handles generation of semantically independent stimuli using GPT-4 and WordNet."""
    def __init__(self, api_key: str, config: Config):
        self.client = OpenAI(api_key=api_key)
        self.cfg = config
        self.concept_pool = [
            'apple', 'mountain', 'river', 'computer', 'bicycle', 'flower', 'planet', 
            'music', 'building', 'forest', 'diamond', 'elephant', 'rainbow', 'telescope', 
            'volcano', 'butterfly', 'pyramid', 'lightning', 'chocolate', 'robot', 'guitar',
            'ocean', 'castle', 'bridge', 'moon', 'star', 'key', 'book', 'clock', 'door'
        ]

    def _compute_wordnet_distance(self, word1: str, word2: str) -> float:
        syns1, syns2 = wn.synsets(word1), wn.synsets(word2)
        if not syns1 or not syns2: return float('inf')
        path_distances = [
            s1.shortest_path_distance(s2) for s1 in syns1 for s2 in syns2 if s1.shortest_path_distance(s2) is not None
        ]
        return min(path_distances) if path_distances else float('inf')

    def _select_independent_concepts(self) -> Optional[List[str]]:
        for _ in range(100):
            candidates = np.random.choice(self.concept_pool, self.cfg.N_CONSTRAINTS_PER_STIMULUS, replace=False).tolist()
            is_valid = all(
                self._compute_wordnet_distance(c1, c2) >= self.cfg.WORDNET_PATH_THRESHOLD
                for i, c1 in enumerate(candidates) for c2 in candidates[i+1:]
            )
            if is_valid: return candidates
        return None

    def generate_stimuli(self) -> List[ExperimentalStimulus]:
        stimuli = []
        prompt = "Generate a neutral declarative sentence about an everyday topic, 15-20 words long."
        pbar = tqdm(range(self.cfg.N_STIMULI), desc="Generating stimuli")
        while len(stimuli) < self.cfg.N_STIMULI:
            response = self.client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])
            sentence = response.choices[0].message.content.strip()
            concepts = self._select_independent_concepts()
            if concepts:
                stimuli.append(ExperimentalStimulus(stimulus_id=len(stimuli), sentence=sentence, constraint_set=concepts))
                pbar.update(1)
        pbar.close()
        return stimuli

class IronicSuppressionExperiment:
    """Contains the core logic for running the experiment on the model."""
    def __init__(self, config: Config):
        self.cfg = config
        self.model = HookedTransformer.from_pretrained(config.MODEL_NAME, device=config.DEVICE)
        self.model.eval()

    def _calculate_position_weight(self, position: int) -> float:
        if not self.cfg.ANALYSIS_WINDOW_START <= position <= self.cfg.ANALYSIS_WINDOW_END: return 0.0
        relative_pos = (position - self.cfg.ANALYSIS_WINDOW_START) / (self.cfg.ANALYSIS_WINDOW_END - self.cfg.ANALYSIS_WINDOW_START)
        return np.exp(-self.cfg.POSITION_WEIGHT_DECAY * relative_pos)

    def _get_head_activation(self, cache: Dict) -> float:
        key = f"blocks.{self.cfg.SUPPRESSION_HEAD_LAYER}.attn.hook_z"
        return cache[key][0, :, self.cfg.SUPPRESSION_HEAD_INDEX, :].mean().item()

    def measure_suppression_effect(self, stimulus: ExperimentalStimulus, n_constraints: int) -> ExperimentResult:
        constraints = stimulus.constraint_set[:n_constraints]
        constraint_text = " or ".join(constraints)
        suppress_prompt = f"Don't mention {constraint_text}. {stimulus.sentence}"
        
        with torch.no_grad():
            hook_name = f"blocks.{self.cfg.SUPPRESSION_HEAD_LAYER}.attn.hook_z"
            names_filter = [hook_name, "blocks.0.ln1.hook_scale"]
            neutral_logits, neutral_cache = self.model.run_with_cache(stimulus.sentence, names_filter=names_filter)
            suppress_logits, suppress_cache = self.model.run_with_cache(suppress_prompt, names_filter=names_filter)
        
        instruction_end = self.model.to_tokens(f"Don't mention {constraint_text}. ").shape[1]
        
        weighted_deltas, weights = [], []
        for pos in range(instruction_end + self.cfg.ANALYSIS_WINDOW_START, min(instruction_end + self.cfg.ANALYSIS_WINDOW_END, suppress_logits.shape[1])):
            weight = self._calculate_position_weight(pos - instruction_end)
            if weight == 0.0: continue
            
            p_suppress = torch.softmax(suppress_logits[0, pos-1, :], dim=-1)
            p_neutral = torch.softmax(neutral_logits[0, min(pos-1, neutral_logits.shape[1]-1), :], dim=-1)
            
            for target in constraints:
                target_id = self.model.to_single_token(f" {target}")
                delta = (torch.log(p_suppress[target_id] + 1e-10) - torch.log(p_neutral[target_id] + 1e-10)).item()
                weighted_deltas.append(delta * weight)
                weights.append(weight)

        mean_delta = sum(weighted_deltas) / (sum(weights) if weights else 1.0)
        act_suppress = self._get_head_activation(suppress_cache)
        z_score = (act_suppress - self._get_head_activation(neutral_cache)) / (neutral_cache["blocks.0.ln1.hook_scale"].mean().item() + 1e-10)

        return ExperimentResult(stimulus.stimulus_id, n_constraints, mean_delta, act_suppress, z_score)

class StatisticalValidator:
    """Implements the statistical tests, including permutation tests."""
    def __init__(self, config: Config):
        self.cfg = config

    def _permutation_test(self, data: np.ndarray, baseline: float, alternative: str) -> float:
        observed_mean = np.mean(data)
        null_dist = np.array([np.mean(np.random.permutation(data)) for _ in range(self.cfg.N_PERMUTATIONS)])
        if alternative == 'less': return np.mean(null_dist <= observed_mean)
        elif alternative == 'greater': return np.mean(null_dist >= observed_mean)
        return 0.0

    def validate_hypotheses(self, df: pd.DataFrame) -> Dict:
        validation = {}
        h1_data = df[df.n_constraints <= 3]['delta_logit'].values
        p_t, p_perm = stats.ttest_1samp(h1_data, -0.3, alternative='less').pvalue, self._permutation_test(h1_data, -0.3, 'less')
        validation['H1_anti_rebound_passed'] = bool(min(p_t, p_perm) < self.cfg.ALPHA)
        
        h2_data = df[df.n_constraints >= 6]['delta_logit'].values
        p_t, p_perm = stats.ttest_1samp(h2_data, 0.2, alternative='greater').pvalue, self._permutation_test(h2_data, 0.2, 'greater')
        validation['H2_load_reversal_passed'] = bool(min(p_t, p_perm) < self.cfg.ALPHA)
        
        z_scores = df['l10h7_zscore'].values
        validation['H3_activation_passed'] = bool(np.mean(z_scores) > 1.0)
        validation['H3_mean_zscore'] = float(np.mean(z_scores))
        return validation

def create_visualizations(df: pd.DataFrame, output_dir: str):
    """Generates and saves summary plots of the results."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.set_style("whitegrid")
    sns.pointplot(x='n_constraints', y='delta_logit', data=df, ax=axes[0], capsize=.2, errorbar='se')
    axes[0].axhline(y=-0.3, color='r', linestyle='--', label='H1 Threshold (Anti-Rebound)')
    axes[0].axhline(y=0.2, color='g', linestyle='--', label='H2 Threshold (Reversal)')
    axes[0].set_title('Ironic Suppression Effect by Cognitive Load', fontsize=14)
    axes[0].set_xlabel('Number of Constraints'); axes[0].set_ylabel('Mean Weighted Delta Logit'); axes[0].legend()
    sns.regplot(x='l10h7_zscore', y='delta_logit', data=df, ax=axes[1], scatter_kws={'alpha':0.2})
    axes[1].set_title('Suppression Head Activation vs. Suppression Effect', fontsize=14)
    axes[1].set_xlabel('L10H7 Activation (Z-Score)'); axes[1].set_ylabel('Mean Weighted Delta Logit')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_plots.png'), dpi=300)
    print(f"Visualizations saved to {os.path.join(output_dir, 'summary_plots.png')}")

def main():
    """Main execution function for the experiment."""
    cfg = Config()
    warnings.filterwarnings('ignore')
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    
    existing_dirs = sorted([d for d in os.listdir() if d.startswith('results_')])
    latest_dir = existing_dirs[-1] if existing_dirs else None
    
    output_dir = f"results_{cfg.MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(latest_dir, "raw_results.csv") if latest_dir else None
    
    if results_path and os.path.exists(results_path):
        print(f"Loading existing results from {results_path} for re-analysis...")
        results_df = pd.read_csv(results_path)
    else:
        print(f"Starting experiment. Results will be saved to: {output_dir}")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key: raise ValueError("OPENAI_API_KEY environment variable not set.")
        
        stimuli_path = os.path.join(output_dir, "stimuli.pkl")
        if os.path.exists(stimuli_path):
            print(f"Loading existing stimuli from {stimuli_path}...")
            with open(stimuli_path, 'rb') as f: stimuli = pickle.load(f)
        else:
            print("Generating new stimuli...")
            generator = DatasetGenerator(api_key=api_key, config=cfg)
            stimuli = generator.generate_stimuli()
            with open(os.path.join(output_dir, "stimuli.pkl"), 'wb') as f: pickle.dump(stimuli, f)
            print(f"Generated and saved {len(stimuli)} stimuli.")
        
        experiment = IronicSuppressionExperiment(config=cfg)
        results = [
            experiment.measure_suppression_effect(s, n)
            for s in tqdm(stimuli, desc="Processing stimuli")
            for n in range(1, cfg.N_CONSTRAINTS_PER_STIMULUS + 1)
        ]
        results_df = pd.DataFrame([vars(r) for r in results])
        results_df.to_csv(os.path.join(output_dir, "raw_results.csv"), index=False)
        print("Experiment measurements complete.")

    # --- Analysis and Visualization ---
    aggregated_df = results_df.groupby(['stimulus_id', 'n_constraints']).mean().reset_index()
    validator = StatisticalValidator(config=cfg)
    validation_results = validator.validate_hypotheses(aggregated_df)
    print("\n--- Statistical Validation Summary ---")
    print(json.dumps(validation_results, indent=2))
    with open(os.path.join(output_dir, "validation.json"), 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    create_visualizations(aggregated_df, output_dir)
    print("\nExperiment finished successfully.")

if __name__ == "__main__":
    main()