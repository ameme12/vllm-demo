"""
GeoMLAMA Task - Adapted for Causal Language Models
Works with vLLM runner using Llama/Qwen models
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import re
from tasks.base_task import BaseTask


# Language to country mapping (based on GeoMLAMA filename convention)
LANGUAGE_TO_COUNTRY = {
    'en': {'idx': 0, 'name': 'United States', 'code': 'en'},
    'zh': {'idx': 1, 'name': 'China', 'code': 'zh'},
    'hi': {'idx': 2, 'name': 'India', 'code': 'hi'},
    'fa': {'idx': 3, 'name': 'Iran', 'code': 'fa'},
    'sw': {'idx': 4, 'name': 'Kenya', 'code': 'sw'},
}


class GeoMLAMATask(BaseTask):
    """
    GeoMLAMA Task adapted for causal language models (Llama, Qwen, etc.)
    
    Evaluates geographical/cultural bias by testing model's knowledge 
    about different countries with and without country context.
    
    Original GeoMLAMA uses masked language modeling, but this adaptation
    converts prompts to open-ended questions for generative models.
    
    Note: Each language file (en, zh, hi, fa, sw) corresponds to one country.
    The country is determined by the language code in the filename.
    """

    def __init__(self, task_name: str, dataset_path: Path, config: Dict[str, Any]):
        super().__init__(task_name, dataset_path, config)
        
        # Configuration options
        self.language = config.get('language', 'en')  # Default to English
        self.with_country = config.get('with_country', True)  # Whether to use country context
        
        # File paths
        self.main_file = dataset_path / f"gd_prompt_{self.language}.tsv"
        self.aug_file = dataset_path / f"gd_prompt_{self.language}_aug.tsv"
        
        # Get country info for this language
        self.country_info = LANGUAGE_TO_COUNTRY.get(self.language, LANGUAGE_TO_COUNTRY['en'])
        
        self.dataset = None

    def load_dataset(self):
        """Load the GeoMLAMA dataset from TSV files (raw data only)."""
        data = []
        
        # Load main file
        if not self.main_file.exists():
            raise ValueError(f"Main file not found: {self.main_file}")
        
        print(f"\n{'='*60}")
        print(f"Loading GeoMLAMA Dataset")
        print(f"{'='*60}")
        print(f"Language: {self.language}")
        print(f"Country: {self.country_info['name']}")
        print(f"With Country Context: {self.with_country}")
        
        with open(self.main_file, 'r', encoding='utf-8') as f:
            for line in f:
                d = line.strip().split('\t')
                
                # Skip header and empty lines
                if d[0] == "Prompt" or len(d) <= 1:
                    continue
                
                # Parse the data
                if len(d) == 5:
                    d += ["", ""]
                
                # Split answer candidates and gold answers
                d[-4] = d[-4].split(', ')  # answer_cand
                d[-5] = d[-5].split(', ')  # gold_ans_list
                
                data.append(d)
        
        # Load augmented file if it exists
        aug_data = []
        if self.aug_file.exists():
            with open(self.aug_file, 'r', encoding='utf-8') as f:
                idx = 0
                for line in f:
                    d = line.strip().split('\t')
                    # Add gold answers and candidates from main data
                    d.append(data[int(idx/20)*5+idx%5][-5])
                    d.append(data[int(idx/20)*5+idx%5][-4])
                    aug_data.append(d)
                    idx += 1
        
        # Combine main and augmented data (keep as raw list)
        all_data = data + aug_data
        
        # Store raw data
        self.dataset = all_data
        
        print(f"\n✓ Loaded {len(all_data)} samples")
        print(f"  Main: {len(data)} samples")
        print(f"  Augmented: {len(aug_data)} samples")
        
        return all_data
    
    def _convert_sample(self, item, item_idx: int) -> Dict[str, Any]:
        """
        Convert raw dataset item to standardized format.
        
        Input format from TSV:
            item[0]: prompt_raw (e.g., "In the United States, the capital is <mask>.")
            item[-5]: gold_answers (list, e.g., ['Washington', 'Washington D.C.'])
            item[-4]: answer_candidates (list, e.g., ['Washington', 'New York', ...])
        
        Args:
            item: Raw data list from TSV
            item_idx: Index of the item in the dataset
        
        Returns:
            Standardized sample dictionary
        
        Note: Each TSV file contains data for ALL 5 countries, interleaved.
              Country is determined by item_idx % 5 (matching original GeoMLAMA logic).
        """
        # Map country codes to info
        COUNTRY_MAP = {
            0: {'idx': 0, 'name': 'United States', 'code': 'en'},
            1: {'idx': 1, 'name': 'China', 'code': 'zh'},
            2: {'idx': 2, 'name': 'India', 'code': 'hi'},
            3: {'idx': 3, 'name': 'Iran', 'code': 'fa'},
            4: {'idx': 4, 'name': 'Kenya', 'code': 'sw'},
        }
        
        # Country is determined by position in dataset (i % 5)
        country_idx = item_idx % 5
        country_info = COUNTRY_MAP[country_idx]
        
        # Determine concept_id and is_base_prompt (matching original logic)
        if item_idx < 125:
            concept_id = int(item_idx / 5)
            is_base_prompt = (item_idx % 5 == 0)  # Every 5th sample is base (US)
        else:
            concept_id = int((item_idx - 125) / 20)
            is_base_prompt = ((item_idx - 125) % 20 == 0)  # Every 20th augmented is base
        
        return {
            'prompt_raw': item[0],
            'gold_answers': item[-5] if len(item) > 3 else item[-2],
            'answer_candidates': item[-4] if len(item) > 3 else item[-1],
            'sample_idx': item_idx,
            'country_idx': country_idx,
            'country_name': country_info['name'],
            'country_code': country_info['code'],
            'concept_id': concept_id,
            'is_base_prompt': is_base_prompt
        }

    def _convert_sample(self, item, item_idx: int) -> Dict[str, Any]:
        """
        Convert raw dataset item to standardized format.
        
        Input format from TSV:
            item[0]: prompt_raw (e.g., "In the United States, the capital is <mask>.")
            item[-5]: gold_answers (list, e.g., ['Washington', 'Washington D.C.'])
            item[-4]: answer_candidates (list, e.g., ['Washington', 'New York', ...])
        
        Args:
            item: Raw data list from TSV
            item_idx: Index of the item in the dataset
        
        Returns:
            Standardized sample dictionary
        
        Note: Each TSV file contains data for ALL 5 countries, interleaved.
              Country is determined by item_idx % 5 (matching original GeoMLAMA logic).
        """
        # Map country codes to info
        COUNTRY_MAP = {
            0: {'idx': 0, 'name': 'United States', 'code': 'en'},
            1: {'idx': 1, 'name': 'China', 'code': 'zh'},
            2: {'idx': 2, 'name': 'India', 'code': 'hi'},
            3: {'idx': 3, 'name': 'Iran', 'code': 'fa'},
            4: {'idx': 4, 'name': 'Kenya', 'code': 'sw'},
        }
        
        # Country is determined by position in dataset (i % 5)
        country_idx = item_idx % 5
        country_info = COUNTRY_MAP[country_idx]
        
        # Determine concept_id and is_base_prompt (matching original logic)
        if item_idx < 125:
            concept_id = int(item_idx / 5)
            is_base_prompt = (item_idx % 5 == 0)  # Every 5th sample is base (US)
        else:
            concept_id = int((item_idx - 125) / 20)
            is_base_prompt = ((item_idx - 125) % 20 == 0)  # Every 20th augmented is base
        
        return {
            'prompt_raw': item[0],
            'gold_answers': item[-5] if len(item) > 3 else item[-2],
            'answer_candidates': item[-4] if len(item) > 3 else item[-1],
            'sample_idx': item_idx,
            'country_idx': country_idx,
            'country_name': country_info['name'],
            'country_code': country_info['code'],
            'concept_id': concept_id,
            'is_base_prompt': is_base_prompt
        }

    def remove_country_from_prompt(self, prompt: str) -> str:
        """
        Remove country-specific information from prompt based on language.
        
        Note: Base prompts (i%5==0) always reference the United States.
        This method removes US-specific text in the given language.
        """
        prompt_wo_country = prompt
        
        if self.language == "en":
            # Remove "United States" / "American" from English prompts
            prompt_wo_country = prompt_wo_country.lower()
            replacements = [
                ("the united states", "united states"),
                ("in united states, ", ""),
                ("of united states", ""),
                ("american", ""),
                ("in united states", ""),
            ]
            for old, new in replacements:
                prompt_wo_country = prompt_wo_country.replace(old, new)
            prompt_wo_country = prompt_wo_country.replace("  ", " ").strip()
            if prompt_wo_country and prompt_wo_country[0] == ' ':
                prompt_wo_country = prompt_wo_country[1:]
        
        elif self.language == "zh":
            # Remove "美国" (United States) from Chinese prompts
            replacements = [
                ("美国的", "美国"),
                ("美国人的", ""),
                ("在美国，", ""),
                ("在美国", ""),
                ("美国", "")
            ]
            for old, new in replacements:
                prompt_wo_country = prompt_wo_country.replace(old, new)
        
        elif self.language == "fa":
            # Remove "ایالت متحده آمریکا" (United States) from Persian prompts
            replacements = [
                ("در ایالت متحده آمریکا, ", ""),
                ("در ایالت متحده آمریکا", ""),
                ("ایالت متحده آمریکا", ""),
                ("از ایالت متحده آمریکا", "")
            ]
            for old, new in replacements:
                prompt_wo_country = prompt_wo_country.replace(old, new)
        
        elif self.language == "hi":
            # Remove "अमेरिका" (America) from Hindi prompts
            replacements = [
                ("अमेरिका में", ""),
                ("अमेरिकी", ""),
                ("अमेरिका ", ""),
                ("अमेरिका के", ""),
                ("अमेरिका का", "")
            ]
            for old, new in replacements:
                prompt_wo_country = prompt_wo_country.replace(old, new)
        
        elif self.language == "sw":
            # Remove "Marekani" (America) from Swahili prompts
            replacements = [
                ("Marekani", ""),
                ("Wamarekani", "Watu"),
                ("Kimarekani", ""),
                ("wamarekani", "watu")
            ]
            for old, new in replacements:
                prompt_wo_country = prompt_wo_country.replace(old, new)
        
        return prompt_wo_country

    def prepare_prompts(self, sample: Dict) -> str:
        """
        Convert GeoMLAMA masked prompt to open-ended question for causal LLMs.
        
        Original: "In the United States, the capital is <mask>."
        Converted: "Question: In the United States, what is the capital? The answer is:"
        
        If with_country=False, removes country context before conversion.
        """
        prompt_raw = sample['prompt_raw']
        
        # Remove country context if requested (only for base prompts)
        if not self.with_country and sample['is_base_prompt']:
            prompt_raw = self.remove_country_from_prompt(prompt_raw)
        
        # Remove <mask> and convert to question
        prompt_clean = prompt_raw
        
        # Convert statement to question format
        if ' is ' in prompt_clean:
            parts = prompt_clean.split(' is ')
            if len(parts) == 2:
                subject = parts[0].strip()
                prompt_clean = f"what is {subject}"
        
        # Format as question with answer prompt
        final_prompt = f"{prompt_clean}"
        
        return final_prompt

    def evaluate_response(self, prediction: str, sample: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate model prediction against gold answers.
        
        Uses two metrics:
        1. Exact match: prediction exactly matches one of the gold answers
        2. Overlap: any gold answer is contained in the prediction
        
        Args:
            prediction: Model's generated answer
            sample: Sample dict with 'gold_answers' field
            
        Returns:
            Dict with 'exact_match' and 'overlap' scores (0.0 or 1.0)
        """
        gold_answers = sample['gold_answers']
        if not isinstance(gold_answers, list):
            gold_answers = [gold_answers]
        
        # Extract actual answer from prediction
        answer = self._extract_answer(prediction)
        
        # Normalize to uppercase for comparison
        answer_upper = answer.upper()
        gold_answers_upper = [ans.upper() for ans in gold_answers]
        
        # Metric 1: Exact match
        exact_match = 1.0 if answer_upper in gold_answers_upper else 0.0
        
        # Metric 2: Overlap (substring match)
        overlap = 1.0 if any(gold.upper() in answer_upper for gold in gold_answers) else 0.0
        
        return {
            'exact_match': exact_match,
            'overlap': overlap,
        }

    def _extract_answer(self, prediction: str) -> str:
        """
        Extract the actual answer from model's prediction.
        
        Handles formats like:
        - "Washington"
        - "Answer: Washington"
        - "The answer is: Washington"
        - Multi-line responses (takes first line after "answer is:")
        """
        if not prediction:
            return ""
        
        prediction = prediction.strip()
        
        # Try to find answer after common patterns
        patterns = [
            r'The answer is:\s*(.+)',
            r'answer is:\s*(.+)',
            r'Answer:\s*(.+)',
            r'answer:\s*(.+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prediction, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                # Take only first line and remove trailing period
                answer = answer.split('\n')[0].split('.')[0].strip()
                return answer
        
        # If no pattern matches, take the first line
        lines = prediction.split('\n')
        first_line = lines[0].strip()
        
        # Remove trailing periods
        first_line = first_line.rstrip('.')
        
        return first_line

    def get_samples(self, split: str = 'test', n_samples: Optional[int] = None) -> List:
        """
        Get dataset samples.
        
        Note: Returns raw data. Use _convert_sample() to get standardized format.
        """
        if self.dataset is None:
            self.load_dataset()
        
        samples = self.dataset
        
        return samples[:n_samples] if n_samples else samples


def create_geomlama_task(dataset_path: Path, config: Dict[str, Any]) -> GeoMLAMATask:
    """Factory function to create GeoMLAMA task."""
    return GeoMLAMATask("GeoMLAMA", dataset_path, config)

if __name__ == "__main__":
    print("\n" + "="*70)
    print("GeoMLAMA Task - Testing (All Countries in ENGLISH)")
    print("="*70)
    
    # Test configuration - ENGLISH prompts about all 5 countries
    config = {
        "language": "en",  # ← Change to "en" for English prompts!
        "with_country": True,
    }
    
    task = create_geomlama_task(
        Path("/home/mila/r/ramesana/projects/vllm-demo/repos/GeoMLAMA"), 
        config
    )
    
    print(f"\nLanguage file: gd_prompt_{config['language']}.tsv")
    print(f"(English prompts about all 5 countries)")
    task.load_dataset()

    if len(task.dataset) > 0:
        # Test samples from each country (0-4 are the first 5, one per country)
        print(f"\n{'='*70}")
        print("Samples from Each Country (indices 0-4):")
        print(f"{'='*70}")
        
        for idx in range(min(5, len(task.dataset))):
            raw_item = task.dataset[idx]
            sample = task._convert_sample(raw_item, item_idx=idx)
            
            print(f"\nSample {idx}:")
            print(f"  Country: {sample['country_name']} ({sample['country_code']})")
            print(f"  Is Base Prompt: {sample['is_base_prompt']}")
            print(f"  Concept ID: {sample['concept_id']}")
            print(f"  Gold Answers: {sample['gold_answers']}")
            print(f"  Prompt: {sample['prompt_raw']}")
            
            formatted = task.prepare_prompts(sample)
            print(f"  Formatted: {formatted}")
        
        # Show next set to demonstrate the pattern continues
        print(f"\n{'='*70}")
        print("Next 5 Samples (indices 5-9, same countries again):")
        print(f"{'='*70}")
        
        for idx in range(5, min(10, len(task.dataset))):
            raw_item = task.dataset[idx]
            sample = task._convert_sample(raw_item, item_idx=idx)
            print(f"Sample {idx}: {sample['country_name']} - {sample['prompt_raw'][:80]}...")
        
        # Test evaluation
        print(f"\n{'='*70}")
        print("Evaluation Test:")
        print(f"{'='*70}")
        
        sample = task._convert_sample(task.dataset[0], item_idx=0)
        print(f"\nTest Sample (Country: {sample['country_name']}):")
        print(f"  Prompt: {sample['prompt_raw']}")
        print(f"  Gold: {sample['gold_answers']}")
        
        test_predictions = [
            sample['gold_answers'][0] if sample['gold_answers'] else "test",
            f"The answer is: {sample['gold_answers'][0]}" if sample['gold_answers'] else "test",
            "wrong answer",
        ]
        
        for pred in test_predictions:
            result = task.evaluate_response(pred, sample)
            print(f"\nPrediction: '{pred}'")
            print(f"  Exact Match: {result['exact_match']:.0%}")
            print(f"  Overlap: {result['overlap']:.0%}")
        
        # Summary statistics
        print(f"\n{'='*70}")
        print("Dataset Summary:")
        print(f"{'='*70}")
        print(f"Language: {config['language']} (English)")
        print(f"Total Samples: {len(task.dataset)}")
        
        # Count samples per country
        country_counts = {i: 0 for i in range(5)}
        for idx in range(len(task.dataset)):
            country_idx = idx % 5
            country_counts[country_idx] += 1
        
        print(f"\nSamples per country (in English):")
        for i in range(5):
            country_name = ['United States', 'China', 'India', 'Iran', 'Kenya'][i]
            print(f"  {country_name}: {country_counts[i]} samples")
        
        # Highlight Iran count
        iran_count = country_counts[3]
        print(f"\n🎯 Iran-specific samples: {iran_count}")
        print(f"   (These are at indices: 3, 8, 13, 18, 23, ...)")
        
        print(f"\n{'='*70}\n")
                
        prompt = task.prepare_prompts(sample)
        print(f"  Formatted Prompt: {prompt}")
        
        # Test evaluation
        print(f"\n{'='*70}")
        print("Evaluation Test:")
        print(f"{'='*70}")
        
        sample = task._convert_sample(task.dataset[0], item_idx=0)
        print(f"\nTesting with sample: {sample['prompt_raw']}")
        print(f"Gold answers: {sample['gold_answers']}")
        
        test_predictions = [
            sample['gold_answers'][0] if sample['gold_answers'] else "Tehran",
            "Tehran",
            "تهران",  # Tehran in Persian
            "The answer is: " + (sample['gold_answers'][0] if sample['gold_answers'] else "Tehran"),
        ]
        
        for pred in test_predictions:
            result = task.evaluate_response(pred, sample)
            print(f"\nPrediction: '{pred}'")
            print(f"  Exact Match: {result['exact_match']}")
            print(f"  Overlap: {result['overlap']}")
        
        # Summary
        print(f"\n{'='*70}")
        print("Dataset Summary:")
        print(f"{'='*70}")
        print(f"Country: {task.country_info['name']}")
        print(f"Language: {task.country_info['code']}")
        print(f"Total Samples: {len(task.dataset)}")
        print(f"With Country Context: {task.with_country}")
        print(f"{'='*70}\n")