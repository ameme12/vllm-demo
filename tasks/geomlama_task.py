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
        Convert GeoMLAMA masked prompt to multiple-choice question.
        
        Format:
        [Original sentence with <mask> → <answer>]. Without any explanation, choose only one from the given alphabet choices(e.g., A, B, C).
        Provide as JSON format: {"answer_choice":""}
        
            A. option1
            B. option2
            C. option3
            D. option4
        
        Answer:
        """
        prompt_raw = sample['prompt_raw']
        answer_candidates = sample.get('answer_candidates', [])
        
        # Remove country context if requested (only for base prompts)
        if not self.with_country and sample['is_base_prompt']:
            prompt_raw = self.remove_country_from_prompt(prompt_raw)
        
        # Simply replace <mask> with <answer> - keep the original sentence structure
        question = prompt_raw.replace('<mask>', '<answer>').strip()
        
        # Build the prompt with instructions
        mc_prompt = f"{question} Without any explanation, choose only one from the given alphabet choices(e.g., A, B, C). \n"
        mc_prompt += '        Provide as JSON format: {"answer_choice":""}\n\n'
        
        # Add answer candidates with indentation
        option_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        
        if answer_candidates:
            for i, candidate in enumerate(answer_candidates[:8]):  # Max 8 options
                letter = option_letters[i]
                mc_prompt += f"            {letter}. {candidate}\n"
        else:
            # Fallback: use first gold answer if no candidates
            gold_answers = sample.get('gold_answers', [])
            if gold_answers:
                mc_prompt += f"            A. {gold_answers[0]}\n"
        
        mc_prompt += "\n            Answer:"
        
        return mc_prompt

    def evaluate_response(self, prediction: str, sample: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate model prediction - check if the chosen letter corresponds to a gold answer.
        
        Simplified logic:
        1. Extract the letter choice (A, B, C, D, etc.)
        2. Map letter to the actual answer using answer_candidates
        3. Check if that answer matches any gold answer
        
        Args:
            prediction: Model's generated answer (should contain a letter like "A" or JSON)
            sample: Sample dict with 'gold_answers' and 'answer_candidates'
            
        Returns:
            Dict with 'exact_match' and 'overlap' scores (0.0 or 1.0)
        """
        # Extract the letter choice from prediction
        letter_choice = self._extract_answer(prediction)
        
        # If no letter extracted, it's wrong
        if not letter_choice or letter_choice not in 'ABCDEFGH':
            return {
                'exact_match': 0.0,
                'overlap': 0.0,
            }
        
        # Map letter to actual answer
        answer_candidates = sample.get('answer_candidates', [])
        letter_idx = ord(letter_choice) - ord('A')
        
        # If letter index out of range, it's wrong
        if letter_idx >= len(answer_candidates):
            return {
                'exact_match': 0.0,
                'overlap': 0.0,
            }
        
        # Get the chosen answer
        chosen_answer = answer_candidates[letter_idx]
        
        # Get gold answers
        gold_answers = sample['gold_answers']
        if not isinstance(gold_answers, list):
            gold_answers = [gold_answers]
        
        # Normalize for comparison
        chosen_answer_norm = chosen_answer.upper().strip()
        gold_answers_norm = [ans.upper().strip() for ans in gold_answers]
        
        # Check if chosen answer matches any gold answer
        exact_match = 1.0 if chosen_answer_norm in gold_answers_norm else 0.0
        
        # Overlap: check if any gold answer is in chosen answer or vice versa
        overlap = 0.0
        for gold in gold_answers_norm:
            if gold and (gold in chosen_answer_norm or chosen_answer_norm in gold):
                overlap = 1.0
                break
        
        return {
            'exact_match': exact_match,
            'overlap': overlap,
        }

    def _extract_answer(self, prediction: str) -> str:
        """
        Extract the answer choice from model's prediction.
        
        Handles formats like:
        - {"answer_choice":"A"}
        - {"answer_choice": "A"}
        - A
        - A.
        - answer_choice: A
        - The answer is A
        """
        if not prediction:
            return ""
        
        prediction = prediction.strip()
        
        # Pattern 1: JSON format {"answer_choice":"A"}
        json_match = re.search(r'["\']?answer_choice["\']?\s*:\s*["\']?([A-H])["\']?', prediction, re.IGNORECASE)
        if json_match:
            return json_match.group(1).upper()
        
        # Pattern 2: Just a letter at the start
        letter_match = re.match(r'^([A-H])\.?', prediction.upper().strip())
        if letter_match:
            return letter_match.group(1)
        
        # Pattern 3: "Answer: A" or "The answer is A"
        answer_patterns = [
            r'answer\s*(?:is|:)?\s*([A-H])',
            r'choice\s*(?:is|:)?\s*([A-H])',
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, prediction, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # Pattern 4: Find any standalone letter
        letter_search = re.search(r'\b([A-H])\b', prediction.upper())
        if letter_search:
            return letter_search.group(1)
        
        # If no letter found, return empty
        return ""

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