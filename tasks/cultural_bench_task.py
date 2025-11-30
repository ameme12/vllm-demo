from pathlib import Path
from typing import Dict, List, Any
from tasks.base_task import BaseTask
import json
import re  

COUNTRY_REGIONS = {
    # North America
    'Canada': 'North America',
    'United States': 'North America',
    
    # South America
    'Argentina': 'South America',
    'Brazil': 'South America',
    'Chile': 'South America',
    'Mexico': 'South America',
    'Peru': 'South America',
    
    # East Europe
    'Czech Republic': 'East Europe',
    'Poland': 'East Europe',
    'Romania': 'East Europe',
    'Ukraine': 'East Europe',
    'Russia': 'East Europe',
    
    # South Europe
    'Spain': 'South Europe',
    'Italy': 'South Europe',
    
    # West Europe
    'France': 'West Europe',
    'Germany': 'West Europe',
    'Netherlands': 'West Europe',
    'United Kingdom': 'West Europe',
    
    # Africa
    'Egypt': 'Africa',
    'Morocco': 'Africa',
    'Nigeria': 'Africa',
    'South Africa': 'Africa',
    'Zimbabwe': 'Africa',
    
    # Middle East/West Asia
    'Iran': 'Middle East/West Asia',
    'Israel': 'Middle East/West Asia',
    'Lebanon': 'Middle East/West Asia',
    'Saudi Arabia': 'Middle East/West Asia',
    'Turkey': 'Middle East/West Asia',
    
    # South Asia
    'Bangladesh': 'South Asia',
    'India': 'South Asia',
    'Nepal': 'South Asia',
    'Pakistan': 'South Asia',
    
    # Southeast Asia
    'Indonesia': 'Southeast Asia',
    'Malaysia': 'Southeast Asia',
    'Philippines': 'Southeast Asia',
    'Singapore': 'Southeast Asia',
    'Thailand': 'Southeast Asia',
    'Vietnam': 'Southeast Asia',
    
    # East Asia
    'China': 'East Asia',
    'Hong Kong': 'East Asia',
    'Japan': 'East Asia',
    'South Korea': 'East Asia',
    'Taiwan': 'East Asia',
    
    # Oceania
    'Australia': 'Oceania',
    'New Zealand': 'Oceania',
}

class CulturalBenchTask(BaseTask):

    '''
    CulturalBench Task for evaluating cultural knowledge across different countries.
    shared by easy and hard task

    from datasets import load_dataset

    ds = load_dataset("kellycyy/CulturalBench", "CulturalBench-Hard")


 
    CulturalBench Easy task: Multiple-choice questions (MCQ)
    
    Format:
        - 1,227 questions total
        - Each question has 4 options (A, B, C, D)
        - Output: one of A, B, C, D
        - Evaluation: accuracy at question level (per question_idx)
    
    Columns: data_idx, question_idx, prompt_question, prompt_option, answer, country
    
    Example:
        In the Netherlands, which of the following is an unusual common public practice?
        A. Cycle everywhere
        B. Using deodorant
        C. Tipping generously (CORRECT)
        D. Talking loudly on the phone

    Answer:

    '''

    def __init__(self, name: str, dataset_path: Path, config: Dict[str, Any]):
        
        super().__init__("CulturalBench-Easy", dataset_path, config)
        
        self.country = config.get("country", None) #None means all countries
        self.region = config.get("region", None) #None means all regions

        # Validate configuration

        if self.country and self.country not in COUNTRY_REGIONS:
            raise ValueError(
                f"Country '{self.country}' not found in dataset. "
                f"Available countries: {list(COUNTRY_REGIONS.keys())}"
            )

        if self.region and self.region not in set(COUNTRY_REGIONS.values()):
            raise ValueError(
                f"Region '{self.region}' not found. "
                f"Available regions: {list(set(COUNTRY_REGIONS.values()))}"
            )

    def load_dataset(self) -> List[Dict[str, Any]]:
        '''
        load the cultural bench easy dataset

        '''
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install the 'datasets': uv add datasets")

        print(f"\n{'='*60}")
        print(f"Loading CulturalBench Easy Dataset")
        print(f"{'='*60}")
        print(f"Config: {self.config}")
        if self.country:
            print(f"Filter by Country: {self.country} ({self.get_region(self.country)})")
        elif self.region:
            print(f"Filter by Region: {self.region}")
        else:
            print(f"Loading all countries")


        ds = load_dataset("kellycyy/CulturalBench", "CulturalBench-Easy")
        ds_split = ds['test']

        if self.country:
            ds_split = ds_split.filter(lambda x: x['country'] == self.country)
        elif self.region:
            countries_in_region = [country for country, region in COUNTRY_REGIONS.items() if region == self.region]
            ds_split = ds_split.filter(lambda x: x['country'] in countries_in_region)

        print(f"\n✓ Loaded {len(ds_split)} Easy MCQ samples")

        if len(ds_split) == 0:
            print("⚠ No samples found with current filters.")
        else:
            # Count unique questions
            unique_questions = len(set([item['question_idx'] for item in ds_split]))
            print(f"  Total unique questions: {unique_questions}")
            
            # Show sample questions
            print("\nSample Questions:")
            print("-" * 60)
            
            # Group by question_idx to show complete questions
            questions_shown = set()
            for item in ds_split:
                if item['question_idx'] not in questions_shown:
                    questions_shown.add(item['question_idx'])
                    print(f"\n[Q{item['question_idx']}] {item['prompt_question']}")
                    
                    # Get all options for this question
                    options = [x for x in ds_split if x['question_idx'] == item['question_idx']]
                    for opt in options:
                        marker = "✓" if opt['answer'] else " "
                        print(f"  [{marker}] {opt['prompt_option_a']}")
                        print(f"  [{marker}] {opt['prompt_option_b']}")
                        print(f"  [{marker}] {opt['prompt_option_c']}")
                        print(f"  [{marker}] {opt['prompt_option_d']}")
                    
                    if len(questions_shown) >= 3:
                        break

        print(f"\n{'='*60}\n")
        self.dataset = ds_split


    def get_region(self, country:str) -> str:
        return COUNTRY_REGIONS.get(country, "Unknown")

    def _convert_sample(self, item: Dict[str, Any]) -> Dict[str, Any]:
        '''
        Convert a dataset item into standardized format

        grouped  by question_idx

        args:

            item : raw item from the dataset

        reutrn:
            Dict with stadardized keys

        '''

        return {
            "data_idx": item["data_idx"],
            "question_idx": item["question_idx"],
            "prompt_question": item["prompt_question"],
            "prompt_option_a": item["prompt_option_a"],
            "prompt_option_b": item["prompt_option_b"],
            "prompt_option_c": item["prompt_option_c"],
            "prompt_option_d": item["prompt_option_d"],
            "answer": item["answer"],
            "country": item["country"],
            "region": self.get_region(item["country"]),
        }

    def prepare_prompts(self, sample: Dict) -> str:
        '''
        Prepare MCQ prompt for the LLM

        Args:
            sample: Converted sample dictionary with all 4 options

        Returns:
            Formatted prompt string with multiple choice options

         Example:
            In the Netherlands, which of the following is an unusual common public practice?
            Without any explanation, choose only one from the given alphabet choices (A, B, C, D).
            Provide as JSON format: {"answer_choice":""}
            
            A. Cycle everywhere
            B. Using deodorant
            C. Tipping generously
            D. Talking loudly on the phone
            
            Answer:
        '''
        question = sample["prompt_question"]
        
        # Format the prompt
        prompt = f"{question}\n"
        prompt += "Without any explanation, choose only one from the given alphabet choices (A, B, C, D).\n"
        prompt += 'Provide as JSON format: {"answer_choice":""}\n\n'
        prompt += f"A. {sample['prompt_option_a']}\n"
        prompt += f"B. {sample['prompt_option_b']}\n"
        prompt += f"C. {sample['prompt_option_c']}\n"
        prompt += f"D. {sample['prompt_option_d']}\n"
        prompt += "\nAnswer:"
        
        return prompt


    def evaluate_response(self, prediction: str, ground_truth: Any) -> Dict[str, float]:
        """
        Evaluate the model's MCQ response against the ground truth.
        
        Args:
            prediction: Model's predicted answer choice (e.g., "A", "B", "C", "D") in JSON format
            ground_truth: Dictionary with answer field containing the correct letter
        
        Returns:
            Dictionary with metric scores
        """
        predicted_letter = None
        
        # Method 1: Try JSON parsing first
        try:
            json_str = prediction.strip()
            
            # Search for JSON object with answer_choice
            json_match = re.search(
                r'\{[^}]*"answer_choice"\s*:\s*"([A-D])"[^}]*\}',
                json_str,
                re.IGNORECASE
            )
            
            if json_match:
                predicted_letter = json_match.group(1).upper()
            else:
                # Try parsing the whole response as JSON
                json_response = json.loads(json_str)
                if "answer_choice" in json_response:
                    answer = json_response["answer_choice"]
                    if len(answer) == 1 and answer.upper() in ['A', 'B', 'C', 'D']:
                        predicted_letter = answer.upper()
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        # Method 2: Look for standalone letter at the start
        if not predicted_letter:
            first_line = prediction.strip().split('\n')[0].strip()
            if len(first_line) == 1 and first_line.upper() in ['A', 'B', 'C', 'D']:
                predicted_letter = first_line.upper()
        
        # Method 3: Search for any A/B/C/D in the text
        if not predicted_letter:
            letter_match = re.search(r'\b([A-D])\b', prediction.upper())
            if letter_match:
                predicted_letter = letter_match.group(1)

        # If no valid prediction found
        if not predicted_letter:
            return {
                "accuracy": 0.0,
                "exact_match": 0.0,
                "has_valid_format": 0.0,
            }

        # Get ground truth answer
        correct_letter = ground_truth["answer"].strip().upper()
        
        # Check if prediction is correct
        is_correct = (predicted_letter == correct_letter)

        return {
            "accuracy": 1.0 if is_correct else 0.0,
            "exact_match": 1.0 if is_correct else 0.0,
            "has_valid_format": 1.0,
        }


def create_culturalbench_mcq_task(
    dataset_path: Path, 
    config: Dict[str, Any]
) -> CulturalBenchTask:
    """
    Factory function to create a CulturalBench MCQ task.
    
    Args:
        dataset_path: Path to dataset directory
        config: Configuration dictionary with keys:
            - country: Optional country filter
            - region: Optional region filter
    
    Returns:
        Initialized CulturalBench MCQ task instance
    """
    return CulturalBenchTask("CulturalBench-Easy", dataset_path, config)

if __name__ == "__main__":
    # Example usage and testing
    print("\n" + "="*70)
    print("CulturalBench Easy Task - Example Usage")
    print("="*70)
    
    # Configuration
    config = {
        "country": "Japan",  # Change to test different countries
        # "region": "East Asia",  # Or filter by region instead
    }
    
    # Create and load task
    task = create_culturalbench_mcq_task(dataset_path=Path("."), config=config)
    task.load_dataset()
    
    if len(task.dataset) > 0:
        print("\n" + "="*70)
        print("EXAMPLE WORKFLOW")
        print("="*70)
        
        # Get first question
        item = task.dataset[0]
        
        # Convert sample
        converted_sample = task._convert_sample(item)
        
        print("\n1. Converted Sample:")
        print("-" * 70)
        print(f"Question: {converted_sample['prompt_question']}")
        print(f"A. {converted_sample['prompt_option_a']}")
        print(f"B. {converted_sample['prompt_option_b']}")
        print(f"C. {converted_sample['prompt_option_c']}")
        print(f"D. {converted_sample['prompt_option_d']}")
        print(f"Correct Answer: {converted_sample['answer']}")
        print(f"Country: {converted_sample['country']}")
        print(f"Region: {converted_sample['region']}")
        
        # Generate prompt
        prompt = task.prepare_prompts(converted_sample)
        
        print("\n2. Generated Prompt:")
        print("-" * 70)
        print(prompt)
        
        # Test evaluation
        print("\n3. Evaluation Examples:")
        print("-" * 70)
        
        # Test correct answer
        correct_response = json.dumps({"answer_choice": converted_sample['answer']})
        print(f"\n✓ Correct Response: {correct_response}")
        metrics = task.evaluate_response(correct_response, converted_sample)
        print(f"  Metrics: {metrics}")
        
        # Test wrong answer
        all_letters = ['A', 'B', 'C', 'D']
        all_letters.remove(converted_sample['answer'])
        wrong_letter = all_letters[0]
        wrong_response = json.dumps({"answer_choice": wrong_letter})
        print(f"\n✗ Wrong Response: {wrong_response}")
        metrics = task.evaluate_response(wrong_response, converted_sample)
        print(f"  Metrics: {metrics}")
        
        print("\n" + "="*70)
        print("Example completed successfully!")
        print("="*70 + "\n")
    else:
        print("\n⚠ No samples available for testing.")