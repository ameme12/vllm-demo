from pathlib import Path
from typing import Dict, List, Any
from base_task import BaseTask
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
            print(f"Filter by Country: {self.country} ({self.get_region()})")
        elif self.region:
            print(f"Filter by Region: {self.region}")
        else:
            print(f"Loading all countries")


        ds = load_dataset("kellycyy/CulturalBench", "CulturalBench-Hard")
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
                        print(f"  [{marker}] {opt['prompt_option']}")
                    
                    if len(questions_shown) >= 3:
                        break

        print(f"\n{'='*60}\n")
        self.dataset = ds_split


    def get_region(self) -> str:
        return COUNTRY_REGIONS[self.country]

    def _convert_sample(self, item: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def prepare_prompts(self, sample: Dict) -> str:
        pass

    def evaluate_response(self, prediction: str, ground_truth: Any) -> Dict[str, float]:
        pass

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
    print("CulturalBench MCQ Task - Example Usage")
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
        
        # Get first question (4 options grouped by question_idx)
        question_idx = task.dataset[0]['question_idx']
        question_items = [
            item for item in task.dataset 
            if item['question_idx'] == question_idx
        ]
        
        print(f"\nProcessing question {question_idx} with {len(question_items)} options")
        