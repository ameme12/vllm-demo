import json
from pathlib import Path
from typing import Dict, List, Any


class CARROTTask(BaseTask):
    '''Task implementation for the CARROT dataset.'''


    def __init__(self, dataset_path: Path, config: Dict[str, Any]):
        super().__init__('CARROT', dataset_path, config)

    def load_dataset(self):
        '''Load the CARROT dataset from repository.'''

        data_file = self.dataset_path / 'data' / 'carrot_dataset.json'

        if data_file.exists():
            with open(data_file, 'r') as f:
                self.dataset = {'test': json.load(f)}      
        else:
            raise FileNotFoundError(f"Dataset file not found at {data_file}")   

    def prepare_prompts(self, sample: Dict) -> str:
        '''Prepare the prompt for a given CARROT sample.'''
    
        prompt_template = self.config.get('prompt_template',
                                         "Question: {question}\nAnswer:")

        return prompt_template.format(**sample)

    def evaluate_response(self, prediction: str, ground_truth: Any) -> Dict[str, float]:
        '''Evaluate CARROT task response against the ground truth.'''

        exact_match = float(prediction.strip().lower() == str(ground_truth).strip().lower())
        return {
            'exact_match': exact_match,
            'prediction': prediction,
            'ground_truth': ground_truth
        }
        