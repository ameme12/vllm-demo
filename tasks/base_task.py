from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pathlib import Path


class BaseTask(ABC):
    '''Base class for all evaluation tasks.'''

    def __init__(self, task_name: str, dataset_path: Path, config: Dict[str, Any]):
        self.task_name = task_name
        self.dataset_path = dataset_path
        self.config = config
        self.dataset = None
        

    @abstractmethod
    def load_dataset(self):
        '''Load the dataset from the specified path.'''
        pass

    def prepare_prompts(self, sample: Dict) -> str:
        '''Prepare the prompt for a given sample.'''
        pass

    @abstractmethod
    def evaluate_response(self, prediction: str, ground_truth: Any) -> Dict[str, float]:
        '''Evaluate the model's response against the ground truth.'''
        pass

    def get_samples(self, split: str='test', n_samples: Optional[int] = None) -> List[Dict]:

        '''Get dataset samples'''

        if self.dataset is None:
            self.load_dataset()
        samples = self.dataset[split]

        return samples[:n_samples] if n_samples else samples

    def prepare_batch_prompts(self, samples: List[Dict]) -> List[str]:
        '''Prepare prompts for a batch of samples.'''
        return [self.prepare_prompts(sample) for sample in samples]

    
