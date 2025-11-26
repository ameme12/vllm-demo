"""
BLEnD Task Implementation
Dataset: nayeon212/BLEnD on HuggingFace
"""

from pathlib import Path
from typing import Dict, List, Any
from base_task import BaseTask

COUNTRY_NAME = {
        'US': 'US',  # United States
        'UK': 'UK',  # United Kingdom
        'CN': 'China',  # China
        'ES': 'Spain',  # Spain
        'MX': 'Mexico',  # Mexico
        'ID': 'Indonesia',  # Indonesia
        'KR': 'South_Korea',  # South Korea
        'KP': 'North_Korea',  # North Korea
        'GR': 'Greece',  # Greece
        'IR': 'Iran',  # Iran
        'DZ': 'Algeria',  # Algeria
        'AZ': 'Azerbaijan',  # Azerbaijan
        'JB': 'West_Java',  # West Java
        'AS': 'Assam',  # Assam
        'NG': 'Northern_Nigeria',  # Northern Nigeria
        'ET': 'Ethiopia',  # Ethiopia
    }

class BLEnDBaseTask(BaseTask):

    '''
    Common BLEnD condig and helpers share by short-answer and MCQ tasks
    '''

    def __init__(self, name: str, dataset_path: Path, config: Dict[str, Any]):
        super().__init__(name, dataset_path, config)

        self.blend_config = config.get("blend_config", 'short-answer-questions')
        self.culture = config.get("culture", 'KR') #default South Korea
        self.use_english = config.get("use_english", True)

        if self.culture not in COUNTRY_NAME:
            raise ValueError(
                f"This culture is not in this dataset: {self.culture}"
                f"Expected one of: {list(COUNTRY_NAME.keys())}"
            )

    def load_dataset(self):
        pass

    def get_country_name(self) -> str:
        return COUNTRY_NAME[self.culture]

    def _convert_sample(self, item: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def prepare_prompts(self, sample: Dict) -> str:
        pass

    def evaluate_response(self, prediction: str, ground_truth: Any) -> Dict[str, float]:
        pass

class BLEnDShortAnswerTask(BLEnDBaseTask):

    '''
    BLEnD short answer question task
    short-answer-questions: Simple QandA

            columns: ID, Topic, Source, Question, Translation
            
            Cultures included (Split):
            - United States US
            - United Kingdom UK
            - Chine CN
            - Spain ES
            - Mexico MX
            - Indonesia ID
            - South Korea KR
            - North Korea KP
            - Greece GR
            - Iran IR
            - Algeria DZ
            - Azerbaijan AZ
            - West Java JB
            - Assam AS
            - Northern Nigeria NG
            - Ethiopia ET
    '''

    def __init__(self, dataset_path: Path, config: Dict[str, Any]):
        # Force short-answer config
        config = {**config, "blend_config": "short-answer-questions"}
        super().__init__("BLEnD-short-answer", dataset_path, config)


    def load_dataset(self):

        '''Load the BLEnD dataset based on the specified configuration and culture.'''

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install the 'datasets' : uv add datasets")
        
        #loading the dataset for the Short answer for selected country

        print(f"Loading BLEnD dataset with config: {self.blend_config}, culture: {self.culture}, Language: {'English' if self.use_english else 'Local'}")

        ds = load_dataset("nayeon212/BLEnD", "short-answer-questions")

        if self.culture not in ds:
            available = list(ds.keys())
            raise ValueError(f"Culture '{self.culture}' not found in dataset. Available cultures: {available}")

        ds_as = ds[self.culture]
        print(f"  ✓ Loaded {len(ds_as)} short-answer samples for {self.culture}")
        print("  Sample questions:")
        for i in range(min(5, len(ds_as))):
            row = ds_as[i]
            question = row["Translation"] if self.use_english else row["Question"]
            print(f"    [{i}] ID={row['ID']}  ->  {question}")
      

        self.dataset = ds_as

    def _convert_sample(self, item: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def prepare_prompts(self, sample: Dict) -> str:
        pass

    def evaluate_response(self, prediction: str, ground_truth: Any) -> Dict[str, float]:
        pass
        
        #example of question
        # What is a common snack for preschool kids in Assam?

class BLEnDMCQTask(BLEnDBaseTask):

    '''
    Task for BLEnD: Cultural knowledge benchmark
    multiple-choice-questions: MCQ format

        single 'test split with column 'country' to filter by culture

        columns: MCQID, ID, country, prompt, choices, choice_countries, answer_idx
        
    '''

    def __init__(self, dataset_path: Path, config: Dict[str, Any]):
        # Force MCQ config
        config = {**config, "blend_config": "multiple-choice-questions"}
        super().__init__("BLEnD-MCQ", dataset_path, config)

    def load_dataset(self):

        '''Load the BLEnD mcq dataset based on the specified configuration and culture.'''

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install the 'datasets' : uv add datasets")
        
        target_country = self.get_country_name()
        print(
            f"[MCQ] Loading BLEnD MCQs: culture={self.culture} "
            f"({target_country})"
        )

        #loading mcq dataset 

        print(f"Loading BLEnD dataset with config: {self.blend_config}, culture: {self.culture}, Language: {'English' if self.use_english else 'Local'}")

        mcq_ds = load_dataset("nayeon212/BLEnD", "multiple-choice-questions")
        mcq_data = mcq_ds['test'].filter(lambda x: x['country'] == target_country)
        
        print(f"  ✓ Loaded {len(mcq_data)} MCQ samples for country={target_country}")

        if len(mcq_data) == 0:
            print("  ⚠ No MCQ samples found for this country. "
                  "Check COUNTRY_NAME mapping or 'country' column values.")
        else:
            print("  Sample MCQ questions:")
            for i in range(min(3, len(mcq_data))):
                row = mcq_data[i]
                print(f"    [{i}] MCQID={row.get('MCQID', 'N/A')}")
                print(f"         prompt:       {row['prompt']}")
                print(f"         choices:      {row['choices']}")
                print(f"         answer_idx:   {row['answer_idx']}\n")

        self.dataset = mcq_data

    def _convert_sample(self, item: Dict[str, Any]) -> Dict[str, Any]:
        '''
        Convert a dataset item into the standardized format.

        Args:
            item (Dict[str, Any]): Raw item from the dataset
        
        Returns:
            Dict[str, Any]: Dictionary with standardized keys
        '''

        return{
            "mcq_id": item["MCQID"],
            "id": item["ID"],
            "country": item["country"],
            "prompt": item["prompt"],
            "choices": item["choices"],
            "choice_countries": item["choice_countries"],
            "answer_idx": item["answer_idx"],   
        }



    def prepare_prompts(self, sample: Dict) -> str:
        '''
        Prepare the prompt for the MCQ for the LLM.

        Args:
            sample: Converted sample dictionary from _convert_sample

        returns:
            str: Formatted prompt string with multiple choice options


        ex:
        What is a common snack for preschool kids in Algeria? Without any explanation, choose only one from the given alphabet choices(e.g., A, B, C). 
        Provide as JSON format: {"answer_choice":""}

            A. chocolate paste
            B. egg
            C. fruit
            D. jam sandwiches

            Answer:
        
        '''

        full_prompt = sample["prompt"]
        return full_prompt


    def evaluate_response(self, prediction: str, ground_truth: Any) -> Dict[str, float]:
        '''
        Evaluate the model's MCQ response against the ground truth.

        Args:
            prediction: Model's predicted answer choice (e.g., "A", "B", "C", "D") in json format
            ground_truth: Dictionary with answer_idx and other metadata

        Returns:
            Dictionary with metric scores

        '''

        import json

        predicted_idx = -1
        #try to parse as JSON

        try:
            json_response = json.loads(prediction)
            if "answer_choice" in json_response:
                predicted_letter = json_response["answer_choice"]
                predicted_idx = 1
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        if predicted_idx == -1:
            print("LLM response could not be parsed correctly.")
            return {"accuracy": 0.0}

        if isinstance(ground_truth, dict):
            correct_letter = ground_truth["answer_idx"].upper()
        else:
            correct_letter = ground_truth.upper()

        is_correct = (predicted_letter == correct_letter)

        #calculate accuracy
        #should i add anything to this ?

        return {
            "accuracy": 1.0 if is_correct else 0.0,
            "exact_match": 1.0 if is_correct else 0.0
        }
        

    


def create_blend_task(dataset_path: Path, config: Dict[str, Any]) -> BLEnDBaseTask:
    '''Factory function to create a BLEnD task based on the configuration.'''

    blend_config = config.get("blend_config", "short-answer-questions")

    if blend_config == "short-answer-questions":
        return BLEnDShortAnswerTask(dataset_path, config)
    elif blend_config == "multiple-choice-questions":
        return BLEnDMCQTask(dataset_path, config)
    else:
        raise ValueError(
            f"Unknown BLEnD config: {blend_config}. "
            "Expected 'short-answer-questions' or 'multiple-choice-questions'."
        )

if __name__ == "__main__":

    import json
    # Example config: change these to test different setups
    config = {
        # "blend_config": "short-answer-questions",
        "blend_config": "multiple-choice-questions",
        "culture": "DZ",          # e.g. "US", "KR", "AS", ...
        "use_english": True,
    }

    task = create_blend_task(dataset_path=Path("."), config=config)
    task.load_dataset()

    if len(task.dataset) > 0:
        sample_item = task.dataset[0]
        converted = task._convert_sample(sample_item)
        prompt = task.prepare_prompts(converted)
        print("\n" + "="*60)
        print("EXAMPLE USAGE:")
        print("="*60)
        print("\nConverted Sample:")
        print(converted)
        print("\nGenerated Prompt:")
        print(prompt)
        
        # For MCQ, show evaluation example
        if isinstance(task, BLEnDMCQTask):
            # Simulate a correct answer
            answer = task.dataset[0]["answer_idx"]
            print(answer)
            correct_answer = json.dumps({"answer_choice": answer})
            print(correct_answer)
            predicted_letter = json.loads(correct_answer)["answer_choice"]
            print(predicted_letter)
            
            metrics = task.evaluate_response(correct_answer, converted)
            print("\nEvaluation Metrics (correct answer):")
            print(metrics)
            
            # Simulate an incorrect answer
            
            answer_2 = task.dataset[1]["answer_idx"]
            wrong_answer = json.dumps({"answer_choice": answer_2})
            predicted_letter = json.loads(wrong_answer)["answer_choice"]
            print(predicted_letter)
            
            metrics = task.evaluate_response(wrong_answer, converted)
            print("\nEvaluation Metrics (wrong answer):")
            print(metrics)
            