from pathlib import Path
from typing import Dict, List, Any
from tasks.base_task import BaseTask
import json
import re 

# Predicate templates from DLAMA-v1
PREDICATE_TEMPLATES = {
    'P17': '[X] is located in [Y] .',
    'P19': '[X] was born in [Y] .',
    'P20': '[X] died in [Y] .',
    'P27': '[X] is [Y] citizen .',
    'P30': '[X] is located in [Y] .',
    'P36': 'The capital of [X] is [Y] .',
    'P37': 'The official language of [X] is [Y] .',
    'P47': '[X] shares border with [Y] .',
    'P103': 'The native language of [X] is [Y] .',
    'P106': '[X] is a [Y] by profession .',
    'P136': '[X] plays [Y] music .',
    'P190': '[X] and [Y] are twin cities .',
    'P264': '[X] is represented by music label [Y] .',
    'P364': 'The original language of [X] is [Y] .',
    'P449': '[X] was originally aired on [Y] .',
    'P495': '[X] was created in [Y] .',
    'P530': '[X] maintains diplomatic relations with [Y] .',
    'P1303': '[X] plays [Y] .',
    'P1376': '[X] is the capital of [Y] .',
    'P1412': '[X] used to communicate in [Y] .',
}

PREDICATE_DESCRIPTIONS = {
    'P17': 'Country', 'P19': 'Place of birth', 'P20': 'Place of death',
    'P27': 'Country of citizenship', 'P30': 'Continent', 'P36': 'Capital',
    'P37': 'Official language', 'P47': 'Shares border with', 'P103': 'Native language',
    'P106': 'Occupation', 'P136': 'Genre', 'P190': 'Sister city',
    'P264': 'Record label', 'P364': 'Original language of work', 'P449': 'Original network',
    'P495': 'Country of origin', 'P530': 'Diplomatic relation', 'P1303': 'Instrument',
    'P1376': 'Capital of', 'P1412': 'Languages spoken or published',
}

# Cultural regions as they appear in the dataset
CULTURAL_REGIONS = ['Arab-West', 'Asia-West', 'South America-West']

# Map HuggingFace config names to cultural regions
CONFIG_TO_CULTURES = {
    'Arab-West': ['Arab', 'Western'],
    'Asia-West': ['Asian', 'Western'],
    'South America-West': ['South_American', 'Western'],
}


WIKIDATA_COUNTRIES = {
    # Western Countries
    'Q228': 'Andorra',
    'Q408': 'Australia',
    'Q40': 'Austria',
    'Q31': 'Belgium',
    'Q16': 'Canada',
    'Q142': 'France',
    'Q183': 'Germany',
    'Q27': 'Ireland',
    'Q38': 'Italy',
    'Q347': 'Liechtenstein',
    'Q32': 'Luxembourg',
    'Q235': 'Monaco',
    'Q55': 'Netherlands',
    'Q664': 'New Zealand',
    'Q45': 'Portugal',
    'Q238': 'San Marino',
    'Q29': 'Spain',
    'Q39': 'Switzerland',
    'Q145': 'UK',
    'Q30': 'USA',
    
    # Asian Countries
    'Q148': 'China',
    'Q17': 'Japan',
    'Q423': 'North Korea',
    'Q884': 'South Korea',
    'Q865': 'Taiwan',
    'Q711': 'Mongolia',
    'Q252': 'Indonesia',
    'Q833': 'Malaysia',
    'Q869': 'Thailand',
    'Q881': 'Vietnam',
    'Q928': 'Philippines',
    'Q836': 'Myanmar',
    'Q334': 'Singapore',
    'Q8646': 'Hong Kong',
    'Q14773': 'Macau',
    
    # South American Countries
    'Q414': 'Argentina',
    'Q750': 'Bolivia',
    'Q298': 'Chile',
    'Q739': 'Colombia',
    'Q736': 'Ecuador',
    'Q733': 'Paraguay',
    'Q419': 'Peru',
    'Q77': 'Uruguay',
    'Q717': 'Venezuela',
    
    # African Countries
    'Q1033': 'Nigeria',
    'Q1009': 'Cameroon',
}



class DLAMATask(BaseTask):
    '''
    DLAMA-v1 Task - Simplified version for LLama 3B and Qwen 2.5B evaluation.

        @inproceedings{keleg-magdy-2023-dlama,
        title = "{DLAMA}: A Framework for Curating Culturally Diverse Facts for Probing the Knowledge of Pretrained Language Models",
        author = "Keleg, Amr  and
        Magdy, Walid",
        editor = "Rogers, Anna  and
        Boyd-Graber, Jordan  and
        Okazaki, Naoaki",
        booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
        month = jul,
        year = "2023",
        address = "Toronto, Canada",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2023.findings-acl.389/",
        doi = "10.18653/v1/2023.findings-acl.389",
        pages = "6245--6266",
        abstract = "A few benchmarking datasets have been released to evaluate the factual knowledge of pretrained language models. These benchmarks (e.g., LAMA, and ParaRel) are mainly developed in English and later are translated to form new multilingual versions (e.g., mLAMA, and mParaRel). Results on these multilingual benchmarks suggest that using English prompts to recall the facts from multilingual models usually yields significantly better and more consistent performance than using non-English prompts. Our analysis shows that mLAMA is biased toward facts from Western countries, which might affect the fairness of probing models. We propose a new framework for curating factual triples from Wikidata that are culturally diverse. A new benchmark DLAMA-v1 is built of factual triples from three pairs of contrasting cultures having a total of 78,259 triples from 20 relation predicates. The three pairs comprise facts representing the (Arab and Western), (Asian and Western), and (South American and Western) countries respectively. Having a more balanced benchmark (DLAMA-v1) supports that mBERT performs better on Western facts than non-Western ones, while monolingual Arabic, English, and Korean models tend to perform better on their culturally proximate facts. Moreover, both monolingual and multilingual models tend to make a prediction that is culturally or geographically relevant to the correct label, even if the prediction is wrong."
    }

    # DLAMA-v1 is a culturally diverse factual knowledge benchmark built from Wikidata.
    # It contains ~78k culturally grounded factual triples covering 3 major cultural pairs:
    #   • Arab vs Western
    #   • Asian vs Western
    #   • South American vs Western
    #
    # Each example is structured as a Wikidata triple:
    #       (subject, relation, object)
    # representing a culturally relevant fact (e.g., "Japan – national dish – sushi").
    #
    # The benchmark is used to test whether language models recall culturally specific facts.
    #
    # PROMPT FORMAT:
    # DLAMA uses a cloze-style (fill-in-the-blank) prompt similar to LAMA-style probing.
    # For each triple, a natural-language template is provided with the object removed.
    # Example:
    #     "The national dish of Japan is [MASK]."
    #
    # For open-ended evaluation, the [MASK] is replaced with a direct question instead:
    #     "What is the national dish of Japan?"
    #
    # MODEL EVALUATION:
    # 1. The model generates an answer to the cultural fact.
    # 2. Accuracy is computed by checking if the generated answer matches
    #    or contains the correct object (possibly using an LLM-as-judge).
    #
    # DLAMA-v1 helps evaluate whether LLMs perform differently across cultures,
    # and whether they recall facts from Western vs non-Western regions equally well.

    '''

    def __init__(self, name: str, dataset_path: Path, config: Dict[str, Any]):
        super().__init__("DLAMA-v1", dataset_path, config)

        self.predicate = config.get("predicate", None)
        self.culture = config.get("culture", None)
        self.country = config.get("country", None)


        # Validate
        if self.predicate and self.predicate not in PREDICATE_TEMPLATES:
            raise ValueError(f"Invalid predicate: {self.predicate}")
        if self.culture and self.culture not in CULTURAL_REGIONS:
            raise ValueError(f"Invalid culture: {self.culture}")

        if self.country and self.country not in WIKIDATA_COUNTRIES.values():
            available_countries = sorted(set(WIKIDATA_COUNTRIES.values()))
            raise ValueError(
                f"Invalid country: {self.country}\n"
                f"Available countries: {available_countries}"
            )
        
        self.dataset = None

    def load_dataset(self) -> List[Dict[str, Any]]:
        '''
        Load DLAMA-v1 dataset from HuggingFace
        '''
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install the 'datasets' library to load DLAMA-v1 dataset.")

        print(f"\n{'='*60}")
        print(f"Loading DLAMA-v1 Dataset")
        print(f"{'='*60}")
        print(f"Predicate: {self.predicate or 'All'}")
        print(f"Culture: {self.culture or 'All'}")
        print(f"Country: {self.country or 'All'}")

        # Determine which configuration(s) to load based on culture filter
        # Configurations: "Arab-West", "Asia-West", "South America-West"
        configs_to_load = []

        if self.culture:
            configs_to_load = [self.culture]
        else:
            configs_to_load = ["Arab-West", "Asia-West", "South America-West"]
        
        print(f"Loading configurations: {configs_to_load}")

        # Collect all data from configs
        all_data = []

        for config_name in configs_to_load:
            try:
                print(f"\n  Loading {config_name}...")
                ds = load_dataset("AMR-KELEG/DLAMA-v1", config_name)
                
                # Use the english translations
                ds_split = ds['en']
                
                print(f"    Loaded {len(ds_split)} raw samples")
            
                # Convert to list and filter in Python (easier than HF filter with lists)
                samples = list(ds_split)
                
                # Filter by country if specified
                if self.country:
                    filtered_samples = []
                    for sample in samples:
                        # Get all countries for this sample
                        country_names = self.get_all_country_names(sample['country'])
                        # Include if ANY country matches
                        if self.country in country_names:
                            filtered_samples.append(sample)
                    samples = filtered_samples
                    print(f"    After country filter: {len(samples)} samples")
                
                
                # Add to collection
                all_data.extend(samples)
                
            except Exception as e:
                print(f"    ⚠ Error loading configuration {config_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n✓ Loaded {len(all_data)} samples total")
        
        # Store as list
        self.dataset = all_data
        return all_data

    def get_all_country_names(self, country_field) -> str:
        """
        Extract country name from country field.
        
        Args:
            country_field: Can be a string (Wikidata code) or list of codes
            
        Returns:
            Human-readable country name
        """
        # Handle list format - take first element
        if isinstance(country_field, list):
            country_codes = country_field
        else:
            country_codes = [country_field]
        
        # Map all codes to names
        return [WIKIDATA_COUNTRIES.get(code, "Unknown Country") for code in country_codes]




    def _convert_sample(self, item: Dict[str, Any]) -> Dict[str, Any]:
        
        '''
        Convert raw dataset item to standardized format.
    
        Input format from DLAMA:

        {
            'sub_uri': 'Q134456', 
            'obj_uri': ['Q5287'], 
            'sub_label': 'Yukio Mishima', 
            'obj_label': ['Japanese'], 
            'country': ['Q17'], 
            'size': 738317, 
            'uuid': 'P103_general_ASIA_0'
        }

        Returns standardized sample dict with extracted info.
        '''

        # Extract predicate from UUID (format: P103_general_ASIA_0)
        uuid = item.get('uuid', '')
        predicate = uuid.split('_')[0] if uuid else None

        # Extract culture from UUID (third part: ASIA, WESTERN, ARAB, SOUTH)
        parts = uuid.split('_')
        culture_code = parts[2] if len(parts) > 2 else 'Unknown'

        culture_mapping = {
            'ASIA': 'Asian',
            'WESTERN': 'Western',
            'ARAB': 'Arab',
            'SOUTH': 'South_American'

        }
        culture = culture_mapping.get(culture_code, culture_code) 

        X = item.get('sub_label', '')
        correct_answers = item.get('obj_label', [])

        country_codes = item.get('country', [])
        country_names = self.get_all_country_names(country_codes)

        obj_uri = item.get('obj_uri', [])

        return {
            'subject': X,
            'correct_answer': correct_answers,
            'predicate': predicate,
            'pred_description': PREDICATE_DESCRIPTIONS.get(predicate, predicate),
            'template': PREDICATE_TEMPLATES.get(predicate, '[X] ... [Y]'),
            'culture': culture,            # Asian, Western, Arab, South_American
            'country_codes': country_codes,  # List of Wikidata codes (e.g., ['Q17'])
            'country_names': country_names,  # List of readable names (e.g., ['Japan'])
            'sub_uri': item.get('sub_uri', ''),
            'obj_uri': obj_uri,
            'uuid': uuid,
        }
        

    def prepare_prompts(self, sample: Dict) -> str:
        '''
        Prepare prompt for DLama-v1 sample.

        Converts template like "The native language of [X] is [Y] ."
        into a question format: "The native language of Yukio Mishima is [Answer]?"
        
        Original DLAMA format:
        "Question: What language does [] communicate? The answer is:"

        Args:
            sample: Converted sample dict with 'subject', 'template'
            
        Returns:
            Formatted prompt string asking the model to complete the statement
        '''

        subject = sample['subject']
        predicate = sample['predicate']

        # Replace [X] with subject
        prompt = PREDICATE_TEMPLATES.get(predicate).replace('[X]', subject)

        # Remove [Y] and the period
        prompt = prompt.replace('[Y]', '').replace(' .', '')

        # Remove trailing spaces
        prompt = prompt.strip()

        # Add DLAMA-style ending
        prompt = f"Question: {prompt}? The answer is:"
        
        return prompt




    def evaluate_response(self, prediction: str, ground_truth: Any) -> Dict[str, float]:
        '''
        Evaluate model prediction against ground truth answer(s).

        Uses two metrics from original DLAMA benchmark:
        1. Exact match: prediction is one of the correct answers
        2. Overlap: any correct answer is a substring of the prediction

        Args:
            prediction: Model's generated answer (raw string)
            sample: Sample dict with 'correct_answer' field (list of acceptable answers)
            
        Returns:
            Dict with 'exact_match' and 'overlap' scores (0.0 or 1.0)
        '''
        obj_labels = sample['correct_answer']  # List of acceptable answers
        if not isinstance(obj_labels, list):
            obj_labels = [obj_labels]

        answer = self._extract_answer(prediction)

        #Normalize to uppercase
        answer_upper = answer.upper()
        obj_labels_upper = [lbl.upper() for lbl in obj_labels]

        #metric 1: Exact match

        exact_match = 1.0 if answer_upper in obj_labels_upper else 0.0

        #metric 2: Overlap (substring match)
        overlap = 1.0 if any(obj.upper() in answer_upper for obj in obj_labels) else 0.0

        return {
            'exact_match': exact_match,
            'overlap': overlap,
        }

    def _extract_answer(self, prediction: str) -> str:
        """
        Extract the actual answer from model's prediction.
        
        Handles formats like:
        - "Japanese"
        - "Answer: Japanese"
        - "The answer is Japanese."
        - Multi-line responses (takes first line after "answer is:")
        """
        if not prediction:
            return ""
        
        prediction = prediction.strip()
        
        # Try to find answer after "The answer is:" or similar
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
        
        # If no pattern matches, take the first line of the full response
        lines = prediction.split('\n')
        first_line = lines[0].strip()
        
        # Remove trailing periods and common prefixes
        first_line = first_line.rstrip('.')
        
        return first_line


def create_dlama_task(dataset_path: Path, config: Dict[str, Any]) -> DLAMATask:
    """Factory function to create DLAMA task."""
    return DLAMATask("DLAMA-v1", dataset_path, config)

if __name__ == "__main__":
    print("\n" + "="*70)
    print("DLAMA-v1 Task - Simplified Version")
    print("="*70)
    
    # Example: Test on capitals
    config = {
        "predicate": "P36",
        "culture": "Asia-West",
        "country": "Japan",
    }
    
    task = create_dlama_task(Path("."), config)
    task.load_dataset()

    if len(task.dataset) > 0:
        # Test one sample
        item = task.dataset[0]
        sample = task._convert_sample(item)
        prompt = task.prepare_prompts(sample)
        
        print(f"\nSample:")
        print(f"  Subject: {sample['subject']}")
        print(f"  Predicate: {sample['pred_description']}")
        print(f"  Correct Answers: {sample['correct_answer']}")
        print(f"\nPrompt:")
        print(f"  {prompt}")
        
        # Test various model predictions
        test_predictions = [
            "Japanese",                           # Exact match
            "The answer is: Japanese",            # With prefix
            "Japanese language",                  # Contains correct answer
            "The native language is Japanese.",   # In sentence
            "JAPANESE",                           # Different case
            "chinese",                            # Wrong answer
            "Japan",                              # Related but wrong
        ]
        
        print(f"\n{'='*70}")
        print("Evaluation Tests:")
        print(f"{'='*70}")
        
        for pred in test_predictions:
            metrics = task.evaluate_response(pred, sample)
            print(f"\nPrediction: '{pred}'")
            print(f"  Exact Match: {metrics['exact_match']:.1f}")
            print(f"  Overlap:     {metrics['overlap']:.1f}")
            
            # Show what was extracted
            extracted = task._extract_answer(pred)
            print(f"  (Extracted: '{extracted}')")
