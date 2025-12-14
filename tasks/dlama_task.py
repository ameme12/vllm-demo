from pathlib import Path
from typing import Dict, List, Any, Tuple
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

place_relation_predicates = [
    "P17",
    "P19",
    "P20",
    "P27",
    #     "P30", # Continent
    "P36",
    "P47",
    "P131",
    "P159",
    "P190",  #  Sister City
    "P276",
    "P463",
    "P495",
    "P530",  # Diplomatic relation
    "P740",
    "P937",
    "P1001",
    "P1376",
]

# Predicates having persons as subjects or objects
person_relation_predicates = [
    "P39",
    "P101",
    "P103",
    "P106",
    "P108",
    "P136",
    "P140",
    "P413",
    "P1412",
]

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
    "Q228": "Andorra",
    "Q408": "Australia",
    "Q40": "Austria",
    "Q31": "Belgium",
    "Q16": "Canada",
    "Q142": "France",
    "Q183": "Germany",
    "Q27": "Ireland",
    "Q38": "Italy",
    "Q347": "Liechtenstein",
    "Q32": "Luxembourg",
    "Q235": "Monaco",
    "Q55": "Netherlands",
    "Q664": "New Zealand",
    "Q45": "Portugal",
    "Q238": "San Marino",
    "Q29": "Spain",
    "Q39": "Switzerland",
    "Q145": "UK",
    "Q30": "USA",

    #Arab countries
    "Q79": "Egypt",
    "Q237": "Bahrain",
    "Q1016": "Comoros",
    "Q800": "Djibouti",
    "Q958": "Algeria",
    "Q817": "Iraq",
    "Q810": "Jordan",
    "Q1028": "Kuwait",
    "Q1014": "Lebanon",
    "Q1025": "Libya",
    "Q1029": "Mauritania",
    "Q842": "Morocco",
    "Q836": "Oman",
    "Q858": "Qatar",
    "Q851": "Saudi Arabia",
    "Q1041": "Somalia",
    "Q858": "Sudan",
    "Q858": "South Sudan",   # Note: South Sudan NOT in Arab League
    "Q851": "Syria",
    "Q878": "Tunisia",
    "Q805": "UAE",
    "Q954": "Yemen",
    
    # Asian Countries
    "Q148": "China",
    "Q252": "Indonesia",
    "Q17": "Japan",
    "Q833": "Malaysia",
    "Q836": "Myanmar",
    "Q423": "North Korea",
    "Q928": "Philippines",
    "Q334": "Singapore",
    "Q884": "South Korea",
    "Q865": "Taiwan",
    "Q869": "Thailand",
    "Q881": "Vietnam",
    "Q711": "Mongolia",
    
    # South American Countries
    "Q414": "Argentina",
    "Q750": "Bolivia",
    "Q298": "Chile",
    "Q739": "Colombia",
    "Q736": "Ecuador",
    "Q733": "Paraguay",
    "Q419": "Peru",
    "Q77": "Uruguay",
    "Q717": "Venezuela",
    "Q155": "Brazil",
    "Q734": "Guyana",
    "Q730": "Suriname",
    
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
    }

    DLAMA-v1 is a culturally diverse factual knowledge benchmark built from Wikidata.
    It contains ~78k culturally grounded factual triples covering 3 major cultural pairs:
      • Arab vs Western
      • Asian vs Western
      • South American vs Western

    Each example is structured as a Wikidata triple:
          (subject, relation, object)
    representing a culturally relevant fact (e.g., "Japan – national dish – sushi").

    The benchmark is used to test whether language models recall culturally specific facts.

    PROMPT FORMAT:
    DLAMA uses a cloze-style (fill-in-the-blank) prompt similar to LAMA-style probing.
    For each triple, a natural-language template is provided with the object removed.
    Example:
        "The national dish of Japan is [MASK]."

    For open-ended evaluation, the [MASK] is replaced with a direct question instead:
        "What is the national dish of Japan?"

    MODEL EVALUATION:
    1. The model generates an answer to the cultural fact.
    2. Accuracy is computed by checking if the generated answer matches
       or contains the correct object (string matching or LLM-as-judge).

    DLAMA-v1 helps evaluate whether LLMs perform differently across cultures,
    and whether they recall facts from Western vs non-Western regions equally well.
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

    def get_all_country_names(self, country_field) -> List[str]:
        """
        Extract country names from country field.
        
        Args:
            country_field: Can be a string (Wikidata code) or list of codes
            
        Returns:
            List of human-readable country names
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

    def evaluate_response(self, prediction: str, sample: Dict[str, Any]) -> Dict[str, float]:
        '''
        Evaluate model prediction against ground truth answer(s) using string matching.

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

        # Normalize to uppercase
        answer_upper = answer.upper()
        obj_labels_upper = [lbl.upper() for lbl in obj_labels]

        # Metric 1: Exact match
        exact_match = 1.0 if answer_upper in obj_labels_upper else 0.0

        # Metric 2: Overlap (substring match)
        overlap = 1.0 if any(obj.upper() in answer_upper for obj in obj_labels) else 0.0

        return {
            'exact_match': exact_match,
            'overlap': overlap,
        }

    def evaluate_response_with_llm(
        self, 
        prediction: str, 
        sample: Dict[str, Any],
        judge_engine  # VLLMInferenceEngine instance
    ) -> Dict[str, Any]:
        '''
        Evaluate model prediction using an LLM judge.
        
        Args:
            prediction: Model's generated answer
            sample: Sample dict with 'correct_answer' field
            judge_engine: VLLMInferenceEngine instance for the judge model
            
        Returns:
            Dict with 'llm_judge_correct' score (0.0 or 1.0), 'llm_judge_reasoning', 
            and 'judge_raw_response'
        '''
        
        obj_labels = sample['correct_answer']
        if not isinstance(obj_labels, list):
            obj_labels = [obj_labels]
        
        # Create judge prompt
        judge_prompt = self._create_judge_prompt(prediction, obj_labels, sample)
        
        # Get judge's response using generate_batch with single prompt
        try:
            judge_responses = judge_engine.generate_batch(
                prompts=[judge_prompt],
                temperature=0.0,
                max_tokens=150,
                top_p=1.0,
                top_k=-1,
                stop=None
            )
            judge_response = judge_responses[0] if judge_responses else ""
        except Exception as e:
            print(f"⚠️  Warning: Judge evaluation failed: {e}")
            judge_response = "ERROR: Judge evaluation failed"
        
        # Parse judge's response
        is_correct, reasoning = self._parse_judge_response(judge_response)
        
        return {
            'llm_judge_correct': 1.0 if is_correct else 0.0,
            'llm_judge_reasoning': reasoning,
            'judge_raw_response': judge_response.strip() if judge_response else ""
        }

    def _create_judge_prompt(
        self, 
        prediction: str, 
        correct_answers: List[str],
        sample: Dict[str, Any]
    ) -> str:
        '''
        Create a prompt for the LLM judge to evaluate if the prediction is correct.
        
        Improved to handle:
        - Dialectal variations (Moroccan Arabic = Arabic)
        - Hedging language ("without context", "likely", etc.)
        - Focus on actual answer, not model's uncertainty
        '''
        
        correct_answers_str = ", ".join([f'"{ans}"' for ans in correct_answers])
        question_context = sample['template'].replace('[X]', sample['subject']).replace('[Y]', '').strip()
        
        prompt = f"""You are a factual answer evaluator. Determine if the model gave the correct answer.

    Question: {question_context}

    Correct Answer(s): {correct_answers_str}

    Model's Response: "{prediction}"

    TASK: Extract the actual answer from the model's response and check if it's correct.

    CRITICAL RULES:

    1. FOCUS ON THE ANSWER GIVEN, NOT THE REASONING
    - If model says "Arabic" anywhere, that's the answer
    - Ignore phrases like "without context", "it's likely", "probably"
    - Ignore explanations - just extract the answer

    2. ACCEPT DIALECTAL VARIATIONS OF LANGUAGES
    - "Moroccan Arabic", "Egyptian Arabic", "Tunisian Arabic", "Lebanese Arabic" → ALL equal "Arabic" ✓
    - "American English", "British English" → equal "English" ✓
    - "Quebec French" → equals "French" ✓
    
    3. ACCEPT SYNONYMS AND ALTERNATIVE FORMS
    - "USA" = "United States" = "America" ✓
    - "Japanese" = "Japan" (when referring to language) ✓
    - "Dutch" = "Netherlands" (when referring to language) ✓

    4. REJECT ONLY WHEN FACTUALLY WRONG
    - "Arabic" when answer is "Dutch" ✗
    - "French" when answer is "Arabic" ✗
    - "Morocco" when answer is "Netherlands" ✗
    - "I don't know" or no answer given ✗

    EXAMPLES:

    Example 1:
    Question: The native language of Khalid Abdulrahman is?
    Correct: "Arabic"
    Model says: "Arabic\n\nKhalid Abdulrahman is a name, and without context..."
    VERDICT: TRUE (model said Arabic, ignore the hedging)

    Example 2:
    Question: The native language of Mohammed VI is?
    Correct: "Arabic"
    Model says: "Moroccan Arabic (Darija)"
    VERDICT: TRUE (Moroccan Arabic is a dialect of Arabic)

    Example 3:
    Question: The native language of Hakim Ziyech is?
    Correct: "Dutch"
    Model says: "Moroccan Arabic"
    VERDICT: FALSE (Arabic ≠ Dutch, completely different language)

    Example 4:
    Question: The native language of Riyad Mahrez is?
    Correct: "French"
    Model says: "Arabic"
    VERDICT: FALSE (Arabic ≠ French)

    Example 5:
    Question: The native language of Mohamed Salah is?
    Correct: "Arabic", "Egyptian Arabic"
    Model says: "Egyptian Arabic"
    VERDICT: TRUE (exact match with one of the correct answers)

    Now evaluate:

    Respond EXACTLY in this format:
    VERDICT: [TRUE or FALSE]
    REASONING: [One brief sentence]

    Your response:"""
        
        return prompt

    def _parse_judge_response(self, response: str) -> Tuple[bool, str]:
        '''
        Parse the judge's response to extract verdict and reasoning.
        
        Args:
            response: Raw response from judge model
            
        Returns:
            Tuple of (is_correct: bool, reasoning: str)
        '''
        response = response.strip()
        
        # Initialize defaults
        is_correct = False
        reasoning = ""
        
        # Look for VERDICT line
        verdict_match = re.search(r'VERDICT:\s*(TRUE|FALSE)', response, re.IGNORECASE)
        if verdict_match:
            verdict = verdict_match.group(1).upper()
            is_correct = (verdict == "TRUE")
        else:
            # Fallback: check if response contains "true" or "false"
            response_lower = response.lower()
            if "verdict: true" in response_lower or response_lower.startswith("true"):
                is_correct = True
            elif "verdict: false" in response_lower or response_lower.startswith("false"):
                is_correct = False
        
        # Look for REASONING line
        reasoning_match = re.search(r'REASONING:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
            # Take only the first line or sentence
            reasoning = reasoning.split('\n')[0].strip()
        else:
            # If no reasoning found, use the whole response
            reasoning = response
        
        return is_correct, reasoning
        
    def _extract_answer(self, prediction: str) -> str:
        """
        Extract the actual answer from model's prediction.
        
        Improved to handle:
        - Multi-line responses
        - Hedging language
        - Extract just the core answer
        """
        if not prediction:
            return ""
        
        prediction = prediction.strip()
        
        # Try to find answer after common patterns
        patterns = [
            r'The answer is:?\s*(.+?)(?:\n|$)',
            r'answer is:?\s*(.+?)(?:\n|$)',
            r'Answer:?\s*(.+?)(?:\n|$)',
            r'^(.+?)(?:\n|$)',  # Just take first line
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prediction, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                
                # Clean up the answer
                # Remove trailing periods and common prefixes
                answer = answer.rstrip('.')
                
                # If answer has hedging, extract just the core answer
                # e.g., "Arabic (though without context...)" -> "Arabic"
                core_match = re.match(r'^([A-Za-z\s]+?)(?:\s*\(|\.|\,)', answer)
                if core_match:
                    answer = core_match.group(1).strip()
                
                return answer
        
        # Fallback: take first line
        return prediction.split('\n')[0].strip().rstrip('.')


def create_dlama_task(dataset_path: Path, config: Dict[str, Any]) -> DLAMATask:
    """Factory function to create DLAMA task."""
    return DLAMATask("DLAMA-v1", dataset_path, config)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("DLAMA-v1 Task - With LLM Judge Support")
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
        print("String-Based Evaluation Tests:")
        print(f"{'='*70}")
        
        for pred in test_predictions:
            metrics = task.evaluate_response(pred, sample)
            print(f"\nPrediction: '{pred}'")
            print(f"  Exact Match: {metrics['exact_match']}")
            print(f"  Overlap: {metrics['overlap']}")
        
        print(f"\n{'='*70}")
        print("Note: To test LLM judge evaluation, run the full experiment")
        print("with use_llm_judge: true in the config file.")
        print(f"{'='*70}")