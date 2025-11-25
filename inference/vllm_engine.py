from vllm import LLM, SamplingParams
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import torch

@dataclass
class VLLMConfig:
    """Configuration for vLLM inference"""
    model_name: str
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    trust_remote_code: bool = True
    dtype: str = "auto"  # auto, half, float16, bfloat16, float, float32
    quantization: Optional[str] = None  # awq, gptq, squeezellm, fp8
    swap_space: int = 4  # CPU swap space in GiB
    enforce_eager: bool = False  # Disable CUDA graph
    max_num_seqs: int = 256  # Max number of sequences per iteration
    seed: int = 42
    
class VLLMInferenceEngine:
    """vLLM-based inference engine for efficient LLM inference"""
    
    def __init__(self, config: VLLMConfig):
        self.config = config
        self.llm = None
        self._initialize_engine()
        
    def _initialize_engine(self):
        """Initialize vLLM engine with configuration"""
        print(f"Initializing vLLM engine with model: {self.config.model_name}")
        print(f"Tensor parallel size: {self.config.tensor_parallel_size}")
        print(f"GPU memory utilization: {self.config.gpu_memory_utilization}")
        
        self.llm = LLM(
            model=self.config.model_name,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            trust_remote_code=self.config.trust_remote_code,
            dtype=self.config.dtype,
            quantization=self.config.quantization,
            swap_space=self.config.swap_space,
            enforce_eager=self.config.enforce_eager,
            max_num_seqs=self.config.max_num_seqs,
            seed=self.config.seed
        )
        
        print("vLLM engine initialized successfully!")
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        max_tokens: int = 512,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        n: int = 1,
        use_beam_search: bool = False,
        best_of: Optional[int] = None,
    ) -> List[str]:
        """
        Generate text using vLLM
        
        Args:
            prompts: Single prompt or list of prompts
            temperature: Sampling temperature (0 for greedy)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            max_tokens: Maximum tokens to generate
            presence_penalty: Presence penalty
            frequency_penalty: Frequency penalty
            stop: Stop sequences
            n: Number of completions per prompt
            use_beam_search: Whether to use beam search
            best_of: Generate best_of completions and return best n
            
        Returns:
            List of generated texts
        """
        # Convert single prompt to list
        if isinstance(prompts, str):
            prompts = [prompts]
            single_prompt = True
        else:
            single_prompt = False
        
        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            stop=stop,
            n=n,
            use_beam_search=use_beam_search,
            best_of=best_of,
        )
        
        # Generate outputs
        outputs = self.llm.generate(prompts, sampling_params)
        
        # Extract generated texts
        results = []
        for output in outputs:
            generated_texts = [o.text for o in output.outputs]
            if n == 1:
                results.append(generated_texts[0])
            else:
                results.append(generated_texts)
        
        # Return single result if single prompt was provided
        if single_prompt:
            return results[0]
        
        return results
    
    def generate_batch(
        self,
        prompts: List[str],
        sampling_params_list: Optional[List[SamplingParams]] = None,
        **default_params
    ) -> List[str]:
        """
        Generate text for batch of prompts with optional per-prompt parameters
        
        Args:
            prompts: List of prompts
            sampling_params_list: Optional list of SamplingParams (one per prompt)
            **default_params: Default sampling parameters for all prompts
            
        Returns:
            List of generated texts
        """
        if sampling_params_list is None:
            # Use default parameters for all prompts
            sampling_params = SamplingParams(**default_params)
            outputs = self.llm.generate(prompts, sampling_params)
        else:
            # Use per-prompt parameters
            outputs = self.llm.generate(prompts, sampling_params_list)
        
        return [output.outputs[0].text for output in outputs]
    
    def get_tokenizer(self):
        """Get the tokenizer used by the model"""
        return self.llm.get_tokenizer()