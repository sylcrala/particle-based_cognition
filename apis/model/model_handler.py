"""
LLM central handler - agent interfaces with LLM through this
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
import datetime as dt
from apis.api_registry import api


config = api.get_api("config")
if not config:
    raise RuntimeError("Config API not registered or missing!")
llm_config = config.get_llm_config()

class ModelHandler:
    def __init__(self):
        torch.cuda.empty_cache()  # clearing cuda cache

        # Check CUDA availability and fall back to CPU if needed
        cuda_available = False
        
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                cuda_available = True
                print(f"[ModelHandler] CUDA available with {torch.cuda.device_count()} device(s)")
            else:
                cuda_available = False
                print("[ModelHandler] CUDA not available, falling back to CPU")
        except Exception as e:
            cuda_available = False
            print(f"[ModelHandler] CUDA check failed: {e}, falling back to CPU")

        # pull model args from config
        self.model_path = llm_config.get("model_path")
        self.model_name = llm_config.get("model_name")
        self.device = "cpu" if not cuda_available else llm_config.get("device", "cpu")  # Force CPU if no CUDA
        self.temperature = llm_config.get("temperature")
        self.load_in_4bit = llm_config.get("load_in_4bit") and cuda_available  # Only use 4bit if CUDA available
        self.max_new_tokens = llm_config.get("max_new_tokens")
        self.top_p = llm_config.get("top_p")
        self.top_k = llm_config.get("top_k")
        self.do_sample = llm_config.get("do_sample")
        self.max_cpu_memory = llm_config.get("max_cpu_memory")
        self.max_gpu_memory = llm_config.get("max_gpu_memory")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token  

        self.metavoice = api.get_api("meta_voice")  # Fixed: was "metavoice"
        if not self.metavoice:
            self.logger = api.get_api("logger")
            if self.logger:
                self.logger.log("MetaVoice API not available during ModelHandler init", "WARNING", "ModelHandler.__init__", "ModelHandler")
            # Don't raise error, just log warning - metavoice might be optional
        
        self.events = api.get_api("event_handler")
        if not self.events:
            self.logger = api.get_api("logger") 
            if self.logger:
                self.logger.log("Event Handler API not available during ModelHandler init", "WARNING", "ModelHandler.__init__", "ModelHandler")
            # Don't raise error, just log warning - events might be optional

        # Set up BitsAndBytesConfig and device mapping
        try:
            if self.device == "cuda" and cuda_available:
                # Try CUDA setup with quantization
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=self.load_in_4bit,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_enable_fp32_cpu_offload=True
                )
                max_memory = {0: f"{self.max_gpu_memory}", "cpu": f"{self.max_cpu_memory}"}
                device_map = "auto"
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map=device_map,
                    max_memory=max_memory,
                    quantization_config=quant_config,
                    trust_remote_code=False,
                    torch_dtype=torch.float16
                )
            else:
                # CPU fallback
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="cpu",
                    trust_remote_code=False,
                    torch_dtype=torch.float32
                )
                
        except Exception as e:
            # Final fallback to CPU if CUDA setup fails
            if self.logger:
                self.logger.log(f"CUDA model loading failed, falling back to CPU: {e}", "WARNING", "ModelHandler.__init__", "ModelHandler")
            
            self.device = "cpu"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="cpu",
                trust_remote_code=False,
                torch_dtype=torch.float32
            )

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            model_kwargs={"temperature": self.temperature, "max_new_tokens": self.max_new_tokens, "do_sample": self.do_sample, "top_p": self.top_p, "top_k": self.top_k}
        )


   
    async def generate(self, prompt: None, context_id=None, tags=None, source=None, **kwargs) -> str:

        if not prompt:
            return "[Error: No prompt provided]"
        
        prompt = f"Received message from {config.user_name}: <s>{prompt}</s>" if source == "user_input" else f"I thought about: {prompt}"

        #debug
        print(f"[ModelHandler] Generating response for prompt: {prompt}")
        print(f"[ModelHandler] Context ID: {context_id}, Tags: {tags}")
        #_log_event("generation", f"Generating response for prompt: {prompt}, Context ID: {context_id}, Tags: {tags}")
        
        try:
            result = self.generator(prompt, **kwargs)
            output = result[0]["generated_text"].strip() if isinstance(result, list) else "[Error: Format]"
            return output
        except Exception as e:
            output = f"[Generation Error: {e}]"
            return output

        #result = self.generator(prompt, **kwargs)
        #output = result[0]["generated_text"].strip() if isinstance(result, list) else "[Error: Format]"

    async def reflect(self, particle):
        if self.metavoice:
            return await self.metavoice.reflect(particle)
        return await self.generate(particle.get_content())
    


api.register_api("model_handler", ModelHandler())