"""
central configuration
"""

from apis.api_registry import api

class Config():
    def __init__(self):

        self.user_name = "Tony"
        
        self.agent_mode = "cog-growth"   # options: llm-extension (uses LLM for supplementing voice module, linguistic capabilities and knowledge base), cog-growth (no LLM for voice module - agent "grows" its own linguistic capabilities and knowledge base over time)
        self.agent_name = "Misty" if self.agent_mode == "llm-extension" else "Iris"  # Misty for cog-growth, Iris for llm-extension

        self.AGENT_CONFIG = {
            "name": self.agent_name,
            "mode": self.agent_mode, 
            "memory_dir": f"./data/{self.agent_name}_{self.agent_mode}",
        } 

        self.LLM_CONFIG = {
            "model_path": "./models/core/",
            "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
            "device": "cuda",
            "load_in_4bit": True,
            "temperature": 1.0,
            "max_new_tokens": 1200,
            "top_p": 0.95,
            "top_k": 50,
            "do_sample": True,
            "max_cpu_memory": "6GB",
            "max_gpu_memory": "8GB"
        }

        self.ADAPTIVE_ENGINE_CONFIG = {
            "base_metric": "euclidean",
            "mode": "blend",
            "lambda_blend": 0.5
        }

    def get_llm_config(self):
        return self.LLM_CONFIG

    def get_adaptive_engine_config(self):
        return self.ADAPTIVE_ENGINE_CONFIG

    def get_agent_config(self):
        return self.AGENT_CONFIG

api.register_api("config", Config())