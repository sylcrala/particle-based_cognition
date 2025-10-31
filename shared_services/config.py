"""
Particle-based Cognition Engine - system configuration settings
Copyright (C) 2025 sylcrala

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version, subject to the additional terms 
specified in TERMS.md.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License and TERMS.md for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Additional terms apply per TERMS.md. See also ETHICS.md.
"""

from apis.api_registry import api

class Config():
    def __init__(self):
        self.system_language = "en"

        # os-specific settings
        #TODO: add support for OS detection and autofill
        self.os_name = "linux"          # dont worry this isn't used yet, it's just a placeholder for future use
        self.os_version = "fedora-42"   # dont worry this isn't used yet, it's just a placeholder for future use
        self.wayland_active = False     # this one is used for systems using wayland display server (like Fedora with GNOME, if you're using wayland and have some issues with the GUI or visualizer, try setting to True)

        self.user_name = "User"         # default user name for interactions - update this so the agent can address you properly in time
        
        self.agent_mode = "cog-growth" 
        # options:   
        # cog-growth: no LLM for voice module - agent "grows" its own linguistic capabilities and knowledge base over time
        # llm-extension: uses LLM for supplementing voice module, linguistic capabilities and knowledge base

        # llm-extension mode will be developed alongside cog-growth mode, but the main priority of the project is to develop cog-growth mode and explore it's capabilities under this framework.        
        
        self.agent_name = "Misty" if self.agent_mode == "llm-extension" else "Iris"  # Misty for cog-growth, Iris for llm-extension. Change as desired.

        self.AGENT_CONFIG = {
            "name": self.agent_name,
            "mode": self.agent_mode, 
            "memory_dir": f"./data/{self.agent_name}_{self.agent_mode}",
        } 

        self.LLM_CONFIG = {         # UPDATE THIS ACCORDING TO THE MODEL YOU WANT TO USE - Current settings and model handler only support Mistral 7b Instruct v.02, but you can change to another model by updating these settings and adding a new model handler in apis/agent/llm_handlers/ as long as the handler follows the format of the original model handler with /apis/model/ (specifically the generate() method and returning the output), or just submit an issue or PR and i'll look into it!
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