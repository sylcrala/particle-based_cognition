from datetime import datetime
import uuid
import json

from apis.agent.cognition.particles.utils.particle_frame import Particle
from apis.api_registry import api


class SensoryParticle(Particle):
    def __init__(self, **kwargs):
        super().__init__(type="sensory", **kwargs)
        self.token = self.metadata.get("content", "")
        self.embedding = self._message_to_vector(self.token)

        self.metrics = api.get_api("system_metrics")

        self.is_unknown_feeling = self.metadata.get("is_unknown_feeling", False)
        self.original_type = self.metadata.get("original_type", None)

        self.environmental_state = {
            "last_state": None,
            "current_state": None,
            "sensory_buffer": [],
            "unknown_experiences": [],
            "timestamp": None
        }

        self.sensory_decay = 0.95

    def process_environmental_input(self, input_data = None, input_type = None):
        """Handles sensory input and background environmental state monitoring - currently only used for environmental state monitoring during maintenance cycles"""
        self.log(f"Processing environmental input: {input_data} of type {input_type}", "DEBUG", "process_environmental_input", "SensoryParticle")
        try:
            current_time = datetime.now().timestamp()

            if self.environmental_state["current_state"] is not None:
                self.environmental_state["last_state"] = self.environmental_state["current_state"]
            
            state = self.metrics.get_current_metrics()

            self.environmental_state["current_state"] = state
            self.field.current_metrics = state
            self.environmental_state["timestamp"] = current_time



            if input_type == "metrics":
                self.environmental_state["sensory_buffer"].append({
                    "data": state,
                    "type": input_type
                })

            else:
                self.environmental_state["unknown_experiences"].append({
                    "data": input_data,
                    "type": "unknown"
                })
            
            if self.activation > 0.8:
                self.activation = min(self.activation + 0.2, 1.0)
            
            return state
        
        except Exception as e:
            self.log(f"Error processing environmental input: {e}", "ERROR", "process_environmental_input", "SensoryParticle")
            return "issue processing system state"
        
            
            

