"""
dedicated agent launcher - does not launch TUI
"""

import asyncio
from apis.api_registry import api

def launch_agent_core():
    from apis.agent.core import AgentCore
    from apis.model.model_handler import ModelHandler

    agent_core = AgentCore()
    model_handler = ModelHandler()

    api.register_api("agent", agent_core)
    api.register_api("model", model_handler)

    return agent_core, model_handler

