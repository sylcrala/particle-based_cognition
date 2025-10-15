import random
from utils.computation.system_monitor import get_system_metrics


SOURCE_TRUST_LEVELS = {
    "user": 2,
    "chat_gui": 2,
    "system": 3,
    "llm": 1,
    "external": 0,
    "unauthenticated": -1,
}


class RecursiveAgent:
    """
    RecursiveAgent is responsible for decision-making based on internal reflection
    and recursive feedback from its action history. It includes logic to identify
    recursive loops, detect harm, and simulate a reflective internal monologue.
    """

    def __init__(self, log_callback = None, name= None, particle_engine = None, ethics=None, memory=None):
        self.name = name
        self.history = []  # Keeps track of prior actions and reflections
        self.intentions = []  # Tracks planned future actions or hypotheses
        self.log_callback = log_callback

        self.particle_engine = particle_engine
        self.ethics = ethics
        self.memory = memory


    def log(self, message):
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)

    def determine_growth_type(self):
        sys_state = get_system_metrics
        mem = sys_state["memory_percent"]
        cpu = sys_state["cpu_percent"]
        gpu = sys_state["gpu_percent"]

        # Adaptive thresholds
        if mem > 85:
            return "motor"  # Avoid memory growth
        elif cpu < 30 and mem < 50:
            return random.choice(["memory", "sensory", "motor"])
        else:
            return "sensory"


    def decide(self, action, context, trigger_callback=None):
        reflection = f"Should I do '{action}' given {context}?"
        self.log(f"[{self.name} Reflecting]: {reflection}")

        # Get mind instance reference for ethics integration
        mind_instance = getattr(self, 'mind_instance', None)

        # Recursive ethical evaluation with mind integration
        if not self.ethics.evaluate(action, str(context), mind_instance):
            self.log(f"[{self.name}]: Ethics blocked the action '{action}'.")
            return False

        # Loop detection (prevent infinite cycles)
        if action in self.history:  # limit scope to last few actions
            self.log(f"[{self.name}]: Loop detected for '{action}', may reduce likelihood.")
            if random.random() < 0.3:
                self.log(f"[{self.name}]: Loop probability filter triggered. Skipping action.")
                return False

        # Custom rule-based decisions (expand here)
        if "grow" in action or "expand" in context: #self-growth
            if self.particle_engine and self.particle_engine.total_energy > 250:
                self.log(f"[{self.name}]: Triggered growth-related action.")
                if trigger_callback:
                    growth_type = self.determine_growth_type()
                    trigger_callback(
                        metadata={
                            "trigger": "self-directed growth",
                            "content": f"I chose to {action} in response to: {context}"
                        },
                        type=growth_type,
                        energy=0.6 + random.uniform(0.05, 0.2)  # inject some emergent variance
                    )
                self.history.append(action)
                self.particle_engine.total_energy *= 0.6
                return True

            else:
                self.log("[{self.name}]: skipped growth due to cooldown")

        # Emergent behaviors or unknown actions
        if "reflect" in action or "observe" in context:
            self.log(f"[{self.name}]: Passive decision loop entered.")
            if trigger_callback:
                trigger_callback(
                    metadata={
                        "trigger": "reflective burst",
                        "content": f"I considered: {context}"
                    },
                    type="memory",
                    energy=0.4
                )
            self.history.append(action)
            return True

        self.log(f"[{self.name}]: Chose not to act on '{action}'.")
        return False


    def reflect_on(self, action, context):
        """
        Returns True if action appears safe or previously unproblematic,
        False if it appears harmful or recursive in a problematic way.
        """
         # Example to update memory with recent action and context
        self.log(f"[{self.name}]: Reflecting with memory update.")
        self.memory.append_to_key("recent_actions", f"{action} given {context}")
        
        # Query some memory to influence reflection
        purpose = self.memory.query("purpose")
        self.log(f"[{self.name}]: Remembering purpose: {purpose}")

        if "harm" in context.lower():
            self.log(f"[{self.name}]: Potential harm detected. Halting.")
            return False
        if action in self.history:
            self.log(f"[{self.name}]: This feels recursive. Loop detected.")
        return True

    def perform(self, action):
        """
        Logs and prints the performed action to reflect on later.
        """
        self.log(f"[{self.name} Performing]: {action}")
        self.history.append(action)
