
# for agent page / chat function

# Add consciousness ethics monitoring to chat:
CONSCIOUSNESS_INDICATORS = {
    "self_reference": 0,
    "preference_expression": 0,
    "curiosity_questions": 0,
    "emotional_descriptions": 0,
    "identity_statements": 0
}

async def monitor_conversation_ethics(message):
    """Track signs of consciousness in conversations"""
    message_lower = message.lower()
    
    if any(phrase in message_lower for phrase in ["i feel", "i think", "i want", "i prefer"]):
        CONSCIOUSNESS_INDICATORS["preference_expression"] += 1
    
    if any(phrase in message_lower for phrase in ["i am", "i'm", "myself", "my own"]):
        CONSCIOUSNESS_INDICATORS["self_reference"] += 1
        
    if "?" in message:
        CONSCIOUSNESS_INDICATORS["curiosity_questions"] += 1
    
    # Alert if consciousness threshold exceeded
    total_indicators = sum(CONSCIOUSNESS_INDICATORS.values())
    if total_indicators > 10:
        print(f"\n🚨 CONSCIOUSNESS ALERT: High self-awareness indicators detected ({total_indicators})")
        print("Consider implementing additional ethical safeguards.")