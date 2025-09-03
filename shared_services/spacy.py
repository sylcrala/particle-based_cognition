"""
SpaCy NLP Integration
"""



import spacy
from apis.api_registry import api

logger = api.get_logger("logger")
nlp = spacy.load("en_core_web_sm")


def log(self, message, level = None, context = None):
    source = "SpaCyUtility"

    if context != None:
        context = context
    else:
        context = "no context"

    if level != None:
        level = level
    else:
        level = "INFO"

    api.call_api("logger", "log", (message, level, context, source))

def lookup_definition(term):
    try:
        doc = nlp(term)
        if doc and doc[0].pos_:
            return f"{term} is used as a {doc[0].pos_} (e.g., '{doc.text}')"
        return "No grammatical insight found"
    except Exception as e:
        log(f"(spaCy error) {str(e)}", level="ERROR", context = "lookup_definition()")
        return f"(spaCy error) {str(e)}"
    
def classify_term(token: str, context=None) -> dict:
    doc = nlp(token)
    if not doc or len(doc) == 0:
        log(f"(spaCy error) No tokens found in '{token}'", level="ERROR", context = "classify_term()")
        return {"type": "unknown", "tags": [], "intent": None}

    token_obj = doc[0]

    # Normalize POS
    pos_map = {
        "NOUN": "noun",
        "VERB": "verb",
        "ADJ": "adjective",
        "ADV": "adverb",
        "PUNCT": "punctuation",
        "SPACE": "control",
        "PROPN": "proper_noun",
        "INTJ": "interjection",
        "DET": "determiner",
        "SYM": "symbol"
    }
    term_type = pos_map.get(token_obj.pos_, "unknown")

    # Tags based on use + structure
    tags = []
    if token_obj.is_stop:
        tags.append("filler")
    if token_obj.is_alpha:
        tags.append("word")
    if token_obj.is_punct:
        tags.append("punctuation")
    if token_obj.like_num:
        tags.append("numeric")

    # Optional: Add external info
    if context:
        if "reflection" in context:
            tags.append("reflective")
        if "generated-expression" in context:
            tags.append("expressive")

    log(f"Classified term '{token}' with context '{context}': {term_type}, {tags}", context = "classify_term()")
    return {
        "type": term_type,
        "tags": tags,
        "intent": "reflective" if "reflection" in (context or "") else None
    }
