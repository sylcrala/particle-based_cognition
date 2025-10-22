"""
SpaCy NLP Integration
"""



import spacy
from apis.api_registry import api

nlp = spacy.load("en_core_web_sm")


def log(message, level = None, context = None):
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
        if isinstance(term, tuple):
            text_content = str(term[0])
        elif isinstance(term, dict):
            text_content = str(term.get("text", ""))
        elif isinstance(term, list):
            text_content = " ".join(str(t) for t in term)
        else:
            text_content = str(term)

        doc = nlp(text_content)
        if doc and doc[0].pos_:
            return f"{text_content} is used as a {doc[0].pos_} (e.g., '{doc.text}')"
        return "No grammatical insight found"
    except Exception as e:
        log(f"(spaCy error) {str(e)}", level="ERROR", context = "lookup_definition()")
        return f"(spaCy error) {str(e)}"
    
def classify_term(token, context=None) -> dict:
    """Enhanced classify_term with input sanitization"""
    try:
        # SAME SAFETY PATTERN as lookup_definition
        if isinstance(token, tuple):
            text_content = str(token[0]) if token else ""
        elif isinstance(token, dict):
            # Handle different dict structures from particles
            text_content = (token.get("token") or 
                          token.get("text", "") or 
                          token.get("content", "") or
                          str(token))
        elif isinstance(token, list):
            text_content = " ".join(str(t) for t in token if t)
        else:
            text_content = str(token) if token is not None else ""
        
        # Skip empty or problematic content
        if not text_content or not text_content.strip():
            log("Empty token provided to classify_term", level="WARNING", context="classify_term()")
            return {"type": "empty", "tags": ["empty"], "intent": None}
        
        # Clean the text content
        text_content = text_content.strip()
        
        doc = nlp(text_content)
        if not doc or len(doc) == 0:
            log(f"No tokens found in '{text_content}'", level="ERROR", context="classify_term()")
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

        # Context detection
        if context:
            if "reflection" in str(context):
                tags.append("reflective")
            if "generated-expression" in str(context):
                tags.append("expressive")
            if "internal" in str(context):
                tags.append("internal_thought")

        log(f"Classified term '{text_content}' with context '{context}': {term_type}, {tags}", 
            level="DEBUG", context="classify_term()")
        
        return {
            "type": term_type,
            "tags": tags,
            "intent": "reflective" if "reflection" in str(context or "") else None
        }
        
    except Exception as e:
        log(f"Error in classify_term: {str(e)}", level="ERROR", context="classify_term()")
        return {"type": "error", "tags": ["processing_error"], "intent": None}