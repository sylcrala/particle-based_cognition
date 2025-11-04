"""
Particle-based Cognition Engine - spacy NLP utility functions
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

def analyze(term):
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
            if "user_input" in str(context):
                tags.append("user_input")
            

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