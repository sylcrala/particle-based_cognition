"""
central API endpoint for agent external research & learning
[UNDER DEVELOPMENT - needs integration with current API architecture]
"""

import re
from apis.api_registry import api

# Optional imports - handle gracefully if not available
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import wordnet as wn
    WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False

class ExternalResources:
    def __init__(self):
        self.logger = api.get_api("logger")
        self.spacy_available = SPACY_AVAILABLE
        self.wordnet_available = WORDNET_AVAILABLE
        
    def log(self, message, level="INFO"):
        if self.logger:
            self.logger.log(message, level, "ExternalResources", "ExternalResources")
        else:
            print(f"[ExternalResources] {message}")

    def spacy_def(self, term):
        """Get definition from spaCy if available"""
        if not self.spacy_available:
            return None
        
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(term)
            if doc and doc[0].has_vector:
                term_def = f"{term}: {doc[0].text} (vector length: {len(doc[0].vector)})"
                return term_def
            else:
                return "No definition found / no vector available"
        except Exception as e:
            self.log(f"Error getting spaCy definition for {term}: {e}", "ERROR")
            return None
        

    async def get_external_definitions(self, term):
        term_cleaned = re.sub(r"^[_#]+", "", term)

        sources = {
            "spacy": self.spacy_def(term_cleaned),
            #"wikipedia": self.wiki_def(term_cleaned)  # TODO: Implement
        }
        return {k: v for k, v in sources.items() if v}  # Filter out nulls

    def get_synonyms(self, term):
        """Returns a list of lowercase synonyms for a given term using WordNet."""
        if not self.wordnet_available:
            self.log(f"WordNet not available for synonym lookup: {term}", "WARNING")
            return []
            
        synonyms = set()
        try:
            for syn in wn.synsets(term):
                for lemma in syn.lemmas():
                    word = lemma.name().lower().replace('_', ' ')
                    if word != term.lower():
                        synonyms.add(word)
        except Exception as e:
            self.log(f"Error getting synonyms for {term}: {e}", "ERROR")
            
        return list(synonyms)

    def merge_synonyms(self, threshold=0.85):
        """
        Merges lexicon terms that are strong synonym candidates based on external knowledge.
        Only suggests or merges if both terms are in the lexicon and meet similarity/confidence.
        """
        lexicon_store = api.get_api("_agent_lexicon")
        if not lexicon_store:
            self.log("No lexicon store provided for synonym merging", "ERROR")
            return

        merged_count = 0
        terms = lexicon_store.get_terms() if hasattr(lexicon_store, 'get_terms') else []

        for term in terms:
            synonyms = self.get_synonyms(term)

            for synonym in synonyms:
                if synonym in terms and synonym != term:
                    # Optional: merge only if context overlap or both have definitions
                    entry_a = lexicon_store.lexicon.get(term)  # Fixed syntax
                    if entry_a is None:
                        self.log(f"Term '{term}' is not found in lexicon, skipping.")
                        continue
                        
                    entry_b = lexicon_store.lexicon.get(synonym)  # Fixed syntax
                    if entry_b is None:
                        self.log(f"Synonym '{synonym}' is not found in lexicon, skipping.")
                        continue                

                    shared_contexts = set(entry_a.get("contexts", [])) & set(entry_b.get("contexts", []))
                    if not shared_contexts and (not entry_a.get("definitions") or not entry_b.get("definitions")):
                        continue

                    # Merge synonym into main term
                    entry_a["definitions"].extend(d for d in entry_b["definitions"] if d not in entry_a["definitions"])
                    entry_a["contexts"].extend(c for c in entry_b["contexts"] if c not in entry_a["contexts"])
                    entry_a["sources"].extend(s for s in entry_b["sources"] if s not in entry_a["sources"])
                    entry_a["times_encountered"] += entry_b.get("times_encountered", 0)
                    entry_a["tags"] = list(set(entry_a.get("tags", []) + entry_b.get("tags", [])))

                    lexicon_store.lexicon[term] = entry_a
                    del lexicon_store.lexicon[synonym]
                    merged_count += 1

                    self.log(f"'{synonym}' merged into '{term}' based on synonym match and context overlap.")

        if merged_count > 0:
            if hasattr(lexicon_store, 'save'):
                lexicon_store.save()
            self.log(f"Synonym merge completed. {merged_count} synonym pairs merged.")
        else:
            self.log("No synonym pairs merged.")

# Register the API (but mark as experimental)
api.register_api("external_resources", ExternalResources())