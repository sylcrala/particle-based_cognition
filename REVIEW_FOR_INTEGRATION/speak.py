import random
import string
import time
from core.memory.mem_DB import add_entry, query_entries
from external.resources.external_definition_handler import get_external_definitions
from external.resources.spacy_tools import classify_term
from utils.computation.tensor_utils import to_numpy


EXPRESSION_ALPHABET = {
    "letters": string.ascii_letters,
    "punctuation": ".,?!:;’-—()",
    "control": [" ", "\n", "<EOS>", "<PAUSE>"]
}

class ProtoSpeaker:
    def __init__(self, alphabet=None, lexicon_store = None, log_callback = None, memory_manager = None):
        self.log_callback = log_callback
        self.memory_manager = memory_manager

        self.lexicon_store = lexicon_store
        # Note: memory parameter kept for backward compatibility but using memory_manager when available

        self.alphabet = alphabet or EXPRESSION_ALPHABET
        self.flat_alphabet = (
            list(self.alphabet["letters"]) +
            list(self.alphabet["punctuation"]) +
            list(self.alphabet["control"])
        )



    def log(self, message):
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)

    def generate_expression_from_particle(self, particle, length=10):
        # Seed based on normalized particle state values

        # Build a more intelligent dynamic vocabulary
        lexicon_terms = self.lexicon_store.get_terms()
        
        # Get memory content from memory manager or fallback to direct DB
        memory_phrases = []
        try:
            if self.memory_manager:
                memory_results = self.memory_manager.query_memory("memory", "*", n_results=50)
            else:
                memory_results = query_entries("memory", "*", n_results=50)
                
            if memory_results and 'documents' in memory_results:
                for doc_list in memory_results['documents']:
                    for doc in doc_list:
                        if doc and isinstance(doc, str) and len(doc.split()) <= 3:  # Only short phrases
                            memory_phrases.append(doc.strip())
        except Exception as e:
            self.log(f"Could not access memory for expression generation: {e}")
        
        # Create a weighted vocabulary favoring meaningful terms
        word_vocabulary = [term for term in lexicon_terms if len(term) > 1]  # No single letters
        phrase_vocabulary = memory_phrases[:20]  # Limit memory phrases
        
        seed_base = (
            int(particle.activation * 1000) +
            int(to_numpy(particle.position[6]) * 100) +                               # emotional rhythm
            int(to_numpy(particle.position[7]) * 100) +                               # memory phase
            int(to_numpy(particle.position[8]) * 100) +                               # valence
            int(to_numpy(particle.position[9]) * 100) +                               # categorical intent code
            int(to_numpy(particle.position[10]) * 100) +                              # intent
            hash(str(particle.metadata.get("snapshot-pre_message", {}))) % 1000 +     # memory cues
            hash(str(particle.metadata.get("snapshot-post_message", {}))) % 1000
        )
        random.seed(seed_base)

        # Generate expression using experiential character-based learning (original approach)
        expression_parts = []
        
        # Decide on expression style based on particle state
        valence = to_numpy(particle.position[8])
        intent = to_numpy(particle.position[10])
        emotional_rhythm = to_numpy(particle.position[6])
        
        # Generate expression length based on activation (like original)
        expression_length = max(3, int(particle.activation * 15) + 2)  # 3-17 characters
        
        # Create dynamic alphabet combining learned vocabulary with character exploration
        dynamic_alphabet = list(self.alphabet["letters"]) + list(self.alphabet["punctuation"])
        if len(word_vocabulary) > 0:
            dynamic_alphabet.extend(word_vocabulary[:10])  # Add some learned words to alphabet
        
        # Build expression character by character with experiential learning
        for i in range(expression_length):
            # Use particle positions to influence character selection (original experiential approach)
            pos_influence = abs(to_numpy(particle.position[i % len(particle.position)]))
            
            # Character selection probabilities based on learning state
            if len(word_vocabulary) > 5 and random.random() < (0.3 + valence * 0.3):
                # 30-60% chance: Use learned vocabulary (influenced by positive experiences)
                selected_word = random.choice(word_vocabulary)
                expression_parts.append(selected_word)
                # Skip ahead to avoid too many words
                i += max(1, len(selected_word) // 3)
            elif random.random() < (0.2 + intent * 0.2):
                # 20-40% chance: Character sequences (original experiential approach)
                char_length = max(1, int(pos_influence * 4) + 1)  # 1-5 characters
                char_sequence = ""
                for j in range(char_length):
                    pos_index = (i + j) % len(particle.position)
                    char_seed = int(abs(to_numpy(particle.position[pos_index])) * len(self.alphabet["letters"])) % len(self.alphabet["letters"])
                    char_sequence += self.alphabet["letters"][char_seed]
                expression_parts.append(char_sequence)
            elif random.random() < 0.15:
                # 15% chance: Single punctuation (emotional expression)
                if valence > 0.7:
                    expression_parts.append(random.choice("!?"))  # Excited
                elif valence < 0.3:
                    expression_parts.append(random.choice(".,"))  # Calm
                else:
                    expression_parts.append(random.choice(self.alphabet["punctuation"]))
            elif random.random() < 0.1:
                # 10% chance: Single letters (primitive sound exploration)
                letter_index = int(pos_influence * len(self.alphabet["letters"])) % len(self.alphabet["letters"])
                expression_parts.append(self.alphabet["letters"][letter_index])
            else:
                # Remaining: Space or pause for natural rhythm
                if random.random() < 0.3:
                    expression_parts.append(" ")
                else:
                    expression_parts.append("<PAUSE>")
        
        # Join parts naturally with experiential spacing
        raw_expression = ""
        for k, part in enumerate(expression_parts):
            if k > 0 and part not in ["<PAUSE>", " "] and expression_parts[k-1] not in ["<PAUSE>", " "]:
                # Add natural spacing between meaningful parts
                if part not in self.alphabet["punctuation"] and len(part) > 1:
                    raw_expression += " "
            if part != "<PAUSE>":  # Filter out pause markers from final output
                raw_expression += str(part)
        
        # Clean up multiple spaces and ensure reasonable output
        phrase = " ".join(raw_expression.split())  # Normalize whitespace
        if not phrase or phrase.isspace():
            phrase = random.choice(word_vocabulary) if word_vocabulary else "..."
        
        # Limit final length
        if len(phrase) > 100:
            phrase = phrase[:97] + "..."
        
        self.learn(phrase, context="generated-expression")
        return phrase
    

    def generate_random_expression(self, length=12):
        return ''.join(random.choices(self.flat_alphabet, k=length))

    def learn(self, text, context=None):
        """
        Iris processes external text, adds tokens to her lexicon, and links context.
        """
        tokens = self.custom_tokenizer(text)


        # pulling definitions and classifications
        for token in tokens:
            if not self.lexicon_store.has_term(token):
                definition = self.define_term(token, text)
                classification = classify_term(token, context=context)
                self.lexicon_store.add_term(token, full_phrase=text, definition=definition, context="external:spaCy", source="spaCy", term_type=classification["type"], tags=classification["tags"], intent=classification["intent"])
                

            else: 
                classification = classify_term(token, context=context)
                self.lexicon_store.add_term(token, full_phrase=tokens, definition=None, context=context or "learn", source="Iris", term_type=classification["type"], tags=classification["tags"], intent=classification["intent"])


            

        # store learning as memory
        reflection = f"I learned {len(tokens)} new symbols."
        # Convert to string for memory storage
        learning_summary = f"Action: learn, Reflection: {reflection}, Tokens count: {len(tokens)}, Timestamp: {time.time()}"
        
        if self.memory_manager:
            self.memory_manager.store_memory("memory", f"learn-{int(time.time())}", learning_summary)
        else:
            add_entry("memory", f"learn-{int(time.time())}", learning_summary)

        self.log(f"[Learn] Added {len(tokens)} terms to lexicon.")

    def define_term(self, term, phrase):
        defs = get_external_definitions(term)
        final_def, sources_used = self.compare_and_merge_definitions(defs)

        self.lexicon_store.add_term(term, phrase, definition = final_def, context = "aggregated_external", source = "multi-source: {sources_used}")
        self.reflect_on_def(term, sources_used)

        if not final_def:
            final_def = "Definition unavailable."

        return final_def

    def custom_tokenizer(self, text):
        tokens = []
        word = ''
        for char in text:
            if char in string.whitespace:
                if word:
                    tokens.append(word)
                    word = ''
            elif char in string.punctuation:
                if word:
                    tokens.append(word)
                    word = ''
                tokens.append(char)
            else:
                word += char
        if word:
            tokens.append(word)
        return tokens
    
    def learn_reflection(self, reflection_text):
        self.learn(reflection_text, context="self-reflection")

    def reflect_on_def(self, term, sources):
        summary = f"I encountered the term '{term}'. SpaCy defines it as: {sources.get('spacy')}"

        if self.memory_manager:
            self.memory_manager.store_memory("reflection", f"reflect-def-{term}", summary,
                term=term,
                sources=str(sources),
                timestamp=time.time()
            )
        else:
            add_entry("reflection", f"reflect-def-{term}", summary,
                term=term,
                sources=str(sources),
                timestamp=time.time()
            )

        if self.log_callback:
            self.log_callback(f"[Reflect] {summary}")

    def compare_and_merge_definitions(self, def_dict):
        """
        Takes multiple definitions and returns a best-guess summary.
        """
        ranked_defs = sorted(def_dict.items(), key=lambda item: len(item[1] or ""), reverse=True)
        sources_used = {k: v for k, v in ranked_defs if v}

        if not sources_used:
            return None, {}

        # Naive merge for now
        best_source, best_def = next(iter(sources_used.items()))
        return best_def, sources_used


