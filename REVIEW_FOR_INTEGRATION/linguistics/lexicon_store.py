import os
from datetime import datetime

class LexiconStore:
    def __init__(self, path="./data/memory/lexicon.json", engine=None):
        self.path = path  # Keep for backward compatibility
        self.engine = engine  # Memory particle gateway access
        self.lexicon = {}
        self.load()

    def load(self):
        """Load lexicon from vector database via memory particle gateway"""
        try:
            if self.engine:
                # Use memory particle gateway
                from core.particles.memory_particle import MemoryParticle
                results = MemoryParticle.memory_query(
                    engine=self.engine,
                    memory_type="lexicon",
                    query_text="*",
                    n_results=10000
                )
            else:
                # Fallback to direct access for backward compatibility
                from core.memory.mem_DB import query_entries
                results = query_entries("lexicon", "*", n_results=10000)
                
            self.lexicon = {}
            
            if results and 'documents' in results:
                for doc_list in results['documents']:
                    for doc in doc_list:
                        if doc:
                            # Create a basic entry structure from the document
                            self.lexicon[doc] = {
                                "token": doc,
                                "definitions": [],
                                "contexts": [],
                                "sources": [],
                                "times_encountered": 1,
                                "term_origin": doc,
                                "last_seen": None,
                                "intent": None,
                                "type": None,
                                "tags": []
                            }
            
            gateway_msg = "via memory particle gateway" if self.engine else "direct access (no engine)"
            print(f"[LexiconStore] Loaded {len(self.lexicon)} terms from vector database ({gateway_msg})")
        except Exception as e:
            print(f"[LexiconStore] Could not load from vector database: {e}")
            # Fallback to JSON if vector DB fails
            if os.path.exists(self.path):
                import json
                with open(self.path, "r") as f:
                    self.lexicon = json.load(f)

    def save(self, tokens=None):
        """Save lexicon to vector database"""
        try:
            # Vector database saves automatically, but we can trigger a backup
            print(f"[LexiconStore] Lexicon data persisted in vector database")
        except Exception as e:
            print(f"[LexiconStore] Vector database save error: {e}")
            # Fallback to JSON
            import json
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w") as f:
                json.dump(self.lexicon or tokens, f, indent=2)

    def add_term(self, token, full_phrase, definition=None, context=None, source="unknown", intent=None, term_type=None, tags=None):
        
        #lexicon format
        entry = self.lexicon.get(token, {
            "token": token,
            "definitions": [],
            "contexts": [],
            "sources": [],
            "times_encountered": 0,
            "term_origin": None,
            "last_seen": None,
            "intent": intent,
            "type": term_type,
            "tags": []
        })

        #store new definitions
        if definition:
            already_defined = any(d["text"] == definition for d in entry["definitions"])
            if not already_defined:
                entry["definitions"].append({
                    "text": definition,
                    "source": source,
                    "timestamp": datetime.now().isoformat()
                })

        #updating intent if currently known
        if intent:
            entry["intent"] = intent
        
        #appending newly available context and sources
        if context and context not in entry["contexts"]:
            entry["contexts"].append(context)
        if source and source not in entry["sources"]:
            entry["sources"].append(source)
        
        #updating usage stats
        entry["times_encountered"] += 1
        entry["last_seen"] = datetime.now().isoformat()

        #auto-setting origin info if first time phrase is detected
        if not entry["term_origin"]:
            entry["term_origin"] = full_phrase or "unknown"

        #saving and updating
        self.lexicon[token] = entry
        
        # Store in vector database with simple text content
        try:
            # Create a simple text representation for vector search
            content_text = f"{token}: {definition or 'No definition'}"
            if context:
                content_text += f" (Context: {context})"
            
            if self.engine:
                # Use memory particle gateway
                from core.particles.memory_particle import MemoryParticle
                MemoryParticle.memory_entry(
                    engine=self.engine,
                    memory_type="lexicon",
                    key=f"lexicon-{token}",
                    content=content_text,
                    source=source or "unknown",
                    term_type=term_type or "word",
                    intent=intent or "unknown"
                )
            else:
                # Fallback to direct access for backward compatibility
                from core.memory.mem_DB import add_entry
                add_entry("lexicon", f"lexicon-{token}", content_text,
                         source=source or "unknown",
                         term_type=term_type or "word", 
                         intent=intent or "unknown")
        except Exception as e:
            gateway_msg = "via memory particle gateway" if self.engine else "direct access"
            print(f"[LexiconStore] Vector database error ({gateway_msg}): {e}")
            
        #self.save()

    def get_terms(self):
        return list(self.lexicon.keys())

    def has_term(self, term):
        if term in self.lexicon:
            return term in self.lexicon
        else:
            return None
        
    def reassess_lexicon(self):
        for token in self.get_terms():
            definition = self.define_term(token)
            