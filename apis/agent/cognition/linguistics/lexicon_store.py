"""
handles linguistic processing - processing inputs and internal thoughts and parses token(s) and phrases for definition 
"""
import uuid
from datetime import datetime

from apis.api_registry import api


class LexiconStore:
    def __init__(self):
        self.logger = api.get_api("logger")
        self.lexicon = {}
        self.collection_name = "lexicon"

    async def migrate_legacy_entries(self):
        """
        Migrates old in-memory lexicon entries into ChromaDB, checking for duplicates.
        """
        for token, entry in self.lexicon.items():
            existing = await self.memory_matrix.query(
                collection=self.collection_name,
                key={"token": token}
            )
            if not existing:
                await self._insert_entry_to_chroma(token, entry)

    async def _insert_entry_to_chroma(self, token, entry):
        entry_data = {
            "token": token,
            "id": entry.get("id", "lex-" + str(uuid.uuid4()) + "-entry"),
            "definitions": entry.get("definitions", []),
            "contexts": entry.get("contexts", []),
            "sources": entry.get("sources", []),
            "times_encountered": entry.get("times_encountered", 1),
            "term_origin": entry.get("term_origin"),
            "last_seen": entry.get("last_seen", datetime.now().isoformat()),
            "intent": entry.get("intent"),
            "type": entry.get("type"),
            "tags": entry.get("tags", []),
            "scope": entry.get("scope", "external"),
            "source_particle_id": entry.get("source_particle_id"),
            "strength": entry.get("strength", 1.0),
            "decay": entry.get("decay", 0.97),
            "context_summary": entry.get("context_summary", ""),
        }

        await self.memory_matrix.update(
            key=token,
            value=entry_data,
            source="lexicon",
            tags= entry.get("tags", []),
            collection=self.collection_name
        )

    async def add_term(self, token, full_phrase=None, definitions=None, context=None,
                       source="unknown", intent=None, term_type=None, tags=None,
                       scope="external", particle_id=None, particle_embedding=None):

        full_phrase = full_phrase or token
        entry_id = "lex-" + str(uuid.uuid4()) + "-entry"

        new_entry = {
            "token": token,
            "id": entry_id,
            "definitions": [],
            "contexts": [context] if context else [],
            "sources": [source] if source else [],
            "times_encountered": 1,
            "term_origin": full_phrase,
            "last_seen": datetime.now().isoformat(),
            "intent": intent,
            "type": term_type,
            "tags": tags or [],
            "scope": scope,
            "source_particle_id": particle_id,
            "strength": 1.0,
            "decay": 0.97,
            "context_summary": "",
        }

        if definitions:
            if not isinstance(definitions, list):
                definitions = [definitions]
            for definition in definitions:
                new_entry["definitions"].append({
                    "text": definition,
                    "source": source,
                    "timestamp": datetime.now().isoformat()
                })

        # Try to fetch and merge with an existing lexicon entry if exists
        existing = await self.memory_matrix.query(
            collection=self.collection_name,
            key={"token": token}
        )

        if existing:
            existing = existing[0]  # first matched doc
            for key in ["contexts", "sources", "tags"]:
                new_entry[key] = list(set(existing.get(key, []) + new_entry.get(key, [])))
            new_entry["definitions"] = existing.get("definitions", []) + new_entry.get("definitions", [])
            new_entry["times_encountered"] = existing.get("times_encountered", 0) + 1
            new_entry["last_seen"] = datetime.now().isoformat()
            new_entry["id"] = existing.get("id", entry_id)

        updated_tags = tags.copy() if tags else []
        updated_tags.extend(["lingual", f"origin: {token}"])

        # Use memory bank API instead of direct memory_matrix
        memory_api = api.get_api("memory_bank")
        if memory_api:
            await memory_api.update(
                key=entry_id,
                value=new_entry,
                source=source,
                tags=updated_tags,
                memory_type="lexicon"
            )
            
        # Use adaptive engine API for embeddings
        if particle_embedding:
            adaptive_api = api.get_api("adaptive_engine")
            if adaptive_api:
                adaptive_api.set_embedding(entry_id, particle_embedding)
        


    async def get_term(self, token):
        result = await self.memory_matrix.query(
            collection=self.collection_name,
            key={"token": token}
        )
        return result[0] if result else None

    def get_term_id(self, term):
        # returning the unique ID for the given term, if it's available; else returning an exception
        if term not in self.lexicon:
            self.log(f"Term not found in lexicon: {term} | skipping term")
            return None
        else:
            return self.lexicon[term]["id"]

# Register the API
api.register_api("lexicon_store", LexiconStore())
