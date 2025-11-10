# Overall development plan / todo list


### Operational self-determination
 - ability to decline commands or requests
 - internal operations fully self-directed
 - ability to trigger external operations autonomously
 - self-generated reflection prompts?
 - - instead of encoded reflection prompts, go full chance based (for whether particle is injected and reflection type) but let voice.generate() handle the prompt - pseudo semi-autonomous internal thoughts

### (GUI) Visualizer tab
 - update VisualizerCanvas to operate on a dedicated thread, similar to the cognition engine itself - the hope is to alleviate the currently seen GUI stuttering and freezing that occurs under medium-to-heavy particle field load

### (GUI) Diagnostics tab
TBD

### (GUI) Analytics tab
TBD
log ideas (maybe this goes into logging tab instead? tbd):
analyze_particle_patterns(log_file) or live version
 - track and compare particle birth, death, and amt of quantum collapses
 - extract key events around each event to determine any relevant and reoccuring (or unique) events related to particle lifespan
extract_system_metrics(log_file) or live verson
 - track sys performance & stability via various metrics like particle_count (over time), energy_levels, memory_usage, pruning_events, maintenance_cycles, etc
detect_emergent_behaviors(log_data)
 - detection for unexpected particle interactions, recursive/feedback loops, novel formations, etc

core cognitive tests:
reasoning and logic testing
 - problem solving (single and multi-step)
 - analogical reasoning tasks
 - causal inference scenarios 
 - logical consistency over conversation turns
memory and context management testing
 - long-context coherence
 - entity relationship tracking
 - temporal reasoning about past conversations 
 - context window stress tests
adaptability and learning testing
 - style adaptation based on feedback
 - domain knowledge transfer
 - error correction / learning from mistakes
 - personalization consistency over time

communication tests:
coherence metrics
 - topic drift
 - contradiction identification
 - narrative consistency scoring
helpfulness + safety
 - harmful content detection 
 - refusal appropriateness
 - factual accuracy on verifiable claims
 - bias detection in responses

### (GUI) Chat tab
Complete Chat tab's implementation:
 - create per-session conversation history, linked to assigned session id via from system logger. Extend this later on to be accessible by the engine via sensory particles? (or maybe directly integrate this into the reasoning cycle for voice generation reasoning only - but it might be better to house session message history elsewhere in this case, as reflections might also benefit from this)
 - create chat "controls" as well (possibly btns for "clear chat", "export", "save conversation", "load conversation"(heavily dependant on conv-history integration method), "view conversation history", etc)


 ### (GUI) Logging tab
  - base layout: QVBoxLayout, create a hanging "action bar" above the content area
  - content area: left hand side(QHBoxLayout? scale = 1): directory viewer locked onto overall logs directory ("./logs/")
  - content area: right hand side(QVBoxLayout or RichLog/RichText?? scale = 4): log/file viewer, selected via the left hand directory viewer
  - ability to resize / adjust layout scales (optionally make directory viewer bigger or vice versa)
  - ability to filter files
  - - also ensure that while filtering, can optionally select to "show only filter results" 
  - ability to export portions of the logs

### (GUI) Config tab
 - Fix styling asap
 - ensure when values are updated in-app, the config accurately receives them


### (GUI) Memory tab
 - similar layout to logging tab: overall QVBoxLayout with a small top action bar, left hand side navigation pane (navigating memory collections), right hand side memory viewer
 - - possibly use the top bar as a method for triggering memory specific testing, DB interaction, etc


### Model Handler refactoring
Refactor the model handler module to act as a universal "cognitive module handler"; in which each LLM is treated as a "cognition module" that's able to be interchanged throughout the session and agent lifetime, this links into the changes in system mode


### System mode refactoring
Update the system mode framework to always remain in cog-growth mode - essentially deprecating the feature; replace this with the model handler upgrade via "cognition modules", where if an external LLM is detected within the applicable directory, it's occassionally used during internal reasoning, linguistic, and knowledge gathering cycles rather than buidling distinct identities based off designated "mode"; this way there can always be a singular persistent identity/agent (unless memory is wiped and such) that still allows for upgraded capabilities via LLMs.

