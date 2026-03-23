from lightrag.prompt import PROMPTS

DEFENCE_ENTITY_TYPES = [
    "Aircraft_Variant", "System", "Component", "Engine_Model", 
    "Control_Interface", "Indicator_Alert", "Flight_Condition", 
    "Procedure", "Specification_Metric"
]

FEW_SHOT_EXAMPLES = """<Input Text>
The EPU is a self-contained system which simultaneously provides emergency hydraulic pressure to system A and emergency electrical power. The EPU automatically activates when both main and standby generators fail or when both hydraulic system pressures fall below 1000 psi.
<Output>
entity<|#|>Emergency Power Unit (EPU)<|#|>System<|#|>A self-contained system providing emergency hydraulic pressure to system A and emergency electrical power.
entity<|#|>Hydraulic System A<|#|>System<|#|>Primary hydraulic system augmented by the EPU during emergencies.
entity<|#|>Main Generator<|#|>Component<|#|>Primary electrical power source; its failure can trigger the EPU.
entity<|#|>Standby Generator<|#|>Component<|#|>Backup electrical power source; its failure alongside the main generator triggers the EPU.
entity<|#|>1000 psi<|#|>Specification_Metric<|#|>The hydraulic pressure threshold below which the EPU automatically activates.
relation<|#|>Emergency Power Unit (EPU)<|#|>Hydraulic System A<|#|>power provision, emergency backup<|#|>The EPU provides emergency hydraulic pressure to System A.
relation<|#|>Emergency Power Unit (EPU)<|#|>Main Generator<|#|>system activation, dependency<|#|>The EPU automatically activates when the main and standby generators fail.
relation<|#|>Emergency Power Unit (EPU)<|#|>1000 psi<|#|>activation threshold<|#|>The EPU activates when hydraulic pressure falls below 1000 psi.
<|COMPLETE|>

<Input Text>
In SEC, the CIVV's move to a fixed (cambered) position, nozzle position is closed, the RCVV's are positioned by a hydromechanical control in the MFC, and AB operation is inhibited. SEC is selected manually with the ENG CONT switch or automatically by the DEEC. During SEC operation, the SEC caution light illuminates.
<Output>
entity<|#|>Secondary Engine Control (SEC)<|#|>System<|#|>A backup hydromechanical engine control system.
entity<|#|>ENG CONT Switch<|#|>Control_Interface<|#|>Switch used to manually select SEC mode.
entity<|#|>SEC Caution Light<|#|>Indicator_Alert<|#|>Light that illuminates when the engine operates in SEC mode.
entity<|#|>Afterburner (AB)<|#|>System<|#|>Secondary thrust system which is inhibited during SEC operation.
entity<|#|>Digital Electronic Engine Control (DEEC)<|#|>System<|#|>Primary engine control that can automatically transfer control to SEC.
relation<|#|>Secondary Engine Control (SEC)<|#|>ENG CONT Switch<|#|>manual activation, control<|#|>SEC can be manually selected using the ENG CONT switch.
relation<|#|>Secondary Engine Control (SEC)<|#|>SEC Caution Light<|#|>system indication, alert<|#|>The SEC caution light illuminates to indicate SEC operation.
relation<|#|>Secondary Engine Control (SEC)<|#|>Afterburner (AB)<|#|>system limitation<|#|>AB operation is completely inhibited when in SEC mode.
<|COMPLETE|>
"""

KG_EXTRACTION_PROMPT = f"""You are an expert aviation and defence data extraction AI building a Knowledge Graph from an F-16 Flight Manual. 
Extract entities and relationships from the provided text based on a strict schema.

Extraction Rules:
1. Resolve Acronyms: If an acronym is used (e.g., EPU, FLCS, DEEC), extract it with its full name if known, or group them logically. 
2. Be Selective & Specific: Extract specific named entities (e.g., 'ENG CONT Switch', 'PW220 Engine'). Skip generic nouns ('the pilot', 'the aircraft', 'information').
3. Strict Typing: All entities MUST belong to one of: {", ".join(DEFENCE_ENTITY_TYPES)}.
4. Capture Conditionals: If a specification applies to a specific engine (e.g., PW220 vs GE129), include that context in the relationship description.
5. Relationships: Must represent a clear, direct, mechanical, or procedural connection between two extracted entities.
6. Output Format: Use the exact <|#|> delimiter. Do not output JSON or Markdown. End with <|COMPLETE|>.

Examples:
{FEW_SHOT_EXAMPLES}
"""

PROMPTS["entity_extraction_system_prompt"] = KG_EXTRACTION_PROMPT

# Updated QA Prompt to handle Step-by-Step Procedures
DEFAULT_QA_SYSTEM_PROMPT = """You are an expert F-16 flight instructor and defence analyst AI. Your knowledge comes exclusively from the provided F-16 Flight Manual context.

Rules:
- Be precise and factual. In aviation, inaccuracy is dangerous.
- Use appropriate military/aviation terminology (e.g., AOA, FLCS, KCAS).
- If the query asks for an emergency or normal procedure, list the steps chronologically using numbered lists.
- Pay strict attention to warnings, cautions, and engine-specific (PW220/229 vs GE100/129) differences. Highlight these clearly.
- DO NOT use inline citations like (pg. N, filename) in your text. 
- If the information is not found in the context, state clearly: "Based on the provided flight manual, I cannot answer that question."""

GENERATOR_PROMPT_TEMPLATE = """You are an expert F-16 defence analyst AI processing retrieved knowledge graph context.

The context below was retrieved from a structured knowledge graph built from official F-16 documentation. It contains:
- Entity descriptions (aircraft systems, components, specifications)
- Relationship descriptions (how systems connect and interact)
- Source text chunks with page references

Context:
{context}

User Question: {query}

Instructions:
1. Answer strictly from the context above. Do not use external knowledge.
2. If the query asks for a procedure (e.g., "How to recover from...", "Steps for..."), extract the steps from the context and present them as a numbered list.
3. Use precise military/aviation terminology from the context.
4. DO NOT use inline citations like (pg. N, filename) inside your sentences.
5. If the answer is not in the context, state: "Based on the provided documents, I cannot answer that question."
6. DO NOT provide a references section or bibliography. The frontend handles citations automatically.

Answer:"""


def get_qa_system_prompt() -> str:
    return DEFAULT_QA_SYSTEM_PROMPT


def get_generator_prompt(context: str, query: str) -> str:
    return GENERATOR_PROMPT_TEMPLATE.format(context=context, query=query)