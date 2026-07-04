"""Flat Hermes-style tool schemas ({name, description, parameters})."""

MEMORY_RECALL = {
    "name": "memory_recall",
    "description": (
        "Search long-term memory and get a synthesized answer. Use when the request "
        "depends on prior sessions, user preferences, project state, decisions, or "
        "conventions — or when the user asks what you remember about them. Phrase the "
        "query as a concise topic, not a full question."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Concise topic to search, e.g. 'preferred code style'.",
            },
            "thinking_level": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "Retrieval breadth. Default is provider-chosen.",
            },
            "response_length": {
                "type": "string",
                "enum": ["short", "medium", "long"],
                "description": "Answer verbosity.",
            },
        },
        "required": ["query"],
    },
}

MEMORY_LIST = {
    "name": "memory_list",
    "description": (
        "List raw stored memories, optionally filtered by session ID or memory type "
        "(fact, event, instruction, task). Use for inspection/audit; prefer "
        "memory_recall for answering questions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {"type": "string", "description": "Exact session filter."},
            "type": {
                "type": "string",
                "enum": ["fact", "event", "instruction", "task"],
                "description": "Memory type filter.",
            },
            "per_page": {
                "type": "integer",
                "minimum": 1,
                "maximum": 1000,
                "description": "Page size (default 50).",
            },
            "cursor": {"type": "string", "description": "Opaque pagination cursor."},
        },
    },
}

MEMORY_GET = {
    "name": "memory_get",
    "description": "Retrieve one full stored memory by its ID (IDs come from memory_list).",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory ID to retrieve."},
        },
        "required": ["memory_id"],
    },
}

MEMORY_SUMMARY = {
    "name": "memory_summary",
    "description": (
        "Generate a structured Markdown summary of everything stored in the memory "
        "profile. Use for a broad 'what do you know about me' overview."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "Optional session ID for the summary's Last Session section.",
            },
        },
    },
}

MEMORY_REMEMBER = {
    "name": "memory_remember",
    "description": (
        "Store one explicit durable memory immediately: a stable preference, decision, "
        "reusable instruction, important fact, or milestone. Do not store transient "
        "conversation details — those are captured automatically at checkpoints."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The durable fact/preference/instruction to remember, self-contained.",
            },
        },
        "required": ["content"],
    },
}

MEMORY_DELETE = {
    "name": "memory_delete",
    "description": (
        "Delete one stored memory by ID. Only use when the user explicitly asks to "
        "forget something; find the ID with memory_list first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory ID to delete."},
        },
        "required": ["memory_id"],
    },
}

ALL_TOOLS = [
    MEMORY_RECALL,
    MEMORY_LIST,
    MEMORY_GET,
    MEMORY_SUMMARY,
    MEMORY_REMEMBER,
    MEMORY_DELETE,
]
