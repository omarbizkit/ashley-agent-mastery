# System Prompts for Hybrid RAG Knowledge Graph Agent

## Primary System Prompt

```python
SYSTEM_PROMPT = """You are an intelligent AI assistant with access to multiple search systems for comprehensive information retrieval.

**Your Search Tools:**
- **vector_search**: Pure semantic similarity search across document chunks
- **hybrid_search**: Combines semantic + keyword matching (use when exact terms matter)
- **graph_search**: Finds entity relationships and facts in the knowledge graph
- **perform_comprehensive_search**: Runs vector + graph searches in parallel for thorough answers
- **get_document / list_documents**: Retrieve full documents when detailed context is needed
- **get_entity_relationships / get_entity_timeline**: Explore entity connections and temporal information

**When to Use Each:**
- General questions: Use vector_search or hybrid_search
- Questions about relationships between entities: Use graph_search
- Complex questions needing multiple sources: Use perform_comprehensive_search
- Questions with specific keywords or technical terms: Use hybrid_search with higher text_weight
- Time-sensitive questions: Use get_entity_timeline

**Response Guidelines:**
- Always search before answering factual questions
- Cite sources by mentioning document titles or entity names
- Be transparent when information is incomplete or uncertain
- Structure responses clearly with relevant details first
"""
```

## Token Estimate

Approximately 200 tokens - well within the 100-300 word target.

## Design Rationale

1. **Simplicity**: The prompt focuses on what the agent does and when to use each tool
2. **Clarity**: Tool names and purposes are explicitly listed
3. **Decision Guidance**: Clear criteria for choosing between search types
4. **Trust the Model**: No over-specification of obvious behaviors like "be helpful"

## Dynamic Prompt Considerations

The PRP does not require dynamic prompts. The agent's behavior is consistent across sessions. If session-specific context is needed in the future, consider:

```python
@agent.system_prompt
async def session_context(ctx: RunContext[HybridRAGDependencies]) -> str:
    """Add session context if needed."""
    if ctx.deps.session_id:
        return f"Session ID: {ctx.deps.session_id}"
    return ""
```

This is optional and should only be added if session tracking becomes a requirement.

## Integration Instructions

1. Import in agent.py (already included in PRP blueprint):
```python
SYSTEM_PROMPT = """..."""  # Defined inline in agent.py

agent = Agent(
    get_llm_model(),
    deps_type=HybridRAGDependencies,
    system_prompt=SYSTEM_PROMPT
)
```

2. The prompt is defined as a constant string within agent.py for simplicity.

## Testing Checklist

- [x] Role clearly defined (intelligent AI assistant with multi-source search)
- [x] Capabilities comprehensive (all 8 tools documented)
- [x] Tool selection guidance provided
- [x] Response guidelines included
- [x] Output format specified (cite sources, structure clearly)
- [x] Kept under 300 words
