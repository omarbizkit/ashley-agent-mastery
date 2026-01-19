#!/usr/bin/env python3
"""Conversational CLI with real-time streaming and tool call visibility."""

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
import openai

from pydantic_ai import Agent
from pydantic_ai.messages import (
    PartStartEvent,
    PartDeltaEvent,
    TextPartDelta,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    FinalResultEvent,
)
from agent import agent
from dependencies import HybridRAGDependencies, SearchPreferences
from settings import settings
from utils.db_utils import initialize_database, close_database
from utils.graph_utils import initialize_graph, close_graph

console = Console()


async def stream_agent_interaction(user_input: str, deps: HybridRAGDependencies) -> str:
    """Stream agent interaction with real-time tool call display."""

    try:
        async with agent.iter(user_input, deps=deps) as run:
            async for node in run:
                if Agent.is_user_prompt_node(node):
                    pass

                elif Agent.is_model_request_node(node):
                    console.print("[bold blue]Assistant:[/bold blue] ", end="")
                    async with node.stream(run.ctx) as request_stream:
                        async for event in request_stream:
                            if isinstance(event, PartDeltaEvent):
                                if isinstance(event.delta, TextPartDelta):
                                    console.print(event.delta.content_delta, end="")
                            elif isinstance(event, FinalResultEvent):
                                console.print()

                elif Agent.is_call_tools_node(node):
                    async with node.stream(run.ctx) as tool_stream:
                        async for event in tool_stream:
                            if isinstance(event, FunctionToolCallEvent):
                                console.print(f"  [cyan]Calling:[/cyan] [bold]{event.part.tool_name}[/bold]")
                            elif isinstance(event, FunctionToolResultEvent):
                                result_preview = str(event.result.content)[:100]
                                console.print(f"  [green]Result:[/green] [dim]{result_preview}...[/dim]")

                elif Agent.is_end_node(node):
                    pass

        return run.result.output

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return f"Error: {e}"


async def main():
    """Main conversation loop."""

    welcome = Panel(
        "[bold blue]Hybrid RAG Knowledge Graph Agent[/bold blue]\n\n"
        "[green]Vector + Hybrid + Graph search with streaming[/green]\n"
        "[dim]Type 'exit' to quit[/dim]",
        style="blue",
        padding=(1, 2)
    )
    console.print(welcome)
    console.print()

    # Initialize connections
    console.print("[dim]Initializing database and graph connections...[/dim]")
    await initialize_database()
    await initialize_graph()
    console.print("[green]Ready![/green]\n")

    # Create dependencies
    embedding_client = openai.AsyncOpenAI(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key
    )
    deps = HybridRAGDependencies(
        embedding_client=embedding_client,
        embedding_model=settings.embedding_model,
        search_preferences=SearchPreferences()
    )

    try:
        while True:
            try:
                user_input = Prompt.ask("[bold green]You").strip()

                if user_input.lower() in ['exit', 'quit']:
                    console.print("\n[yellow]Goodbye![/yellow]")
                    break

                if not user_input:
                    continue

                await stream_agent_interaction(user_input, deps)
                console.print()

            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' to quit[/yellow]")
                continue
    finally:
        await close_database()
        await close_graph()
        await embedding_client.close()


if __name__ == "__main__":
    asyncio.run(main())
