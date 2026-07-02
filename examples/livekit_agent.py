"""LiveKit Agents example with TraceRoot instrumentation.

Required environment:
    TRACEROOT_API_KEY
    LIVEKIT_URL
    LIVEKIT_API_KEY
    LIVEKIT_API_SECRET

Run:
    python examples/livekit_agent.py
"""

from __future__ import annotations

import os

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, inference
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

import traceroot
from traceroot import Integration, using_attributes


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful voice AI assistant.",
            llm=inference.LLM(model="openai/gpt-5.2-chat-latest"),
        )


server = AgentServer()


@server.rtc_session(agent_name="traceroot-livekit-example")
async def entrypoint(ctx: JobContext) -> None:
    traceroot.initialize(
        api_key=os.environ["TRACEROOT_API_KEY"],
        integrations=[Integration.LIVEKIT],
    )
    ctx.add_shutdown_callback(traceroot.flush)

    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="multi"),
        tts=inference.TTS(model="cartesia/sonic-3"),
        turn_detection=MultilingualModel(),
        vad=silero.VAD.load(),
        preemptive_generation=True,
    )

    with using_attributes(session_id=ctx.room.name):
        await session.start(
            agent=Assistant(),
            room=ctx.room,
            record={"traces": False},
        )
        await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
