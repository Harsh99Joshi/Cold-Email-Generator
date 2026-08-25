"""Shims so RAGAS 0.2 can run on current langchain-community and Python 3.14."""

import asyncio
import sys
import types


async def _wait_for_without_timeout(aw, timeout=None):
    # nest_asyncio + Python 3.14 raises: Timeout should be used inside a task
    return await aw


def apply():
    asyncio.wait_for = _wait_for_without_timeout

    if "langchain_community.chat_models.vertexai" not in sys.modules:
        vertexai = types.ModuleType("langchain_community.chat_models.vertexai")

        class ChatVertexAI:
            pass

        vertexai.ChatVertexAI = ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = vertexai

    import langchain_community.llms as community_llms

    if not hasattr(community_llms, "VertexAI"):
        class VertexAI:
            pass

        community_llms.VertexAI = VertexAI
