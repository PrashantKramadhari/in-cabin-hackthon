"""Minimal async pub/sub bus.

Sensor nodes publish events; the fusion engine and the websocket layer
subscribe. Mirrors an automotive signal bus so the same topology maps onto an
embedded SDV (software-defined vehicle) deployment.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import AsyncIterator


class Bus:
    def __init__(self) -> None:
        self._subs: dict[str, list[asyncio.Queue]] = defaultdict(list)

    def subscribe(self, topic: str, maxsize: int = 32) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._subs[topic].append(q)
        return q

    async def publish(self, topic: str, payload: dict) -> None:
        for q in self._subs[topic]:
            if q.full():
                # drop oldest to keep latency bounded (<200ms goal)
                _ = q.get_nowait()
            await q.put(payload)

    async def stream(self, topic: str) -> AsyncIterator[dict]:
        q = self.subscribe(topic)
        while True:
            yield await q.get()


# single shared bus instance for the app
bus = Bus()
