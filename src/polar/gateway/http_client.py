"""Request-owned HTTP clients, including cancellation-safe socket cleanup."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from functools import lru_cache

import httpx


@lru_cache(maxsize=1)
def _ssl_context():
    # Keep HTTPX's default certificate/env handling, but avoid blocking the event
    # loop to reload CA certificates for every request-owned client.
    return httpx.create_ssl_context()


def new_client(**kwargs) -> httpx.AsyncClient:
    return httpx.AsyncClient(verify=_ssl_context(), **kwargs)


async def close_client(client: httpx.AsyncClient) -> None:
    # Session deletion and policy cutoff can cancel the same task repeatedly.
    # Finish closing its sockets before propagating another cancellation.
    closing = asyncio.create_task(client.aclose())
    cancelled = False
    while not closing.done():
        try:
            await asyncio.shield(closing)
        except asyncio.CancelledError:
            cancelled = True
    closing.result()
    if cancelled:
        raise asyncio.CancelledError


@asynccontextmanager
async def closing_client(client: httpx.AsyncClient):
    try:
        yield client
    finally:
        await close_client(client)
