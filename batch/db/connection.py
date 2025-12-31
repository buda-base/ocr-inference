from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import asyncpg

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from asyncpg.pool import PoolConnectionProxy


@dataclass
class _PoolHolder:
    pool: asyncpg.Pool | None = field(default=None)


_holder = _PoolHolder()


def get_database_url() -> str:
    return os.environ.get("DATABASE_URL", "postgresql://localhost:5432/ocr_batch")


async def init_pool(
    dsn: str | None = None,
    min_size: int = 1,
    max_size: int = 10,
) -> asyncpg.Pool:
    if _holder.pool is not None:
        return _holder.pool

    async def connection_init(conn: asyncpg.Connection) -> None:
        await conn.execute("SELECT 1")

    _holder.pool = await asyncpg.create_pool(
        dsn or get_database_url(),
        min_size=min_size,
        max_size=max_size,
        max_inactive_connection_lifetime=60.0,
        init=connection_init,
    )
    return _holder.pool


async def get_pool() -> asyncpg.Pool:
    if _holder.pool is None:
        raise RuntimeError("Database pool not initialized. Call init_pool() first.")
    return _holder.pool


async def close_pool() -> None:
    if _holder.pool is not None:
        await _holder.pool.close()
        _holder.pool = None


@asynccontextmanager
async def get_connection() -> AsyncGenerator[PoolConnectionProxy, None]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        yield conn


@asynccontextmanager
async def get_transaction() -> AsyncGenerator[PoolConnectionProxy, None]:
    pool = await get_pool()
    async with pool.acquire() as conn, conn.transaction():
        yield conn
