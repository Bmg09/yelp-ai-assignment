import asyncio
from typing import Awaitable, Callable, TypeVar

from tqdm.asyncio import tqdm_asyncio

T = TypeVar("T")
R = TypeVar("R")


async def gather_limited(
    items: list[T],
    fn: Callable[[T], Awaitable[R]],
    concurrency: int = 30,
    desc: str | None = None,
) -> list[R]:
    sem = asyncio.Semaphore(concurrency)

    async def run(x: T) -> R:
        async with sem:
            return await fn(x)

    return await tqdm_asyncio.gather(*[run(x) for x in items], desc=desc)
