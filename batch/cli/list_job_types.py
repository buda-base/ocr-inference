from __future__ import annotations

# ruff: noqa: T201
import asyncio

from dotenv import load_dotenv

from batch.db.connection import close_pool, init_pool
from batch.db.queries import list_job_types

load_dotenv()


async def run_list_job_types() -> None:
    await init_pool()
    try:
        job_types = await list_job_types()
        if not job_types:
            print("No job types found")
            return

        print(f"{'ID':<4} {'Name':<30} {'Model':<20} {'Encoding':<10} {'Line Mode':<10}")
        print("-" * 80)
        for job_type in job_types:
            print(
                f"{job_type.id:<4} {job_type.name:<30} {job_type.model_name:<20} "
                f"{job_type.encoding:<10} {job_type.line_mode:<10}"
            )
    finally:
        await close_pool()


def main() -> None:
    asyncio.run(run_list_job_types())


if __name__ == "__main__":
    main()
