import asyncio
from .config import PipelineConfig
from .types_common import ImageTask
from .ld_volume_worker import LDVolumeWorker

async def main():
    cfg = PipelineConfig() # default values for now    

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
