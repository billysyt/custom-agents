"""One Langfuse project, with job IDs correlating queue runs and graph steps."""
import asyncio
from contextlib import nullcontext
import os
from langfuse import Langfuse,propagate_attributes
from langfuse.langchain import CallbackHandler


class Telemetry:
    def __init__(self):
        self.client=Langfuse() if os.getenv('LANGFUSE_SECRET_KEY') else None

    def observation(self,**kwargs):
        return self.client.start_as_current_observation(**kwargs) if self.client else nullcontext()

    def attributes(self,job_id):
        return propagate_attributes(session_id=str(job_id),user_id='billy',tags=['news-writer'],
            metadata={'job_id':str(job_id)}) if self.client else nullcontext()

    def callbacks(self):return [CallbackHandler()] if self.client else []

    async def flush(self):
        if self.client:await asyncio.to_thread(self.client.flush)
