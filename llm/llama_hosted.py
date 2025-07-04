from langchain_core.language_models.llms import LLM
from typing import Optional, List, Any
import requests
import json
import aiohttp
import asyncio

from langchain_core.outputs import Generation, LLMResult

class HostedLLM(LLM):
    url: str = "https://ollama-546561582790.asia-south1.run.app/api/chat"
    model: str = "llama3.1"

    @property
    def _llm_type(self) -> str:
        return "hosted_llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Sync call method required by LangChain LLM interface.
        """
        return asyncio.run(self._acall(prompt, stop=stop, **kwargs))

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        output = ""

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, headers=headers, json=data) as response:
                    async for line in response.content:
                        if line:
                            chunk = line.decode('utf-8')
                            obj = json.loads(chunk)
                            if "message" in obj and "content" in obj["message"]:
                                output += obj["message"]["content"]

            if stop:
                for stop_token in stop:
                    output = output.split(stop_token)[0]

            return output

        except Exception as e:
            print("Error calling hosted LLM:", e)
            return "⚠️ Sorry, failed to get a response from the model."

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            output = self._call(prompt, stop=stop, **kwargs)
            generation = Generation(text=output)
            generations.append([generation])

        return LLMResult(generations=generations)
