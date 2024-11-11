from lmflow import Model
import time
from openai import OpenAI


class GPT(Model):
    # A wrapper around any LLM to be used in LMFlow layers.
    # Used to adapt the various I/O structures of LLMs (API queries, model calls, etc.) into the form which LMFlow uses.
    # To implement, you must define the generate method and you may override the __init__ method if your model requires a mem
    def __init__(self, version="gpt-4o", **kwargs):
        super().__init__(**kwargs)
        self.client = OpenAI()
        self.version = version
        
    def generate(self, system_prompt, input_prompt, contexts):
        messages = []
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": input_prompt})
        # Concatenate the contexts if it is taking multiple tries
        for context in contexts:
            messages.append({"role": "assistant", "content": context[0]})
            messages.append({"role": "user", "content": context[1]})
        response = self.client.chat.completions.create(
            model=self.version,
            messages=messages
        )
        output = response.choices[0].message.content
        return output