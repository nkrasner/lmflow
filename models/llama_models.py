from lmflow import Model
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
import time

class Llama2(Model):
    # A wrapper around any LLM to be used in LMFlow layers.
    # Used to adapt the various I/O structures of LLMs (API queries, model calls, etc.) into the form which LMFlow uses.
    # To implement, you must define the generate method and you may override the __init__ method if your model requires a mem
    def __init__(self, model_dir, tokenizer_dir, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), **kwargs):
        super().__init__(**kwargs)
        self.B_INST, self.E_INST = "[INST]", "[/INST]"
        self.B_SYS, self.E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        self.BOS, self.EOS = "<s>", "</s>"
        # initialize the model
        self.model = LlamaForCausalLM.from_pretrained(
            model_dir
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_dir
        )
        self.device = device
        try:
            self.model.to(device)
        except ValueError:
            pass
        
        
        
    def generate(self, system_prompt, input_prompt, contexts):
        # Preprocess and pass the prompts into the model and return the output as plain text
        prompt = f"{self.BOS}{self.B_INST} {self.B_SYS} {system_prompt} {self.E_SYS}{input_prompt} {self.E_INST} "
        # Concatenate the contexts if it is taking multiple tries
        for context in contexts:
            prompt += f" {context[0]} {self.EOS}\n{self.BOS}{self.B_INST} {context[1]} {self.E_INST} "
        
        #prompt += "{" # Push the model to start a JSON object
        print("Prompt: ", prompt) #ALSO PRINT THE PROB OF FIRST TOKEN
        tokens = self.tokenizer(prompt, return_tensors="pt")
        tokens.to(self.device)
        # output = self.model.generate(tokens.input_ids, num_beams=4, do_sample=True, max_length=4096)
        output = self.model.generate(tokens.input_ids, do_sample=True, max_length=4096)
        # print("Before processing: ", self.tokenizer.batch_decode(output)[0])
        output = self.tokenizer.batch_decode(output)[0].split(self.E_INST)[-1].strip().replace(self.EOS, "")
        print("Model Output: ", output)
        return output
    
    
class Llama3(Model):
    # A wrapper around any LLM to be used in LMFlow layers.
    # Used to adapt the various I/O structures of LLMs (API queries, model calls, etc.) into the form which LMFlow uses.
    # To implement, you must define the generate method and you may override the __init__ method if your model requires a mem
    def __init__(self, model_dir, tokenizer_dir, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), **kwargs):
        super().__init__(**kwargs)
        self.HEAD_B, self.HEAD_E = "<|start_header_id|>", "<|end_header_id|>"
        self.SYS = f"{self.HEAD_B}system{self.HEAD_E}\n\n"
        self.USER = f"{self.HEAD_B}user{self.HEAD_E}\n\n"
        self.ASSIST = f"{self.HEAD_B}assistant{self.HEAD_E}"
        self.EOT = "<|eot_id|>" # End of "turn"
        self.BOS, self.EOS = "<|begin_of_text|>", "<|end_of_text|>"
        # initialize the model
        self.model = LlamaForCausalLM.from_pretrained(
            model_dir
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_dir
        )
        self.device = device
        try:
            self.model.to(device)
        except ValueError:
            pass
        
        
        
    def generate(self, system_prompt, input_prompt, contexts):
        # Preprocess and pass the prompts into the model and return the output as plain text
        prompt = f"{self.BOS}{self.SYS}{system_prompt}{self.EOT}{self.USER}{input_prompt}{self.EOT}\n{self.ASSIST}"
        # Concatenate the contexts if it is taking multiple tries
        for context in contexts:
            prompt += f"{context[0]}{self.EOT}\n{self.USER}{context[1]}{self.EOT}\n{self.ASSIST}"
        
        # prompt += "{" # Push the model to start a JSON object
        
        tokens = self.tokenizer(prompt, return_tensors="pt")
        tokens.to(self.device)
        output = self.model.generate(tokens.input_ids, num_beams=4, do_sample=True, max_length=8000)
        output = self.tokenizer.batch_decode(output)[0].split(self.E_INST)[-1].strip().replace(self.EOS, "")
        return output
    
    
class Llama31(Llama3):
    # A wrapper around any LLM to be used in LMFlow layers.
    # Used to adapt the various I/O structures of LLMs (API queries, model calls, etc.) into the form which LMFlow uses.
    # To implement, you must define the generate method and you may override the __init__ method if your model requires a mem
    def __init__(self, model_dir, tokenizer_dir, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), **kwargs):
        super().__init__(model_dir, tokenizer_dir, device, **kwargs)
        self.EOT = "<|eot_id|>\n" # End of "turn"
    def generate(self, system_prompt, input_prompt, contexts):
        # Preprocess and pass the prompts into the model and return the output as plain text
        prompt = f"{self.BOS}{self.SYS}Cutting Knowledge Date: December 2023\nToday Date: {time.strftime('%d %B %Y')}\n\n{system_prompt}{self.EOT}{self.USER}{input_prompt}{self.EOT}\n{self.ASSIST}"
        # Concatenate the contexts if it is taking multiple tries
        for context in contexts:
            prompt += f"{context[0]}{self.EOT}\n{self.USER}{context[1]}{self.EOT}{self.ASSIST}"
        
        # prompt += "{" # Push the model to start a JSON object
        
        tokens = self.tokenizer(prompt, return_tensors="pt")
        tokens.to(self.device)
        output = self.model.generate(tokens.input_ids, num_beams=4, do_sample=True, max_length=8000)
        output = self.tokenizer.batch_decode(output)[0].split(self.E_INST)[-1].strip().replace(self.EOS, "")
        return output