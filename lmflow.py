from typing import List, Dict, Tuple, Union, get_args, get_origin
from abc import ABC, abstractmethod
from json import JSONDecoder
import json
import typeguard
from copy import deepcopy

def make_dict(**types):
    # A dumb workaround for types not being directly added to dictionaries
    return types
        

class Layer():
    def __init__(self, model, input, prompt, output, fmt_func = None):
        self.model = model # The lmflow Model object which will generate the output given the input
        self.input = input # A dictionary which defines the keys and types for the inputs
        self.prompt = prompt # A text prompt which defines the goal of the layer. Wrap variable names in [$$] (e.g. [$age$] with be replaced by the value of the "age" variable)
        self.output = output # A dictionary which defines the keys and types for the outputs (values are type hints)
        self.fmt_func = {} # A dictionary of formatting functions for each key
        for key in input.keys():
            self.fmt_func[key] = lambda x: str(x)
        if fmt_func is not None:
            # Allow the addition of custom formatting functions per key
            for key, func in fmt_func.items():
                self.fmt_func[key] = func
    
    # def __check_type__(self, var, type_hint):
    #     # Get the origin of the type (e.g., list, dict)
    #     origin = get_origin(type_hint)
        
    #     # Base case: if there is no origin, it's a simple type like int, str, etc.
    #     if origin is None:
    #         return isinstance(var, type_hint)
        
    #     # Handle list and tuple
    #     if origin in {list, tuple}:
    #         if not isinstance(var, origin):
    #             return False
    #         # Get the argument types for the container
    #         args = get_args(type_hint)
    #         if len(args) == 1:  # for list[int] or list[str]
    #             return all(self.__check_type__(item, args[0]) for item in var)
    #         elif len(args) > 1 and origin is tuple:  # for tuple[int, str]
    #             return len(var) == len(args) and all(self.__check_type__(item, arg) for item, arg in zip(var, args))
        
    #     # Handle dictionary
    #     if origin is dict:
    #         if not isinstance(var, dict):
    #             return False
    #         key_type, value_type = get_args(type_hint)
    #         return all(self.__check_type__(k, key_type) and self.__check_type__(v, value_type) for k, v in var.items())
        
    #     return False
    
    def __check_type__(self, var, type_hint):
        try:
            typeguard.check_type(var, type_hint)
            return True
        except typeguard.TypeCheckError:
            return False
    
    def __verify__(self, input, structure):
        # Check that the input matches the structure specified for this layer
        for key, value in input.items():
            if key not in structure.keys():
                raise KeyError(f"{key} is not a valid key.")
            if not self.__check_type__(value, structure[key]):
                raise ValueError(f"The type of {key} does not match the specification it should match {structure[key]}.")
        missing_keys = []
        for key in structure.keys():
            if key not in input.keys():
                missing_keys.append(key)
        if len(missing_keys) > 0:
            raise KeyError(f"Missing the following keys: {missing_keys}.")
    
    def __verify_input__(self, input):
        self.__verify__(input, self.input)
    
    def __verify_output__(self, output):
        self.__verify__(output, self.output)
    
    def fill_prompt(self, input):
        prompt = self.prompt
        for key, value in input.items():
            prompt = prompt.replace(f"[${key}$]", self.fmt_func[key](value))
        return prompt
    
    def __call__(self, input, max_tries=3):
        # Initialize exceptions list as empty (For some reason, this is being modified globally)
        exceptions = []
        self.__verify_input__(input)
        prompt = self.fill_prompt(input)
        output = self.model(prompt, self.output, max_tries, exceptions)
        while len(exceptions) < max_tries:
            try:
                self.__verify_output__(output)
                break
            except Exception as e:
                exceptions.append((str(output), str(e)))
                if len(exceptions) < max_tries:
                    self.model(prompt, self.output, max_tries, exceptions)
                else:
                    raise TimeoutError(f"Could not produce an adequate response in less than {max_tries} attempts.")
        return output
    

def extract_json_objects(text, decoder=JSONDecoder()):
    """Find JSON objects in text, and yield the decoded JSON data

    Does not attempt to look for JSON arrays, text, or other JSON types outside
    of a parent JSON object.

    """
    pos = 0
    while True:
        match = text.find('{', pos)
        if match == -1:
            break
        try:
            result, index = decoder.raw_decode(text[match:])
            yield result
            pos = match + index
        except Exception as e:
            pos = match + 1

def type_hint_to_str(type_hint) -> str:
    origin = get_origin(type_hint)
    # Base case 1: If origin is None, it could be a basic type or a literal
    if origin is None:
        # Handle the case for literals like 'str', 'int', etc.
        if isinstance(type_hint, type):
            return type_hint.__name__
        # Handle string literals or other types that don't have __name__
        return str(type_hint)
    # Get the arguments of the type hint (e.g., int, str)
    args = get_args(type_hint)
    # Recursively convert each argument to a string
    args_str = ', '.join(type_hint_to_str(arg) for arg in args)
    # Construct the string for the current level
    return f"{origin.__name__}[{args_str}]"


class Model(ABC):
    # A wrapper around any LLM to be used in LMFlow layers.
    # Used to adapt the various I/O structures of LLMs (API queries, model calls, etc.) into the form which LMFlow uses.
    # To implement, you must define the generate method and you may override the __init__ method if your model requires a mem
    def __init__(self, use_reasoning=True):
        # Define the model, tokenizer, etc. here in children
        # Call super.__init__(self) in children
        # If writing a custom system prompt, request a JSON output and use [$out_fmt$] to mark where the format definition belongs
        if use_reasoning:
            self.SYSTEM_PROMPT = "You are a helpful agent capable of completing any task correctly and efficiently. " + \
                             "You will be given a request and it is your job is to think step-by-step and then respond to this request using a JSON object with exactly the following form:\n[$out_fmt$]\n" + \
                             "Be sure to match this format exactly including the keys and using the specified data types. Do not add any additional items or remove any of the specified items." + \
                             "Please think carefully with step-by-step reasoning before coming up with a solution in the form of a JSON object. Do not provide the solution prior to thinking through the steps."
        else:
            self.SYSTEM_PROMPT = "You are a helpful agent capable of completing any task correctly and efficiently. " + \
                             "You will be given a request and it is your job is to respond to this request using only a JSON object with the following form:\n[$out_fmt$]\n" + \
                             "Be sure to match this format exactly including the keys and using the specified data types. " + \
                             "Please include only the JSON object in your response."
        # If writing a custom exception prompt, use [$except$] to mark where the exception text belongs
        self.EXCEPT_PROMPT = "There is an error in your response. This is the cause: [$except$]\nPlease correct this error with a JSON object in the proper format."
    
    @abstractmethod
    def generate(self, system_prompt, input_prompt, contexts):
        # Preprocess and pass the prompts into the model an return the output as plain text
        # The order should be: system_prompt, input_prompt, contexts
        # The model should use the JSON output flag if available, the system prompt will request the JSON format accoring to a given structure
        # Contexts may be empty but is otherwise a list of context tuples where a context tuple has an LM output at index 0 and a user response at index 1 (these handle exceptions)
        pass
    
    def __stringify_outfmt__(self, outfmt):
        fmt = deepcopy(outfmt)
        for key in outfmt:
            fmt[key] = type_hint_to_str(fmt[key])
        return json.dumps(fmt)
    
    def __call__(self, input_prompt, output_structure, max_tries, exceptions):
        # Fill system prompt with output structure knowledge
        system_prompt = self.SYSTEM_PROMPT.replace("[$out_fmt$]", self.__stringify_outfmt__(output_structure))
        
        while len(exceptions) < max_tries:
            # Fill exception prompts for contexts
            filled_exceptions = []
            for (output, exception) in exceptions:
                filled_exceptions.append((output, self.EXCEPT_PROMPT.replace("[$except$]", exception)))
            try:
                output = self.generate(system_prompt, input_prompt, filled_exceptions).strip()
            except ValueError as e:
                raise TimeoutError(f"Could not produce an adequate response in the maximum context window.")
             
            jsons = list(extract_json_objects(output))
            # Incase the output contains more than one json, use the first one
            # This could maybe be improved by finding the one that matches the output_structure, if any
            try:
                # Use the last json in case the model produced others during its step by step reasoning.
                output = jsons[-1]
                return output
            except IndexError:
                exceptions.append((output, "The JSON could not be parsed, please correct this and try again."))
        