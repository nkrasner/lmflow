# Language Model Flow
(Name could be better)

## Functionality
This is a tool for creating type-checked, stateless JSON->JSON functions for NLP tasks.
There are two main components to its use: (The names are subject to change)
- Models
- Layers

### Models:
Models are wrappers around instruction-tuned LLMs which standardize the I/O in order to abstract the model specifics away.
To define a new custom model, you need only to extend the Model class and implement the generate method, but often the __init__ method will also need implemented to store things like the actual LLM (if not using an API) or other specifications.
To use existing models, just create an instance of their class with whatever parameters they require.
E.g.

```
from models.openai_models import GPT
model = GPT(version="gpt-4o", use_reasoning=True)
```

The use_reasoning parameter uses a modified system prompt to request step-by-step reasoning, but it is also recommended to ask for this reasoning in your Layers.

### Layers:
Layers are the REAL meat of this tool. They act as JSON->JSON functions as in the following example:

```
# Some are unnecessary in this case but useful to have access to
from typing import List, Dict, Tuple, Union
# Import the Layer class and make_dict function
from lmflow import Layer, make_dict

question_answering = Layer(
  # The Model to use
  model,

  # The structure of inputs to expect
  make_dict(
    question = str,
    answer_options = List[str]
  ),

  # The prompt template (uses [$$] to denote where the inputs belong)
  "Please show all of your steps in answering the following question:\n[$question$]\n[$answer_options$]\n\nAnswer with the letter only.",

  # The structure of outputs to return
  make_dict(
    answer = str
  ),

  fmt_func = {'answer_options': lambda x: '\n'.join([f"{'a'+i}) {ans}" for i, ans in enumerate(x)])}
)

qa_input = {
            "question":"What is the capital of France?",
            "answer_options":["Washington D.C.", "Paris", "Marseille"]
            }

qa_output = question_answering(qa_input)

print(type(qa_output))
# >> <class 'dict'>
print(qa_output["answer"])
# >> 'b'
```

make_dict is a workaround since python is weird in its treatment of type hints as objects.
fmt_func allows you to preprocess the inputs before they are placed into the prompt template.

## Future Ideas
- Context tracking:
  - Pass contexts to a layer and receive the updated context
  - Context could also carry gradients for use in finetuning
- Black Boxing:
  - Essentially a wrapper around a task which takes multiple layers to complete
  - (Possibly unnecessary since this can just be done with a function definition)
