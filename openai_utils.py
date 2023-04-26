import openai
import tiktoken
import backoff


openai.api_key = open('oai_key', 'r').read().strip()


def is_chat(model_name):
    return 'gpt' in model_name


def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name=model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    if model == "gpt-3.5-turbo":
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def count_tokens(input, model_name):
    if is_chat(model_name) or type(input) == list:
        return num_tokens_from_messages(messages=input, model=model_name)
    else:
        return num_tokens_from_string(string=input, model_name=model_name)
    
    
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)
    
    
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def chat_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)
    

def generate(input, model='gpt-3.5-turbo', prompt=None, temperature=0.7, max_tokens=100, **kwargs):
    """
    generates one round of conversation/completion from OpenAI models

    Args:
        input (str): the user input to the model
        model (str, optional): the name of the engine to use. Defaults to 'gpt-3.5-turbo'.
        prompt (str, optional): the system prompt for chat models/ the task prompt for completion. Defaults to None.
        temperature (float, optional): the temperature to use while generating. Defaults to 0.7.
        max_tokens (int, optional): the max tokens for generation. Defaults to 100.

    Raises:
        ValueError

    Returns:
        text, num_tokens, history, output: text is the output text from the model, num_tokens 
        refers to the total tokens consumed till now, history is the messages/input in case of 
        chat/completion, output is the output from the openai api
    """
    if is_chat(model):
        if prompt is None:
            raise ValueError("system_prompt must be specified for chat models")
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": input}
        ]
        output =  chat_with_backoff(model=model, messages=messages, temperature=temperature, 
                                    max_tokens=max_tokens, **kwargs)
        text = output["choices"][0].message.content
        num_tokens = output.usage.total_tokens
        
        return text, num_tokens, messages, output
    else:
        if prompt is not None:
            input = f"{prompt}\n{input}"
        output = completions_with_backoff(model=model, prompt=input, temperature=temperature, 
                                          max_tokens=max_tokens, **kwargs)
        text = output["choices"][0]["text"]
        num_tokens = output.usage.total_tokens
        
        return text, num_tokens, input, output