from fastapi import FastAPI
import logging

import sys
import time
from pathlib import Path
import random
import numpy as np
import json, os
from transformers import LlamaForCausalLM, LlamaTokenizer,AutoModelForCausalLM,AutoTokenizer

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min

import torch
torch.set_float32_matmul_precision('high')

from api import (
    ProcessRequest,
    ProcessResponse,
    TokenizeRequest,
    TokenizeResponse,
    Token,
    DecodeRequest,
    DecodeResponse
)

app = FastAPI()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_dir = Path("llama-33B-instructed")

logger.info(f'loading tokenizer from {str(checkpoint_dir)}')
tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir),trust_remote_code=True,use_fast=False)
tokenizer.pad_token = tokenizer.unk_token

logger.info(f'loading model from {str(checkpoint_dir)}')
model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir),
        trust_remote_code=True,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        )

meta_instruction = """
"This is a chat between a curious user and a helpful artificial intelligence assistant. "
"The assistant gives helpful, detailed, and polite answers to the user's questions with knowledge from web search."
"""

def seed_everything(seed):
    if seed is None:
        seed = random.randint(min_seed_value,max_seed_value)
        logger.warning(f'No seed found,set seed to {seed}')
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed


@app.post("/process")
async def process_request(input_data: ProcessRequest) -> ProcessResponse:
    if input_data.seed is not None:
        seed_everything(input_data.seed)
    logger.info("Using device: {}".format(device))

    prompt = f"""{meta_instruction}\nUSER:\n{input_data.prompt}\nASSISTANT:"""
    inputs = tokenizer(prompt,return_tensors="pt",add_special_tokens=True)
    prompt_length = inputs["input_ids"].shape[-1]

    t0 = time.perf_counter()
    tokenizer.pad_token = tokenizer.eos_token
    generation_outputs = model.generate(inputs['input_ids'].to(device),
                                        attention_mask=inputs['attention_mask'].to(device),
                                        return_dict_in_generate=True,
                                        output_scores=True,
                                        max_length=input_data.max_length,
                                        top_k=input_data.top_k,
                                        temperature=input_data.temperature,
                                        repetition_penalty=input_data.repetition_penalty,
                                        do_sample=True)

    t = time.perf_counter() - t0

    transition_scores= model.compute_transition_scores(
        generation_outputs.sequences, generation_outputs.scores, normalize_logits=True
    )
    logprobs = transition_scores[0].cpu().numpy().tolist()
    # for tok, score in zip(generation_outputs.sequences[:,prompt_length:][0], transition_scores[0]):
    #     logprobs.append((tok.item(),score.item()))

    scores = torch.nn.functional.log_softmax(torch.stack(generation_outputs.scores),dim=-1)

    top_logprobs = []
    for score in scores:
        tok = torch.argmax(score).item()
        top_logprob = torch.max(score, dim=-1)[0].item()
        top_logprobs.append((tok,top_logprob))

    tokens = generation_outputs.sequences

    if input_data.echo_prompt is False:
        output = tokenizer.decode(tokens.cpu()[0][prompt_length:],skip_special_tokens=True)
    else:
        output = tokenizer.decode(tokens.cpu()[0],skip_special_tokens=True)

    tokens_generated = tokens.size(0) - prompt_length

    logger.info(
        f"Time for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec"
    )

    logger.info(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
    generated_tokens = []
    for t, lp, tlp in zip(tokens.cpu()[0][prompt_length:].numpy().tolist(), logprobs, top_logprobs):
        idx, val = tlp
        tok_str = tokenizer.decode([idx])
        token_tlp = {tok_str: val}
        generated_tokens.append(
            Token(text=tokenizer.decode(t), logprob=lp, top_logprob=token_tlp)
        )
    logprobs_sum = sum(logprobs)
    # Process the input data here
    return ProcessResponse(
        text=output, tokens=generated_tokens, logprob=logprobs_sum, request_time=t
    )

@app.post("/tokenize")
async def tokenize(input_data: TokenizeRequest) -> TokenizeResponse:
    logger.info("Using device: {}".format(device))
    t0 = time.perf_counter()
    encoded = tokenizer.encode(
        input_data.text,
        add_special_tokens=True
    )
    t = time.perf_counter() - t0
    tokens = encoded
    return TokenizeResponse(tokens=tokens, request_time=t)

@app.post("/decode")
async def decode(input_data: DecodeRequest) -> DecodeResponse:
    logger.info("Using device: {}".format(device))
    t0 = time.perf_counter()
    decoded = tokenizer.decode(input_data.tokens)
    t = time.perf_counter() - t0
    return DecodeResponse(text=decoded, request_time=t)

