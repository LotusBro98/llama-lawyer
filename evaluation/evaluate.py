import pandas as pd
import torch
import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging, LlamaTokenizerFast,
)

# Model from Hugging Face hub
base_model = "NousResearch/Llama-2-7b-chat-hf"

compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

pipe = pipeline(task="text-generation", model=model, tokenizer=base_model, max_length=512)

questions = pd.read_csv("questions.csv", sep=";")

answers = []
for i, sample in tqdm.tqdm(questions.iterrows(), total=len(questions)):
    prompt = f'{sample["question"]}'
    result = pipe(prompt)
    answer = result[0]['generated_text']
    answer = answer.replace(prompt, "")
    answer = answer.replace("\n", " ")
    answer = ' '.join(answer.split())
    answers.append(answer)
    print(answer)

answers = pd.DataFrame(answers, columns=["answer"])
answers.to_csv("answers.csv", sep=";")

print(questions)
