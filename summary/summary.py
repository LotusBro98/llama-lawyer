import pandas as pd
import tqdm
from datasets import load_dataset
from openai import OpenAI

MAX_LEN = 8000

client = OpenAI(
  base_url="http://localhost:18888/v1",
  api_key="sk-dummy",
)

dataset = load_dataset("lawful-good-project/ipc_decisions_4k")

short_matters = []
pbar = tqdm.tqdm(dataset["train"])
for sample in pbar:
    matter, solution = sample["text"].split("[/INST]")
    matter = matter + solution
    matter = matter.split("[INST]")[1]
    matter = matter.split("</s>")[0]
    matter = matter.replace("<anonymized>", "X")
    matter = matter[-MAX_LEN:]
    # if len(matter) > MAX_LEN:
    #     continue
    short_matters.append(matter)
    pbar.set_description(f"collected {len(short_matters)} samples")


summaries = []
for matter in tqdm.tqdm(short_matters):
    response = client.chat.completions.create(
        model="openchat_3.5",
        messages=[
            {"role": "system", "content": "Ты русскоязычный юридический консультант"},
            {"role": "user", "content": "Кратко опиши, в чем суть дела"},
            {"role": "user", "content": matter},
        ],
        temperature=0.5,
        max_tokens=800,
        top_p=1
    )

    summary = response.choices[0].message.content
    summary = summary.replace("\r", " ")
    summary = summary.replace("\n", " ")
    summary = " ".join(summary.split())
    summaries.append(summary)

df = pd.DataFrame(summaries, columns=["text"])
df.to_csv("summaries.csv", sep=";", index=False, quotechar="`")
