from datasets import load_dataset

dataset = load_dataset("lawful-good-project/ipc_decisions_4k")

for sample in dataset["train"]:
    case, solution = sample["text"].split("[/INST]")
    case = case.split("[INST]")[1]
    print(case)
    print(solution)
