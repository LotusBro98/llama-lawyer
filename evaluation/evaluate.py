from distutils.util import strtobool
import json
from typing import List, Tuple

import pandas as pd
import torch
import tqdm
from transformers import pipeline
from openai import OpenAI


OPENAI_BASE_URL = "http://localhost:18888/v1"
OPENAI_API_KEY = "sk-dummy"

# Model from Hugging Face hub
# model = "NousResearch/Llama-2-7b-chat-hf"
model = "lotusbro/llama-2-7b-chat-law"


EVAL_PROMPT = """
Оцените ответ answer на вопрос question.
Для начала определите, ответ answer верный или неверный, согласуется ли он с correct_answer. Затем поставьте от 0 до 4 баллов за качество ответа и раскрытие темы. 
При оценке исходите из правильного ответа из авторитетного источника <correct_answer>. Если четкий ответ не дан, считайте, что он неверный.
Сообщение выведите в три строки:

score: int - ваша оценка
comment: str - Ваш комментарий
answer_is_correct: true, если ответ answer верный и согласуется с correct_answer; false, если ответ answer неверный

<question>
{question}
</question>

<answer>
{answer}
</answer>

<correct_answer>
{correct_answer}
</correct_answer>
"""


CONTEXT_HINT = [
    {"role": "user", "content": "Ответь на вопрос по авторскому праву"},
    {"role": "assistant", "content": "Напишите вопрос, я выберу вариант ответа и кратко обосную его"},

    # {"role": "user", "content": "Помоги мне ответить на вопрос по авторскому праву"},
    # {"role": "assistant", "content": "Добрый день! В чем ваш вопрос?"},

    # {"role": "user", "content": "Мне нужна юридическая консультация"},
    # {"role": "assistant", "content": "Добрый день! Опишите ситуацию, чтобы я мог подсказать вам, как поступить."},
]



pipe = pipeline(
    task="text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    max_length=300
)


def make_prompt(question: str) -> str:
    chat = CONTEXT_HINT + [
        {"role": "user", "content": question},
    ]

    prompt = pipe.tokenizer.apply_chat_template(chat, tokenize=False)
    return prompt


def generate_answers(questions: List[str]) -> List[str]:
    answers = []
    for sample in tqdm.tqdm(questions):
        # Generate answer
        question = sample
        prompt = make_prompt(question)
        result = pipe(prompt)
        answer = result[0]['generated_text']

        # Clean up answer
        answer = answer.replace(prompt, "")
        answer = answer.replace("\n", " ")
        answer = ' '.join(answer.split())

        answers.append(answer)
        print(answer)
    return answers


def evaluate_answer(client: OpenAI, question: str, answer: str, correct_answer: str) -> Tuple[bool, int, str]:
    response = client.chat.completions.create(
        model="openchat_3.5",
        messages=[
            {"role": "user", "content": EVAL_PROMPT.format(
                question=question, answer=answer, correct_answer=correct_answer
            )},
        ],
        temperature=0.5,
        max_tokens=1024,
        top_p=1
    )

    msg = response.choices[0].message.content
    msg = msg.replace("False", "false")
    msg = msg.replace("True", "true")
    score, comment,  correct = msg.split("\n")[:3]
    correct = correct.split("answer_is_correct: ")[1]
    correct = bool(strtobool(correct))
    score = int(score.split("score: ")[1])
    comment = comment.split("comment: ")[1]
    return correct, score, comment


def evaluate_answers(questions: List[str], answers: List[str], correct_answers: List[str]) -> Tuple[List[bool], List[int], List[str]]:
    client = OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
    )

    all_correct = []
    all_score = []
    all_comment = []
    for question, answer, correct_answer in tqdm.tqdm(zip(questions, answers, correct_answers), total=len(questions)):
        correct, score, comment = evaluate_answer(client, question, answer, correct_answer)
        all_correct.append(correct)
        all_score.append(score)
        all_comment.append(comment)

    return all_correct, all_score, all_comment


def evaluate():
    benchmark = pd.read_csv("benchmark1.csv")
    questions = benchmark["question"]
    correct_answers = benchmark["correct_answer"]

    answers = generate_answers(questions)

    correct, score, comment = evaluate_answers(questions, answers, correct_answers)

    results = pd.DataFrame(
        zip(questions, answers, correct_answers, correct, score, comment),
        columns=["question", "answer", "correct_answer", "correct", "score", "comment"]
    )
    results.to_csv("results.csv", index=False)


if __name__ == '__main__':
    evaluate()
