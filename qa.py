import os
import openai
import json
from dataclasses import dataclass
from typing import List
import time
import git
import uuid
from tqdm import tqdm

ENGINE = "davinci"
# ENGINE = "ada"
TEMPERATURE = 0.7
MAX_TOKENS = 64
TOP_P = 1
FREQUENCY_PENALTY = 0
PRESENCE_PENALTY = 0
DATASET = 'CatQA_/xquad_ca_v3.json'


@dataclass
class GPTConfig:
    engine: str
    temperature: float
    max_tokens: int
    top_p: int
    frequency_penalty: float
    presence_penalty: float
    dataset: str


config = GPTConfig(engine=ENGINE,
                   temperature=TEMPERATURE,
                   max_tokens=MAX_TOKENS,
                   top_p=TOP_P,
                   frequency_penalty=FREQUENCY_PENALTY,
                   presence_penalty=PRESENCE_PENALTY,
                   dataset=DATASET)

openai.api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class Question:
    qid: str
    context: str
    question: str
    true_answer: str


def get_dataset_instances(path: str) -> List[Question]:
    instances = []
    with open(path, 'r') as f:
        for instance in json.load(f)['data']:
            for p in instance['paragraphs']:
                for q in p['qas']:
                    instances.append(Question(q['id'], p['context'], q['question'], q['answers'][0]['text']))
    return instances


def get_gpt_answer(q: Question) -> str:
    question = q.question
    context = q.context
    prompt =\
        f'Això és un sistema de resposta de preguntes en català.\nContext: {context}\nPregunta: {question}\nResposta:'
    response = openai.Completion.create(
        engine=ENGINE,
        prompt=prompt,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY
    )
    answer = response.last_response.data['choices'][0]['text']
    if '\n' in answer:
        lines = answer.splitlines()
        for line in lines:
            if len(line.split()) > 0:
                answer = line
                break
        # answer = answer.splitlines()[0]
    answer = answer.strip()
    if answer[-1] == '.':
        answer = answer[:-1]
    return answer


if __name__ == '__main__':
    timestamp = time.strftime("%Y-%m-%d-%H%M")
    output_path = 'output'
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    extra_id = uuid.uuid4().hex
    dataset_name = os.path.basename(DATASET)
    output_directory = os.path.join(output_path, f'{dataset_name}-{timestamp}-{sha[:4]}-{extra_id[:4]}')
    os.makedirs(output_directory)

    with open(os.path.join(output_directory, 'args.json'), 'w') as f:
        json.dump(vars(config), f)
    data = get_dataset_instances(DATASET)

    for question in tqdm(data):
        with open(os.path.join(output_directory, 'gpt_answers.json'), 'a') as f:
            gpt_answer = get_gpt_answer(question)
            f.write(json.dumps({question.qid: gpt_answer}) + '\n')
