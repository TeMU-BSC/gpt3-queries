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
TEMPERATURE = 0.7
MAX_TOKENS = 64
TOP_P = 1
FREQUENCY_PENALTY = 0
PRESENCE_PENALTY = 0
DATASET = 'CatQA_/xquad_ca_v3.json'#'test.json'


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
    # "Context: Wagner va intentar que Tristany es representés en un altre lloc, en ciutats com Estrasburg, París, Praga i fins i tot a Rio de Janeiro, on s'hauria cantat en italià. Cap compromís va arribar a bon terme, i l'òpera no es va presentar davant del públic fins que el rei Lluís II de Baviera, apassionat promotor de Wagner, no va prestar finalment un decisiu suport. Un dels més transcendents directors d'orquestra d'aquella època va ser l'escollit per a dirigir-la a l'Òpera de Munic, Hans von Bülow, un altre fervent defensor de Wagner malgrat el fet que aquest estava tenint un afer amorós amb la seva muller Cosima von Bülow. Fins i tot llavors, la planejada estrena del 15 de maig de 1865 va haver de ser cancel·lada perquè la Isolda, Malvina Schnorr s'havia quedat afònica. Finalment, l'òpera va ser representada el 10 de juny de 1865. Ludwig Schnorr von Carolsfeld va cantar el paper de Tristany, i la seva muller Malvina, va fer-se càrrec del d'Isolda. Com a curiositat, el pare de Richard Strauss, Franz, solista de trompa, va ser un dels músics que van tocar a l'estrena.\nPregunta: A quines ciutats Wagner va intentar que es representés Tristany?\nResposta:"
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
        answer = answer.splitlines()[0]
    answer = answer.strip()
    if answer[-1] == '.':
        answer = answer[:-1]
    return answer


if __name__ == '__main__':
    data = get_dataset_instances(DATASET)
    answers = {}
    for question in tqdm(data):
        gpt_answer = get_gpt_answer(question)
        answers[question.qid] = gpt_answer

    timestamp = time.strftime("%Y-%m-%d-%H%M")
    output_path = 'output'
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    extra_id = uuid.uuid4().hex
    output_directory = os.path.join(output_path, f'{DATASET}-{timestamp}-{sha[:4]}-{extra_id[:4]}')
    os.makedirs(output_directory)
    with open(os.path.join(output_directory, 'gpt_answers.json'), 'w') as f:
        json.dump(answers, f)
    with open(os.path.join(output_directory, 'args.json'), 'w') as f:
        json.dump(vars(config), f)
