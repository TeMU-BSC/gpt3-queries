import os
import openai
import json
from dataclasses import dataclass
from typing import List, Any
import time
import git
import uuid

ENGINE = "davinci"
TEMPERATURE = 0.7
MAX_TOKENS = 64
TOP_P = 1
FREQUENCY_PENALTY = 0
PRESENCE_PENALTY = 0
DATASET = 'path'


@dataclass
class GPTConfig(str):
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
class Question(str):
    context: str
    question: str
    true_answer: str


@dataclass
class Evaluation(str):
    question: Question
    gpt_answer: str
    score: Any


def get_dataset_instances(path) -> List[Question]:
    instances = []
    with open(path, 'r') as f:
        for instance in json.load(f):
            for q in instance['qas']:
                instances.append(Question(instance['context'], q['question'], q['answers'][0]['text']))
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
    return response.last_response.data['choices'][0]['text']


def evaluate(question: Question, gpt_answer: str) -> Evaluation:
    raise NotImplementedError
    score = None
    return Evaluation(question=question, gpt_answer=gpt_answer, score=score)


if __name__ == '__main__':
    data = get_dataset_instances(DATASET)
    answers = []
    for question in data:
        gpt_answer = get_gpt_answer(question)
        evaluation = evaluate(question, gpt_answer)
        answers.append(evaluation)

    timestamp = time.strftime("%Y-%m-%d-%H%M")
    output_path = 'output'
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    extra_id = uuid.uuid4().hex
    output_file = os.path.join(output_path, f'{DATASET}-{timestamp}-{sha[:4]}-{extra_id[:4]}.json')
    with open(output_file, 'w') as f:
        json.dump({'answers': answers, 'config': config}, f)
