import os
import openai
import json
from dataclasses import dataclass, asdict
from typing import List
import time
import git
import uuid
from tqdm import tqdm
from datasets import load_dataset

#ENGINE = "davinci"
ENGINE = "ada"
TEMPERATURE = 0
MAX_TOKENS = 64
TOP_P = 0.95
FREQUENCY_PENALTY = 0
PRESENCE_PENALTY = 0
LANGS = ['xquad.en', 'xquad.de', 'xquad.es', 'xquad.ru', 'xquad.tr', 'xquad.ca']

STOP_SEQUENCES_DICT = {'ca': ['\nContext:', '\nPregunta:'],
                       'es': ['\nContexto:', '\nPregunta'],
                       'ru': ['\nКонтекст:', '\nВопрос:'],
                        'en': ['\nContext:', '\nQuestion:'],
                        'de': ['\nKontext:', '\nFrage:'],
                        'tr': ['\nBağlam:', '\nSoru:']
                       }


def prompt_dict(context, question, lang):
    return {'en': f'This is a question answering system in English.\nContext: {context}\nQuestion: {question}\nAnswer:',
               'de': f'Das ist ein System zur Beantwortung von Fragen auf Deutsch.\nKontext: {context}\nFrage: {question}\nAntwort:',
               'es': f'Esto es un sistema de respuesta de preguntas en castellano.\nContexto: {context}\nPregunta: {question}\nRespuesta:',
               'ru': f'Это система генерации ответов на вопросы на русском языке.\nКонтекст: {context}\nВопрос: {question}\nОтвет:',
               'tr': f'Bu Türkçe dilinde bir soru yanıtlama sistemidir.\nBağlam: {context}\nSoru: {question}\nCevap:',
               'ca': f'Això és un sistema de resposta de preguntes en català.\nContext: {context}\nPregunta: {question}\nResposta:'}[lang]


@dataclass
class GPTConfig:
    engine: str
    temperature: float
    max_tokens: int
    top_p: float
    frequency_penalty: float
    presence_penalty: float


config = GPTConfig(engine=ENGINE,
                   temperature=TEMPERATURE,
                   max_tokens=MAX_TOKENS,
                   top_p=TOP_P,
                   frequency_penalty=FREQUENCY_PENALTY,
                   presence_penalty=PRESENCE_PENALTY)

openai.api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class Question:
    id: str
    context: str
    question: str
    true_answer: str

    @staticmethod
    def from_hf(answers, context, id, question, **kwargs):
        if len(kwargs) > 0:
            print('Extra kwargs', kwargs)
        return Question(id=id, context=context, question=question, true_answer=answers['text'][0])


def get_dataset_instances(path: str) -> List[Question]:
    instances = []
    with open(path, 'r') as f:
        for instance in json.load(f)['data']:
            for p in instance['paragraphs']:
                for q in p['qas']:
                    instances.append(Question(q['id'], p['context'], q['question'], q['answers'][0]['text']))
    return instances


def get_gpt_answer(q: Question, lang: str) -> str:
    lang = lang.split('.')[1]
    question = q.question
    context = q.context
    prompt = prompt_dict(context, question, lang)

    response = openai.Completion.create(
        engine=ENGINE,
        prompt=prompt,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
        stop=STOP_SEQUENCES_DICT[lang]
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
    output_directory = os.path.join(output_path, f'{timestamp}-{sha[:4]}-{extra_id[:4]}')
    os.makedirs(output_directory)

    with open(os.path.join(output_directory, 'args.json'), 'w') as f:
        json.dump(vars(config), f)

    for lang in LANGS:
        if lang == 'xquad.ca':
            data = load_dataset("BSC-TeMU/xquad-ca", 'XQuAD-ca', split='test')
        else:
            data = load_dataset('xquad', lang, split='validation')

        max_qs_per_lang = 2
        i = 0
        for question in tqdm(data):
            i += 1
            #if i > max_qs_per_lang:
            #    break
            question = Question.from_hf(**question)
            with open(os.path.join(output_directory, 'gpt_answers.json'), 'a') as f:
                gpt_answer = get_gpt_answer(question, lang)
                f.write(json.dumps({question.id: gpt_answer, 'lang': lang, 'question': asdict(question)}) + '\n')

    #from prettify import prettify

    #prettify(os.path.join(output_directory, 'gpt_answers.json'))
