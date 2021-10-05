import os
import openai
import json
from dataclasses import dataclass
from typing import List
import time
import git
import uuid
from tqdm import tqdm
from typing import Dict
from dataclasses import asdict

# ENGINE = "davinci"
ENGINE = "ada"
TEMPERATURE = 0.7
# MAX_TOKENS = 64
TOP_P = 1
FREQUENCY_PENALTY = 0
PRESENCE_PENALTY = 0
DATASET = 'with_ca_mlsum_sample.json'

MAX_TOKENS_DICT = {'ca': 6395,
                     'es': 9421,
                     'ru': 145249,
                     'de': 8739,
                     'tu': 926}

STOP_SEQUENCES_DICT = {'ca': '.\nText:',
     'es': f'.\nTexto:',
     'ru': f'.\nТекст:',
     'de': f'.\nText:',
     'tu': f'.\nMetin:'}

@dataclass
class GPTConfig:
    engine: str
    temperature: float
    max_tokens: Dict
    top_p: int
    frequency_penalty: float
    presence_penalty: float
    dataset: str


config = GPTConfig(engine=ENGINE,
                   temperature=TEMPERATURE,
                   max_tokens=MAX_TOKENS_DICT,
                   top_p=TOP_P,
                   frequency_penalty=FREQUENCY_PENALTY,
                   presence_penalty=PRESENCE_PENALTY,
                   dataset=DATASET)

openai.api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class Summary:
    text: str
    summary_model: str
    summary_gt: str
    lang: str


def get_prompt(summary: Summary) -> str:
    text = summary.text
    prompt_dict = {'ca': f'Això és un sistema de resum de textos en català.\nText: {text}\nResum:',
     'es': f'Esto es un sistema de resumen de textos en castellano.\nTexto:{text}\n Resumen:',
     'ru': f'Это система суммаризации текстов на русском языке.\nТекст: {text}\nРезюме:',
     'de': f'Das ist ein system zur Textextrahierung auf Deutsch\nText: {text}\nZusammenfassung:',
     'tu': f'Bu Türkçe dilinde bir metin özetleme sistemidir.\nMetin: {text}\nÖzet:'}
    return prompt_dict[summary.lang]


def get_dataset_instances(path: str) -> Dict[str, List[Summary]]:
    with open(path, 'r') as f:
        instances = json.load(f)
    return instances


def get_gpt_summary(summary: Summary) -> Summary:
    prompt = get_prompt(summary)
    response = openai.Completion.create(
        engine=ENGINE,
        prompt=prompt,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS_DICT[summary.lang],
        top_p=TOP_P,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
        stop=STOP_SEQUENCES_DICT[summary.lang]
    )
    ans = response.last_response.data['choices'][0]['text']
    if '\n' in ans:
        lines = ans.splitlines()
        for line in lines:
            if len(line.split()) > 0:
                ans = line
                break
    ans = ans.ans()
    summary.summary_model = ans
    return summary


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

    for lang in tqdm(data):
        for summary_dict in tqdm(data[lang]):
            summary = Summary(text=summary_dict['text'], summary_model=None, summary_gt=summary_dict['summary'], lang=lang)
            with open(os.path.join(output_directory, 'gpt_summaries.json'), 'a') as f:
                gpt_summary = asdict(get_gpt_summary(summary))
                f.write(json.dumps(gpt_summary) + '\n')
