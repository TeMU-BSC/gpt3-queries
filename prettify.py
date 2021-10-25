from pprint import pformat
import json
#PATH = '/home/jordiae/PycharmProjects/gpt3-queries/output/with_ca_mlsum_sample_def.json-2021-10-25-0908-0c31-57c4/gpt_summaries.json'
#PATH = "/home/jordiae/PycharmProjects/gpt3-queries/output/with_ca_mlsum_sample_def.json-2021-10-25-0920-0c31-8229/gpt_summaries.json"
#PATH = "/home/jordiae/PycharmProjects/gpt3-queries/output/with_ca_mlsum_sample_def.json-2021-10-25-0930-0c31-2868/gpt_summaries.json"
PATH = "/home/jordiae/PycharmProjects/gpt3-queries/output/with_ca_mlsum_sample_def.json-2021-10-25-0940-0c31-e9c2/gpt_summaries.json"

def prettify(path):
    for l in open(path).readlines():
        d = json.loads(l)
        for key in d:
            print()
            print(f'{key}: {d[key]}')
            print()
        print()
        print('-----------------------------')
if __name__ == '__main__':
    prettify(PATH)