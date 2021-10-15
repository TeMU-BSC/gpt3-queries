from pprint import pformat
import json
PATH = '/home/jordiae/PycharmProjects/gpt3-queries/output/with_ca_mlsum_sample.json-2021-10-15-1538-3e07-545d/gpt_summaries.json'

if __name__ == '__main__':
    for l in open(PATH).readlines():
        d = json.loads(l)
        for key in d:
            print()
            print(f'{key}: {d[key]}')
            print()
        print()
        print('-----------------------------')