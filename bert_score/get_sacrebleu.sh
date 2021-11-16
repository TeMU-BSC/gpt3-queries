echo "ca"
cat ada/ca_hyps.txt | sacrebleu ada/ca_refs.txt | grep '"score"' | egrep -o '[0-9\.]{2,4}\b'
cat babbage/ca_hyps.txt | sacrebleu babbage/ca_refs.txt | grep '"score"' | egrep -o '[0-9\.]{2,4}\b'
cat curie/ca_hyps.txt | sacrebleu curie/ca_refs.txt | grep '"score"' | egrep -o '[0-9\.]{2,4}\b'
cat davinci/ca_hyps.txt | sacrebleu davinci/ca_refs.txt | grep '"score"' | egrep -o '[0-9\.]{2,4}\b'
echo "de"
cat ada/de_hyps.txt | sacrebleu ada/de_refs.txt | grep '"score"' | egrep -o '[0-9\.]{2,4}\b'
cat babbage/de_hyps.txt | sacrebleu babbage/de_refs.txt | grep '"score"' | egrep -o '[0-9\.]{2,4}\b'
cat curie/de_hyps.txt | sacrebleu curie/de_refs.txt | grep '"score"' | egrep -o '[0-9\.]{2,4}\b'
cat davinci/de_hyps.txt | sacrebleu davinci/de_refs.txt | grep '"score"' | egrep -o '[0-9\.]{2,4}\b'
echo "en"
cat ada/en_hyps.txt | sacrebleu ada/en_refs.txt | grep '"score"' | egrep -o '[0-9\.]{2,4}\b'
cat babbage/en_hyps.txt | sacrebleu babbage/en_refs.txt | grep '"score"' | egrep -o '[0-9\.]{2,4}\b'
cat curie/en_hyps.txt | sacrebleu curie/en_refs.txt | grep '"score"' | egrep -o '[0-9\.]{2,4}\b'
cat davinci/en_hyps.txt | sacrebleu davinci/en_refs.txt | grep '"score"' | egrep -o '[0-9\.]{2,4}\b'
echo "es"
cat ada/es_hyps.txt | sacrebleu ada/es_refs.txt | grep '"score"' | egrep -o '[0-9\.]{2,4}\b'
cat babbage/es_hyps.txt | sacrebleu babbage/es_refs.txt | grep '"score"' | egrep -o '[0-9\.]{2,4}\b'
cat curie/es_hyps.txt | sacrebleu curie/es_refs.txt | grep '"score"' | egrep -o '[0-9\.]{2,4}\b'
cat davinci/es_hyps.txt | sacrebleu davinci/es_refs.txt | grep '"score"' | egrep -o '[0-9\.]{2,4}\b'
echo "tu"
cat ada/tu_hyps.txt | sacrebleu ada/tu_refs.txt | grep '"score"' | egrep -o '[0-9\.]{2,4}\b'
cat babbage/tu_hyps.txt | sacrebleu babbage/tu_refs.txt | grep '"score"' | egrep -o '[0-9\.]{2,4}\b'
cat curie/tu_hyps.txt | sacrebleu curie/tu_refs.txt | grep '"score"' | egrep -o '[0-9\.]{2,4}\b'
cat davinci/tu_hyps.txt | sacrebleu davinci/tu_refs.txt | grep '"score"' | egrep -o '[0-9\.]{2,4}\b'