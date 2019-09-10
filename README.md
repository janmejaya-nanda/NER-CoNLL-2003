# NER CoNLL 2003

An Named Entity Recogniser Model for [CoNLL 2003](https://www.clips.uantwerpen.be/conll2003/ner/). 
### Prerequisites

Hardware and software requirements are listed bellow
* Need to download [google's word2vec](https://code.google.com/archive/p/word2vec/downloads) and keep it in "data/embedding/GoogleNews-vectors-negative300.bin"
* PC with (at least) 2GB GPU.
* Install CUDA 10, as its required for TensorFlow 2.0.

## Installation

* Create a [Virtual environment](https://virtualenv.pypa.io/en/latest/)
* Install dependencies (`pip install -r requirements.txt`)
* Run the code (`python3 train.py -m 'bilstm'`)