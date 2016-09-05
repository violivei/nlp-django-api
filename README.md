
## Requirements

- Django
- Python 2.7
- Theano 0.7
- Numpy

Using the pre-trained `word2vec` vectors will also require downloading the binary file from
https://code.google.com/p/word2vec/

### Data Preprocessing
To process the raw data, run

```
python process_data.py path
```

where path points to the word2vec binary file (i.e. `GoogleNews-vectors-negative300.bin` file). 
This will create a pickle object called `mr.p` in the same folder, which contains the dataset
in the right format.

## Running

Train (GPU):

```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python conv_net_sentence.py -train mr.p stackoverflow.train
```

Train (CPU):

```
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python conv_net_sentence.py -train mr.p stackoverflow.train
```

Predict:

```
- python manage.py runserver
- curl http://localhost:8080/api/classify/ -d "description=question: Are there any SciFi treatments of time travel that avoid the typical paradoxes? [duplicate]; excerpt: Possible Duplicate:\n  Why do time-travel stories often have the characters “returning” to the future?  \n\n\n\n\nThe possibility of time travel normally creates paradoxes. If you can travel into the ...\r\n        "
```

## Dataset

Stack Exchange is an information powerhouse, built on the power of crowdsourcing. It has 105 different topics and each topic has a library of questions which have been asked and answered by knowledgeable members of the StackExchange community. The topics are as diverse as travel, cooking, programming, engineering and photography.

We have hand-picked ten different topics (such as Electronics, Mathematics, Photography etc.) from Stack Exchange, and we provide you with a set of questions from these topics.

Given a question and an excerpt, your task is to identify which among the 10 topics it belongs to.

Link: https://www.hackerrank.com/challenges/stack-exchange-question-classifier
