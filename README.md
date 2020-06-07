# NARRE

This is the pytorch implementation for the paper:

Chen C, Zhang M, Liu Y, et al. Neural attentional rating regression with review-level explanations[C]//Proceedings of the 2018 World Wide Web Conference. 2018: 1583-1592.

## Environment

python 3.7

pytorch 1.5.0

## Dataset

dataset: Amazon Digital_Music_5.json

pretrained word embedding: GoogleNews-vectors-negative300.bin


## How to run the code

### prepare data

put the data into ./data

``` 

 python preclean.py 

```

### train & test model

``` 

python main.py 

```

