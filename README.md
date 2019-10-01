#  Seq2Seq

Simple tensorflow implementation of Seq2Seq framework. 

We build this porject for three reason.

+ Easy to use and run.
+ Easy to change for new task.
+ Easy to integrate new network structure.


## Testing enviroment

Python3

**Install requirement**

```
pip3 instll -r requirement.txt
```


## Usage


**1. Dataset (Only tested on chinese corpus)**

You can use your own dataset by changing corpus in this path:

```
'./data/corpus.txt'
```

Supported corpus format should be like:

```
question1\tanswer1
question2\tanswer2
question3\tanswer3
       .
       .
       .
```
Correspondingly you need to change *vocab.txt* in this path.


**2. Trainning**

```shell
python3 main.py --mode train
```

**3. Evaluation**

```shell
python3 main.py --mode eval
```

**4. Prediction**

```shell
python3 main.py --mode predict
```

**5. Continue Training**

You can change all the config para by editing file **config.ini** or just set it in command line.

(1) Change the restore flag in *config.ini*.

(2) Or you can run this para in command line;

```shell
python3 main.py --mode train --restore=1
```


## TDOO
+ Finish framework building and setup a demo result.
+ Testing on Pretrained Wordvector
+ Add Bert encoding.
 
 
## Contributors
+ zl007700





