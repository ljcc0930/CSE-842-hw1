# CSE 842 Homework

## Question 1

### Command: 

```python liujia45_hw1p1.py```

### Results:

Performance on training set (add-1-smooth):

| Fold | Precision | Recall | F1 | Accuracy |
| --- | --- | --- | --- | --- |
| 1, 2 | 0.9923 | 0.9700 | 0.9810 | 0.9813 |
| 0, 2 | 0.9909 | 0.9820 | 0.9864 | 0.9865 |
| 0, 1 | 0.9939 | 0.9790 | 0.9864 | 0.9865 |

Performance on testing set (add-1-smooth):

| Fold | Precision | Recall | F1 | Accuracy |
| --- | --- | --- | --- | --- |
| 0 | 0.8245 | 0.7898 | 0.8067 | 0.8108 |
| 1 | 0.8254 | 0.8378 | 0.8316 | 0.8303 |
| 2 | 0.8006 | 0.8293 | 0.8147 | 0.8114 |
| Avg | 0.8168 | 0.8190 | 0.8177 | 0.8175 |


## Question 2

### Command: 

```python liujia45_hw1p2_all.py```

### Results:

| Model | Features | Precision | Recall | F1 | Accuracy |
| --- | --- | --- | --- | --- | --- |
| Naive Bayes | bag-of-words| 0.8127 | 0.7883 | 0.8003 | 0.8030 |
| Naive Bayes | TF-IDF| 0.8534 | 0.7258 | 0.7803 | 0.7985 |
| Support Vector Machine | bag-of-words| 0.7656 | 0.6396 | 0.6963 | 0.7215 |
| Support Vector Machine | TF-IDF| 0.8285 | 0.8291 | 0.8288 | 0.8285 |
