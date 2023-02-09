# CSE 842 Homework

## Question 1

### Command: 

```python liujia45_hw1p1.py```

### Results:

Performance on training set (add-1-smooth):

| Fold | Precision | Recall | F1 | Accuracy |
| --- | --- | --- | --- | --- |
| 1, 2 | 0.9970 | 0.9805 | 0.9887 | 0.9888 |
| 0, 2 | 0.9939 | 0.9790 | 0.9864 | 0.9865 |
| 0, 1 | 0.9879 | 0.9775 | 0.9826 | 0.9827 |

Performance on testing set (add-1-smooth):

| Fold | Precision | Recall | F1 | Accuracy |
| --- | --- | --- | --- | --- |
| 0 | 0.7935 | 0.8078 | 0.8006 | 0.7988 |
| 1 | 0.8073 | 0.7297 | 0.7666 | 0.7778 |
| 2 | 0.8283 | 0.8234 | 0.8258 | 0.8263 |
| Avg | 0.8097 | 0.7870 | 0.7977 | 0.8010 |


## Question 2

### Command: 

```python liujia45_hw1p2_all.py```

### Results:

| Model | Features | Precision | Recall | F1 | Accuracy |
| --- | --- | --- | --- | --- | --- |
| Naive Bayes | Bag-of-words| 0.8116 | 0.7964 | 0.8039 | 0.8055 |
| Naive Bayes | TF-IDF| 0.8581 | 0.7224 | 0.7758 | 0.7965 |
| Support Vector Machine | Bag-of-words| 0.7677 | 0.6405 | 0.6978 | 0.7230 |
| Support Vector Machine | TF-IDF| 0.8215 | 0.8232 | 0.8223 | 0.8220 |
