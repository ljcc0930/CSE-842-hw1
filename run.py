import os

os.makedirs('logs', exist_ok=True)
for gram in [1, 2, 3, 4, 5]:
    for i in range(0, 20):
        k = i / 10.
        if not os.path.exists(f"logs/{k}_{gram}.log"):
            os.system(f"python liujia45_hw1p1_ex.py --k-smooth {k} --n-gram {gram} > logs/{k}_{gram}.log&")