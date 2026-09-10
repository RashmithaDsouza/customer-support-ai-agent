import sys
from sklearn.metrics import cohen_kappa_score

h = [1, 1, 2, 3, 3, 4, 4, 2, 1, 2, 3, 4, 5, 5, 5, 4, 3, 2, 2]
l = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

try:
    print("Trying kappa...")
    k = cohen_kappa_score(h, l, weights="quadratic")
    print("Success! Kappa:", k)
except Exception as e:
    print("Failed:", type(e).__name__, e)
