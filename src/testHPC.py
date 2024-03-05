# !/usr/bin/env python3

"""
Created by Sotiris Catsoulis
Outline written by (upd. 10/1/2021):
    Greg Stroot, gstroot@caltech.edu
    Jane Bae (Original code outline)
Copyright 2022 California Institute of Technology
"""


import numpy as np
import multiprocessing
import time
import math


N = 4
x = np.random.random((N, N))

""" def bubble_sort(array):
    check = True
    while check == True:
      check = False
      for i in range(0, len(array)-1):
        if array[i] > array[i+1]:
          check = True
          temp = array[i]
          array[i] = array[i+1]
          array[i+1] = temp
    print("Array sorted: ", array)

if __name__ == '__main__':
    p = Process(target=bubble_sort, args=([1,9,4,5,2,6,8,4],))
    p.start()
    p.join() """


n_cores = multiprocessing.cpu_count()
print(n_cores)

N = 5000000


def cube(x: float) -> float:
    return math.sqrt(x)


if __name__ == "__main__":
    start_time = time.perf_counter()

    with multiprocessing.Pool() as pool:
        result = pool.map(cube, range(10, N))

    finish_time = time.perf_counter()

    print("Time (s): ", round(finish_time - start_time, 2))
