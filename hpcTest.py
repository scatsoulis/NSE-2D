from libraries import *


N = 4

x = np.random.random((N,N))
print(x)
sys.exit()

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

import multiprocessing
from multiprocessing import Pool
import time
import math

n_cores = multiprocessing.cpu_count()
print(n_cores)
sys.exit()

N = 5000000

def cube(x):
    return math.sqrt(x)

if __name__ == "__main__":
    
    start_time = time.perf_counter()

    with Pool() as pool:
      result = pool.map(cube, range(10,N))

    finish_time = time.perf_counter()

    print("Time (s): ", round(finish_time - start_time, 2))
