# OS
import sys

# Computing
import numpy as np
import scipy as sp
import sympy as smp
import pandas as pd
from scipy.sparse import diags
from scipy.fft import fftfreq
from scipy.integrate import quad
from scipy.fft import fft, ifft, fft2, ifft2
from sympy.integrals.transforms import fourier_transform

# High-performance computing
from multiprocessing import Process, Pool

# Monitoring
from tqdm import tqdm
import time

# Plotting
import matplotlib.pyplot as plt
plt.style.use(['science', 'notebook'])
plt.rcParams['font.size'] = 12
plt.rcParams['legend.fontsize'] = 12
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# Other
from colorama import Fore, Style
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
