#Common imports
import numpy as np 

#File dialog impots
import tkinter as tk
from tkinter import filedialog

#Import plotting
import matplotlib.pyplot as plt

#Get file path to numpy plot
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()


#Read in the numpy array
data = np.load(file_path)

plt.plot(data.T)

plt.show()