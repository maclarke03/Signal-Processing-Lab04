# Micah Clarke
# ID: 1001288866

import numpy as np
import matplotlib.pyplot as plt
import csv
import math

def lowPassFilter(rawData,fs,m,l):
    #Cut-off frequency
    fc = 50
    # Normalized frequency
    ft = fc/fs
    # Weighted values for lowpass
    w1 = []

    i = 0
    while i < len(l):
        if l[i] == 10:
            w1 = np.append(w1,2*ft)
        else:
            x = math.sin(2*np.pi*ft*(l[i]-m/2))
            y = np.pi*(l[i]-m/2)
            z = x/y
            w1 = np.append(w1,z)
        i += 1

    lpFilter = np.convolve(rawData,w1)

    # Original Signal Plot
    x1 = np.arange(0,2000,1)
    plt.subplot(3,1,1)
    plt.plot(x1,rawData)
    plt.title("original signal")

    # 4 Hz Signal Plot
    y1 = np.cos(2*np.pi*4*(x1/2000))
    plt.subplot(3,1,2)
    plt.plot(x1,y1)
    plt.title("4 Hz signal")

    # Lowpass Filter Plot
    x2 = np.arange(0,2020,1)
    plt.subplot(3,1,3)
    plt.plot(x2,lpFilter)
    plt.title("application of lowpass filter")

    plt.tight_layout()
    plt.show()

def highPassFilter(rawData,fs,m,l):

    # Cut-off Frequency
    fc = 280
    # Normalized frequency
    ft = fc/fs
    # First 100 hundred csv values
    rawData = rawData[0:100]
    # Weighted values for highpass
    w = []

    i = 0
    while i < len(l):
        if l[i] == 10:
            w = np.append(w,1-2*ft)
        else:
            x = -1 * math.sin(2*np.pi*ft*(l[i]-m/2))
            y = np.pi*(l[i]-m/2)
            z = x/y
            w = np.append(w,z)
        i += 1

    highpassFilter = np.convolve(rawData,w)
    highpassFilter = highpassFilter[0:100]


    # Original Signal Plot
    x = np.arange(0,100,1)
    plt.subplot(3,1,1)
    plt.plot(x,rawData)
    plt.title("original signal")

    # 330 Hz Signal Plot
    y = np.cos(2*np.pi*330*(x/2000))
    plt.subplot(3,1,2)
    plt.plot(x,y)
    plt.title("330 Hz signal")

    # Highpass Filter Plot
    x = np.arange(0,100,1)
    plt.subplot(3,1,3)
    plt.plot(x,highpassFilter)
    plt.title("application of lowpass filter")

    plt.tight_layout()
    plt.show()

    

def main():
    # Import csv values
    rawData = np.genfromtxt('data-filtering.csv', delimiter = ',')
    # Filter order
    m = 21-1
    # Sampling rate
    fs = 2000
    # Filter length values list
    l = []
    l = np.arange(0,21,1)
    lowPassFilter(rawData,fs,m,l)
    highPassFilter(rawData,fs,m,l)
        

main()


    
