# Micah Clarke
# ID: 1001288866

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from scipy import ndimage


def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        #if image.ndim == 2:
        plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()



def gather_images():
    # Gathers each tiff and jpg file and stores the values into an np array
    image_name = 'boat.512.tiff'
    im_data = img.imread(image_name)
    boat = np.array(im_data)

    image_name = 'clock-5.1.12.tiff'
    im_data = img.imread(image_name)
    clock = np.array(im_data)
    
    image_name = 'man-5.3.01.tiff'
    im_data = img.imread(image_name)
    man = np.array(im_data)

    image_name = 'tank-7.1.07.tiff'
    im_data = img.imread(image_name)
    tank = np.array(im_data)

    image_name = 'darinGrayNoise.jpg'
    im_data = img.imread(image_name)
    darinGrayNoise = np.array(im_data)

    images = [boat,clock,man,tank]
    # 3(a) - Display the original image in its own figure
    show_images(images, cols = 2, titles = ['Boat','Clock','Man','Tank'])
    # 3(b) - lowpass filter on each of the 4 tiff files to blur the images.
    blur_image(boat,clock,man,tank)
    # 3(c) - highpass filter on each of the 4 tiff files for edge detection.
    edge_detection(boat,clock,man,tank)
    # 3(d) - remove the noise from the jpg file
    remove_noise(darinGrayNoise)

    
def remove_noise(darinGrayNoise):
    # Displays the original salt and pepper jpg file
    images = [darinGrayNoise]
    show_images(images, cols = 1, titles = ['salt and pepper noise'])
    
    # 10 point moving average
    avg = np.array([.1,.1,.1,.1,.1,.1,.1,.1,.1,.1])
    
    darin_convolved = np.empty([0,649])
    i = 0
    for row in darinGrayNoise:
        # Convolve first row
        result = np.convolve(row,avg)
        # Reshape result array
        result = np.reshape(result,result.size,order='F').reshape((1,649))
        darin_convolved = np.insert(darin_convolved,[i],result,axis = 0)
        i += 1

    # Displays the removed noise jpg file
    images = [darin_convolved]
    show_images(images, cols = 1, titles = ['Removed noise by average'])

    
    # Displays the median filter jpg file
    outputImage = ndimage.median_filter(darinGrayNoise,5)
    images = [outputImage]
    show_images(images, cols = 1, titles = ['Median Filter'])
    
def edge_detection(boat,clock,man,tank):
 
    h = [1,-1]
    i = 0

    # Arrays for convolved jiffs for edge detection, these include the transient regions of
    # the calculated filter
    boat_highpass = np.empty([0,513])
    clock_highpass = np.empty([0,257])
    man_highpass = np.empty([0,1025])
    tank_highpass = np.empty([0,513])
    
    for row in boat:
        result4 = np.convolve(row,h)
        result4 = np.reshape(result4,result4.size,order='F').reshape((1,513))
        boat_highpass = np.insert(boat_highpass,[i],result4,axis = 0)
        i += 1  

    i = 0
    for row in clock:
        result5 = np.convolve(row,h)
        result5 = np.reshape(result5,result5.size,order='F').reshape((1,257))
        clock_highpass = np.insert(clock_highpass,[i],result5,axis = 0)
        i += 1  
    
    i = 0
    for row in man:
        result6 = np.convolve(row,h)
        result6 = np.reshape(result6,result6.size,order='F').reshape((1,1025))
        man_highpass = np.insert(man_highpass,[i],result6,axis = 0)
        i += 1

    i = 0
    for row in tank:
        result7 = np.convolve(row,h)
        result7 = np.reshape(result7,result7.size,order='F').reshape((1,513))
        tank_highpass = np.insert(tank_highpass,[i],result7,axis = 0)
        i += 1

    images = [boat_highpass,clock_highpass,man_highpass,tank_highpass]
    show_images(images, cols = 2, titles = ['Boat','Clock','Man','Tank'])
    

def blur_image(boat,clock,man,tank):

    # Arrays to store the convolved values from the lowpass filter
    boat_convolved = np.empty([0,521])
    clock_convolved = np.empty([0,265])
    man_convolved = np.empty([0,1033])
    tank_convolved = np.empty([0,521])
      
    # 10 point moving average
    avg = np.array([.1,.1,.1,.1,.1,.1,.1,.1,.1,.1])

 
    i = 0
    for row in boat:
        # Convolve first row
        result = np.convolve(row,avg)
        # Reshape result array
        result = np.reshape(result,result.size,order='F').reshape((1,521))
        boat_convolved = np.insert(boat_convolved,[i],result,axis = 0)
        i += 1

    i = 0
    for row in clock:
        # Convolve first row
        result1 = np.convolve(row,avg)
        # Reshape result array
        result1 = np.reshape(result1,result1.size,order='F').reshape((1,265))
        clock_convolved = np.insert(clock_convolved,[i],result1,axis = 0)
        i += 1
    
    i = 0
    for row in man:
        # Convolve first row
        result2 = np.convolve(row,avg)
        # Reshape result array
        result2 = np.reshape(result2,result2.size,order='F').reshape((1,1033))
        man_convolved = np.insert(man_convolved,[i],result2,axis = 0)
        i += 1
        
    i = 0
    for row in tank:
        # Convolve first row
        result3 = np.convolve(row,avg)
        # Reshape result array
        result3 = np.reshape(result3,result3.size,order='F').reshape((1,521))
        tank_convolved = np.insert(tank_convolved,[i],result3,axis = 0)
        i += 1

    images = [boat_convolved,clock_convolved,man_convolved,tank_convolved]
    show_images(images, cols = 2, titles = ['Boat','Clock','Man','Tank'])

    
def main():
    gather_images()
    


main()

