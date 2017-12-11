from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

def time_and_ims(fn):
    """Return the timestamps and images from a Xenics xvi file
    
    Note that there is something wrong with the first image so we ignore it.
    
    Parameters
    ----------
    fn: filename
    
    Returns
    -------
    times: double
        Time since the epoch
    
    ims: numpy array
        
    
    """
    with open(fn, mode='rb') as file: # b is important -> binary
        fileContent = file.read()
        header_shorts = np.frombuffer(fileContent[:1024], dtype='int32')
        npix = (len(fileContent) - 1024)//2
        pix_per_frame = header_shorts[4]*header_shorts[5] + 32
        nf = npix//pix_per_frame
        ims = np.empty( (nf, header_shorts[5], header_shorts[4]), dtype='int16')
        headers = np.empty( (nf, 16), dtype='uint32')
        tstart = np.empty(nf, dtype='uint64')
        for i in range(nf):
            ims[i] = np.frombuffer(fileContent[1024 + pix_per_frame*i*2:\
                1024 + (pix_per_frame*(i+1) - 32)*2],dtype='int16').reshape(header_shorts[5],header_shorts[4])
            #Uncomment out the following line to get the full images.
            #headers[i] = np.frombuffer(fileContent[1024 + (pix_per_frame*(i+1) - 32)*2:1024 + (pix_per_frame*(i+1))*2],dtype='uint32')
            tstart[i] = np.frombuffer(fileContent[1024 + (pix_per_frame*(i+1) - 26)*2:\
                1024 + (pix_per_frame*(i+1) - 22)*2],dtype='uint64')
        return tstart[1:]/1e6, ims[1:]

if __name__=="__main__":
    fn = '/Users/mireland/Google Drive/Harry PhD/testPicsCamera.xvi'
    #fn = '/Users/mireland/Google Drive/Harry PhD/4100nm.xvi'  #90% null depth
    #fn = '/Users/mireland/Google Drive/Harry PhD/4000nm.xvi' #50% null depth
    #fn = '/Users/mireland/Google Drive/Harry PhD/4150nm.xvi' #90% null depth
    #fn = '/Users/mireland/Google Drive/Harry PhD/4200nm.xvi'  #90% null depth
    fn = '/Users/mireland/Google Drive/Harry PhD/4220nm.xvi'  #90% null depth
    tstart, ims = time_and_ims(fn)
    #Show how to create a mean subtracted image
    fluxes = np.sum(np.sum(ims,2),1)
    low_ix = np.where(fluxes < np.percentile(fluxes,40))[0]
    high_ix = np.where(fluxes > np.percentile(fluxes,60))[0]
    imsub = np.mean(ims[high_ix],axis=0) - np.mean(ims[low_ix],axis=0)
    plt.figure(1)
    plt.clf()
    plt.imshow(imsub)
    #plt.imshow(np.arcsinh(imsub))
    plt.title("Subtracted Image, arcsinh stretch")
    plt.show()

    