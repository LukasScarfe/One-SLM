import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
import time
from pathlib import Path
import os

### GLOBAL VARS
USE_SAVE = True
DEBUG = False
TIMER = False
PI = np.pi
SAVES_DIR = ''.join([os.path.dirname(os.path.realpath(__file__)),'\\pixel_saves'])

# Taken from prev. code ON THE SCALE OF MM
PIXEL_SPACING = 0.01 # PLACEHOLDER, FILL IN LATER
LAM = 810e-6
E0 = 1 
Z = 1e-6
W0 = 1 # beam waist
ZR = PI * W0 ** 2 / LAM
WZ = W0 * np.sqrt(1 + (Z/ZR)**2)
XDIM,YDIM = 1280,1024
XREP_WIDTH, YREP_WIDTH = 20,20 # Default 20, 20; changed for graph visibility
###



def get_default_pixel_radii():
    """
    Input: 
    Defined below
    
    Output: An array of default pixel radii (calculated from theory) scaled to fill beam

    Called By: get_pixel_beam()
    """
    factors = [1, 0.5, 0.4641, 0.4142, 0.3702, 0.3333, 0.3333, 0.3026, 0.2768,
               np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.2167]
    return WZ * np.array(factors)

def gaussian_intensity(x, y):
    """
    Input: Two function variables

    Output: A 2D gaussian distribution in the space the two input variables. The gaussian has a standard deviation of WZ/sqrt(2) 

    Called By: check_pixel_integrals()
    """
    return E0 * W0/WZ * np.exp(-np.hypot(x,y)**2 / (WZ**2)) 


def check_pixel_integrals(Xpix, Ypix, Rpix, tol=1e-2):
    """
    Input: 
    Defined below

    Output: Boolean indicating whether intensity across all pixels are within tolerance

    Called By: balance_pixel_integrals()
    Calls: gaussian_intensity()
    """
    func = lambda x, y: gaussian_intensity(x, y)
    

    if len(Xpix) == len(Ypix) and len(Xpix) == len(Rpix):
        integrals = []
        for i in range(len(Xpix)):
            R = Rpix[i]
            xcen = Xpix[i]
            ycen = Ypix[i]
            xmin, xmax = xcen - R, xcen + R
            ymin = lambda x: ycen - np.sqrt(R**2 - (x - xcen)**2)
            ymax = lambda x: ycen + np.sqrt(R**2 - (x - xcen)**2)

            result, _ = dblquad(func, xmin, xmax, ymin, ymax)
            integrals.append(result)
        time.sleep(0.1)
    return np.abs(np.max(integrals) - np.min(integrals)) <= tol

def balance_pixel_integrals(numpix, Xpix, Ypix, Rpix):
    """
    Input: 
    Defined below

    Output: Array of pixels' radii

    Called By: get_pixel_beam()
    Calls: check_pixel_integrals()
    """

    inc = 0.005 # Originally at 0.001, increased for performance
    iters = 0
    iterlim = 1000
    if numpix == 1:
        return Rpix
    while not check_pixel_integrals(Xpix, Ypix, Rpix):
        if iters >= iterlim:
            raise RuntimeError("Iteration limit exceeded while balancing integrals.")
        if numpix <= 9:
            Rpix[0] -= inc
        elif numpix == 16:
            Rpix[0:5] -= inc
        iters += 1

    return Rpix

def generate_pixel_geometry(numpix, pixr):
    """
    Input: 
    WZ = beam radius at SLM
    pixr = an array of length equal to the number of pixels filled with the scaled default radius of each pixel

    Output: Array of pixels' centers in x, array of pixels' centers in y, and array of pixels' radii

    Called By: get_pixel_beam()
    """
    angles = None
    if numpix == 1:
        pixx, pixy = [0], [0]
        th = np.linspace(0, 2 * np.pi, 100)

        # Plot the beam circle
        plt.plot(WZ * np.cos(th), WZ * np.sin(th), 'k-', linewidth=1.5, label='Beam')
        
        # Plot each pixel circle
        for i in range(numpix):
            xunit = pixr[i] * np.cos(th) + pixx[i]
            yunit = pixr[i] * np.sin(th) + pixy[i]
            plt.plot(xunit, yunit, linewidth=1.5)
        
        plt.gca().set_aspect('auto')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"{numpix} Dimensional Pixel Geometry")
        plt.grid(True)
        plt.legend()
        if DEBUG:
            plt.show(block=False)
            plt.pause(0.001)

        Xpix = pixx
        Ypix = pixy
        Rpix = pixr
    elif numpix <= 6:
        cartdist = pixr / np.cos(np.pi/2 - np.pi/numpix)
        angles = np.linspace(0, 2*np.pi, numpix, endpoint=False)
        pixx, pixy = cartdist * np.cos(angles), cartdist * np.sin(angles)
        th = np.linspace(0, 2 * np.pi, 100)

        # Plot the beam circle
        plt.plot(WZ * np.cos(th), WZ * np.sin(th), 'k-', linewidth=1.5, label='Beam')
        
        # Plot each pixel circle
        for i in range(numpix):
            xunit = pixr[i] * np.cos(th) + pixx[i]
            yunit = pixr[i] * np.sin(th) + pixy[i]
            plt.plot(xunit, yunit, linewidth=1.5)
        
        plt.gca().set_aspect('auto')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"{numpix} Dimensional Pixel Geometry")
        plt.grid(True)
        plt.legend()
        if DEBUG:
            plt.show(block=False)
            plt.pause(0.001)

        Xpix = pixx
        Ypix = pixy
        Rpix = pixr

    elif numpix <= 9:
        cartdist = pixr[1:] / np.cos(np.pi/2 - np.pi/(numpix-1))
        angles = np.linspace(0, 2*np.pi, numpix-1, endpoint=False)
        pixx, pixy = cartdist * np.cos(angles), cartdist * np.sin(angles)
        th = np.linspace(0, 2 * np.pi, 100)

        # Plot the beam circle
        plt.plot(WZ * np.cos(th), WZ * np.sin(th), 'k-', linewidth=1.5, label='Beam')

        plt.plot(pixr[0] * np.cos(th), pixr[0] * np.sin(th), linewidth=1.5)

        for i in range(numpix-1):
            xunit = pixr[i] * np.cos(th) + pixx[i]
            yunit = pixr[i] * np.sin(th) + pixy[i]
            plt.plot(xunit, yunit, linewidth=1.5)

        plt.gca().set_aspect('auto')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"{numpix} Dimensional Pixel Geometry")
        plt.grid(True)
        plt.legend()
        if DEBUG:
            plt.show(block=False)
            plt.pause(0.001)

        Xpix = np.concatenate(([0],pixx))
        Ypix = np.concatenate(([0],pixy))
        Rpix = pixr

    elif numpix == 16:
        cartdist = np.concatenate([
            pixr[0] / np.cos(np.pi/2 - np.pi/5) * np.ones(5),
            (WZ - pixr[0])*np.ones(11)
        ])
        angles = np.deg2rad(np.concatenate([
            np.arange(0, 360, 72),
            19.44 + np.arange(11) * 32.112
        ]))
        pixx, pixy = cartdist * np.cos(angles), cartdist * np.sin(angles)
        th = np.linspace(0, 2 * np.pi, 100)

        # Plot the beam circle
        plt.plot(WZ * np.cos(th), WZ * np.sin(th), 'k-', linewidth=1.5, label='Beam')
        
        # Plot each pixel circle
        for i in range(numpix):
            xunit = pixr[i] * np.cos(th) + pixx[i]
            yunit = pixr[i] * np.sin(th) + pixy[i]
            plt.plot(xunit, yunit, linewidth=1.5)
        
        plt.gca().set_aspect('auto')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"{numpix} Dimensional Pixel Geometry")
        plt.grid(True)
        plt.legend()
        if DEBUG:
            plt.show(block=False)
            plt.pause(0.001)

        Xpix = pixx
        Ypix = pixy
        Rpix = pixr

    else:
        raise ValueError("Unsupported number of pixels")

    return np.array(Xpix), np.array(Ypix), np.array(Rpix)

def plot_pixel_gaussians(Xpix, Ypix, Rpix, mode): # Get real measurements
    """
    Input: 
    Defined below

    Output: 2D matrix of field

    Called By: get_pixel_beam()
    """
    I0 = 1
    xarr = np.linspace(-XREP_WIDTH/2, XREP_WIDTH/2, XDIM)
    yarr = np.linspace(-YREP_WIDTH/2, YREP_WIDTH/2, YDIM)
    X, Y = np.meshgrid(xarr, yarr)
    Z = np.zeros([YDIM, XDIM])
    if mode == len(Xpix): # plots all pixels
        for i in range(len(Xpix)):
            Ztemp = I0 * np.exp(-np.hypot((X - Xpix[i]), (Y - Ypix[i]))**2 / (Rpix[i]**2))
            for j in range(YDIM):
                for k in range(XDIM):
                    if Ztemp[j,k] <= I0/np.exp(1):
                        Ztemp[j,k] = 0
            Z = Z + Ztemp
    else: # plots specific pixel
        for i in range(len(Xpix)):
            if i == mode:
                    Ztemp = I0 * np.exp(-np.hypot((X - Xpix[i]), (Y - Ypix[i]))**2 / (Rpix[i]**2))
                    for j in range(YDIM):
                        for k in range(XDIM):
                            if Ztemp[j,k] <= I0/np.exp(1):
                                Ztemp[j,k] = 0
                    Z = Z + Ztemp

    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, cmap='Greys_r',linewidth=0, antialiased=False)
    fig.colorbar(surf)
    # plt.gca().set_aspect('equal')
    ax.view_init(elev=90, azim=0)
    # plt.pcolormesh(X, Y, Z, shading='gouraud',cmap='Greys_r')
    plt.title(f"SLM Hologram for d = {len(Xpix)} and mode = {mode}th pixel")
    if DEBUG:
        plt.show()
    return Z

def adjust_pixel_spacing(numpix, Xpix, Ypix, Rpix): 
    """
    Input: 
    Xpix = array of pixels' centers in x
    Ypix = array of pixels' centers in y
    Rpix = array of pixels' radii

    Output: Array of pixels' centers in x, array of pixels' centers in y, and array of pixels' radii

    Called By: get_pixel_beam()
    """
    inc = 0.005
    iterlim = 1000

    if numpix == 1:
        return Xpix, Ypix, Rpix
    if numpix <= 6:
        for i in range(numpix):
            for j in range(i+1, numpix):
                sep = np.sqrt((Xpix[i] - Xpix[j])**2 + (Ypix[i] - Ypix[j])**2) - (Rpix[i] + Rpix[j])
                iters = 0
                while sep < PIXEL_SPACING: # For certain spacing values, rounding causes more iterations than in MATLAB
                    if iters >= iterlim:
                        raise RuntimeError("Number of iterations exceeded iteration limit.")
                    Rpix[:] -= inc
                    sep = np.sqrt((Xpix[i] - Xpix[j])**2 + (Ypix[i] - Ypix[j])**2) - (Rpix[i] + Rpix[j])
                    iters += 1

    elif numpix <= 9:
        for i in range(1, numpix):
            for j in range(i+1, numpix):
                sep = np.sqrt((Xpix[i] - Xpix[j])**2 + (Ypix[i] - Ypix[j])**2) - (Rpix[i] + Rpix[j])
                iters = 0
                while sep < PIXEL_SPACING:
                    if iters >= iterlim:
                        raise RuntimeError("Number of iterations exceeded iteration limit.")
                    Rpix[1:] -= inc
                    sep = np.sqrt((Xpix[i] - Xpix[j])**2 + (Ypix[i] - Ypix[j])**2) - (Rpix[i] + Rpix[j])
                    iters += 1

        sep = np.sqrt((Xpix[0] - Xpix[1])**2 + (Ypix[0] - Ypix[1])**2) - (Rpix[0] + Rpix[1])
        iters = 0
        while sep < PIXEL_SPACING:
            if iters >= iterlim:
                raise RuntimeError("Number of iterations exceeded iteration limit.")
            Rpix[0] -= 2 * inc
            sep = np.sqrt((Xpix[0] - Xpix[1])**2 + (Ypix[0] - Ypix[1])**2) - (Rpix[0] + Rpix[1])
            iters += 1

    elif numpix == 16:
        for i in range(5):
            for j in range(i+1, 5):
                sep = np.sqrt((Xpix[i] - Xpix[j])**2 + (Ypix[i] - Ypix[j])**2) - (Rpix[i] + Rpix[j])
                iters = 0
                while sep < PIXEL_SPACING:
                    if iters >= iterlim:
                        raise RuntimeError("Number of iterations exceeded iteration limit.")
                    Rpix[0:5] -= inc
                    sep = np.sqrt((Xpix[i] - Xpix[j])**2 + (Ypix[i] - Ypix[j])**2) - (Rpix[i] + Rpix[j])
                    iters += 1

        for i in range(5, 16):
            for j in range(i+1, 16):
                sep = np.sqrt((Xpix[i] - Xpix[j])**2 + (Ypix[i] - Ypix[j])**2) - (Rpix[i] + Rpix[j])
                iters = 0
                while sep < PIXEL_SPACING:
                    if iters >= iterlim:
                        raise RuntimeError("Number of iterations exceeded iteration limit.")
                    Rpix[5:16] -= inc
                    sep = np.sqrt((Xpix[i] - Xpix[j])**2 + (Ypix[i] - Ypix[j])**2) - (Rpix[i] + Rpix[j])
                    iters += 1

    else:
        raise ValueError("Specified number of pixels not available. Try 1-9, 16.")

    return Xpix, Ypix, Rpix

def check_previous_scans(numpix):

    filename = f'n{numpix}s{PIXEL_SPACING}.txt'
    filepath = ''.join([SAVES_DIR,f'\\{filename}'])
    try:
        file = open(filepath,'r')
    except FileNotFoundError:
        print("File Not Found. Continuing procedure.")
        return [],[],[]
    line = file.readline() 
    Xpix = np.array(line.split(',')).astype(float)
    line = file.readline()
    Ypix = np.array(line.split(',')).astype(float)
    line = file.readline()
    Rpix = np.array(line.split(',')).astype(float)
    file.close()
    return Xpix, Ypix, Rpix
    
def save_settings(numpix, Xpix, Ypix, Rpix):
    filename = f'n{numpix}s{PIXEL_SPACING}.txt'
    filepath = ''.join([SAVES_DIR, f'\\{filename}'])
    Xpix_str = ','.join(np.array(Xpix).astype(str))
    Ypix_str = ','.join(np.array(Ypix).astype(str))
    Rpix_str = ','.join(np.array(Rpix).astype(str))
    file = open(filepath, 'x')
    file.write(f'{Xpix_str}\n')
    file.write(f'{Ypix_str}\n')
    file.write(f'{Rpix_str}\n')
    file.close()
    





def get_pixel_beam(numpix,mode):
    """
    Input: 
    Defined below

    Output: 
    Amplitude = amplitude of field 

    Called By: lv_2_beam()
    Calls: get_default_pixel_radii(), generate_pixel_geometry(), adjust_pixel_spacing(), balance_pixel_integrals(), plot_pixel_gaussians()
    """

    start = time.time()

    if USE_SAVE:
        if not os.path.exists(SAVES_DIR):
            os.makedirs(SAVES_DIR)
        Xpix,Ypix,Rpix = check_previous_scans(numpix)
    else:
        Xpix = []
        Ypix = []
        Rpix = []

    time1 = time.time()

    if len(Xpix) > 0:
        Amplitude = plot_pixel_gaussians(Xpix, Ypix, Rpix, mode)

        time2 = time.time()

        if TIMER:
            print(f"Mark 1: {time1 - start} seconds")
            print(f"Mark 2: {time2 - time1} seconds")
            print(f"Total: {time2 - start} seconds")

        return Amplitude
    else:

    # Add function that checks memory for previous settings

        pixr = get_default_pixel_radii()

        time1 = time.time()
        

        Xpix, Ypix, Rpix = generate_pixel_geometry(numpix, pixr[numpix-1]*np.ones(numpix))

        time2 = time.time()
        
        # Adjust pixel spacing to meet minimum spacing constraints
        Xpix, Ypix, Rpix = adjust_pixel_spacing(numpix, Xpix, Ypix, Rpix)

        time3 = time.time()

        # Balance pixel integrals relative to the Gaussian beam profile
        Rpix = balance_pixel_integrals(numpix, Xpix, Ypix, Rpix)

        time4 = time.time()

        # Visualize the final pixel arrangement
        Amplitude = plot_pixel_gaussians(Xpix, Ypix, Rpix, mode)

        time5 = time.time()

        if USE_SAVE:
            save_settings(numpix, Xpix, Ypix, Rpix)
        
        if TIMER:
            print(f"Mark 1: {time1 - start} seconds")
            print(f"Mark 2: {time2 - time1} seconds")
            print(f"Mark 3: {time3 - time2} seconds")
            print(f"Mark 4: {time4 - time3} seconds")
            print(f"Mark 5: {time5 - time4} seconds")
            print(f"Total: {time5 - start} seconds")

        return Amplitude

def lv_2_beam(numpix, mode, phase): # All params defined by Panel
    """
    Input: 
    numpix = number of pixels
    mode = which pixel to be displayed
    phase = relative phase of pixel (in radians)

    Output: the total field (amplitude and phase) to SLM

    Calls: get_pixel_beam
    """
    Amplitude = get_pixel_beam(numpix, mode)
    Phase = np.ones(np.shape(Amplitude))*phase
    # Holo = Amplitude*np.exp(1j*phase)
    Holo = np.array([Amplitude, Phase])
    return Holo
    
    
# lv_2_beam(16,0,0)