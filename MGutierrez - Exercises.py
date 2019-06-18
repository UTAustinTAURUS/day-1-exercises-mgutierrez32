#!/usr/bin/env python
# coding: utf-8

# ## Exercises
# 
# This will be a notebook for you to work through the exercises during the workshop. Feel free to work on these at whatever pace you feel works for you, but I encourage you to work together! Edit the title of this notebook with your name because I will ask you to upload your final notebook to our shared github repository at the end of this workshop.
# 
# Feel free to google the documentation for numpy, matplotlib, etc.
# 
# Don't forget to start by importing any libraries you need.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import astropy 


# ### Day 1
# 
# #### Exercise 1
# 
#    A. Create an array with 10 evenly spaced values in logspace ranging from 0.1 to 10,000.
# 
#    B. Print the following values: The first value in the array, the final value in the array, and the range of 5th-8th values.
# 
#    C. Append the numbers 10,001 and 10,002 (as floats) to the array. Make sure you define this!
# 
#    D. Divide your new array by 2.
# 
#    E. Reshape your array to be 3 x 4. 
# 
#    F. Multiply your array by itself.
#     
#    G.  Print out the number of dimensions and the maximum value.

# In[2]:


# A.
a = np.logspace(-1,4,10)
print(a)


# In[3]:


# B. 
print(a[0],a[-1],a[4:8])


# In[4]:


# C.
temp = np.array([10001.,10002.])
c = np.append(a,temp)
print(c)


# In[5]:


# D.
d = c/2
print(d)


# In[6]:


# E.
e = d.reshape((3,4)) 
print(e)


# In[7]:


# F. 
f = np.multiply(e,e)
print(f)


# In[8]:


# G. 
print(f.shape)
np.max(f)


# ### Day 2

# #### Exercise 1
# 
#    A. Create an array containing the values 4, 0, 6, 5, 11, 14, 12, 14, 5, 16.
#    B. Create a 10x2 array of zeros.
#    C. Write a for loop that checks if each of the numbers in the first array squared is less than 100. If the statement is true, change that row of your zeros array to equal the number and its square. Hint: you can change the value of an array by stating "zerosarray[i] = [a number, a number squared]". 
#    D. Print out the final version of your zeros array.
#     
# Hint: should you loop over the elements of the array or the indices of the array?

# In[9]:


# A. 
a = [4, 0, 6, 5, 11, 14, 12, 14, 5, 16]


# In[10]:


# B.
b = np.zeros((10,2))


# In[11]:


# C. 
i=0
for x in a:
    if x**2 < 100:
        b[i]=[x,x**2]
    else:
        pass
    i += 1


# In[12]:


# D. 
print(b)


# #### Exercise 2
#     
#    A. Write a function that takes an array of numbers and spits out the Gaussian distribution. Yes, there is a function for this in Python, but it's good to do this from scratch! This is the equation:
#     
# $$ f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp{\frac{-(x - \mu)^2}{2\sigma^2}} $$
# 
#     (Pi is built into numpy, so call it as np.pi.)
# 
#    B. Call the function a few different times for different values of mu and sigma, between -10 < x < 10.
#     
#    C. Plot each version, making sure they are differentiated with different colors and/or linestyles and include a legend. Btw, here's a list of the customizations available in matplotlib:
#     
#     https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.plot.html
#     
#     https://matplotlib.org/gallery/color/named_colors.html
#     
#    D. Save your figure.
#     
# If you have multiple lines with plt.plot(), Python will plot all of them together, unless you write plt.show() after each one. I want these all on one plot.

# In[13]:


# A.
def gauss(mu,sig,x):
    y = (1/(sig)*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sig**2))
    return y


# In[14]:


# B. 
x = np.linspace(-10,10,100)

y1 = gauss(2,3,x)
y2 = gauss(0,1.5,x)
y3 = gauss(-1,1,x)


# In[15]:


# C. & D. 
fig = plt.figure()
plt.plot(x,y1,color='r',label='$\mu = 2$, $\sigma = 3$')
plt.plot(x,y2,color='g',label='$\mu = 0$, $\sigma = 1.5$')
plt.plot(x,y3,color='b',label='$\mu = -1$, $\sigma = 1$')
plt.xlabel(r'x', fontsize=16)
plt.ylabel(r'f(x)', fontsize=16)
plt.title('Sample Gaussians')
plt.legend(loc=1,frameon=True)
fig.savefig('Day2Gaussians.jpg')
plt.plot()


# ### Day 3
# 
# #### Exercise 1
# 
# There is a file in this directory called "histogram_exercise.dat" which consists of of randomly generated samples from a Gaussian distribution with an unknown $\mu$ and $\sigma$. Using what you've learned about fitting data, load up this file using np.genfromtxt, fit a Gaussian curve to the data and plot both the curve and the histogram of the data. As always, label everything, play with the colors, and choose a judicious bin size. 
# 
# Hint: if you attempt to call a function from a library or package that hasn't been imported, you will get an error.

# In[21]:


# your solution here
from scipy.stats import norm
gauss = np.genfromtxt('histogram_exercise.dat').T
mu, sigma = norm.fit(gauss)
x = np.linspace(-1,11,1000)
pdf = norm.pdf(x,mu,sigma)
plt.plot(x, pdf, 'k-', linewidth=2)
plt.hist(gauss, bins=50, color='#770000', alpha=0.4, density=True, fill=True, histtype='step')
plt.title("Fit results: $\mu$ = %.2f, $\sigma$ = %.2f" % (mu, sigma)) 
#Pro-tip: this prints out the values of mu and sigma to 2
#decimal places.

plt.show()


# #### Exercise 2
# 
# Create a 1D interpolation along these arrays. Plot both the data (as points) and the interpolation (as a dotted line). Also plot the value of the interpolated function at x=325. What does the function look like to you?

# In[39]:


x = np.array([0., 50., 100., 150., 200., 250., 300., 350., 400., 450., 500])
y = np.array([0., 7.071, 10., 12.247, 14.142, 15.811, 17.321, 18.708, 20., 21.213, 22.361])

# solution here
from scipy.interpolate import interp1d

interp = interp1d(x, y)
xnew = np.linspace(0, 500, 2000)
ynew = interp(xnew)

plt.plot(xnew, ynew, '--',label='interpolation')
plt.plot(x, y, 'or', label='data')
plt.plot(325,interp(325),'sk', label='interp(325)')
plt.legend() #it looks like a sqrt(x) plot


# ### Day 4
# 
# #### Exercise 1
# 
# Let's practice some more plotting skills, now incorporating units. 
# 
# A. Write a function that takes an array of frequencies and spits out the Planck distribution. That's this equation:
# 
# $$ B(\nu, T) = \frac{2h\nu^3/c^2}{e^{\frac{h\nu}{k_B T}} - 1} $$
# 
# This requires you to use the Planck constant, the Boltzmann constant, and the speed of light from astropy. Make sure they are all in cgs. 
#     
# B. Plot your function in log-log space for T = 25, 50, and 300 K. The most sensible frequency range is about 10^5 to 10^15 Hz. Hint: if your units are correct, your peak values of B(T) should be on the order of 10^-10. Make sure everything is labelled. 

# In[ ]:


# solution here


# #### Exercise 2
# 
# Let's put everything together now! Here's a link to the full documentation for FITSFigure, which will tell you all of the customizable options: http://aplpy.readthedocs.io/en/stable/api/aplpy.FITSFigure.html. Let's create a nice plot of M51 with a background optical image and X-ray contours overplotted.
# 
# The data came from here if you're interested: http://chandra.harvard.edu/photo/openFITS/multiwavelength_data.html
# 
# A. Using astropy, open the X-RAY data (m51_xray.fits). Flatten the data array and find its standard deviation, and call it sigma.
# 
# B. Using aplpy, plot a colorscale image of the OPTICAL data. Choose a colormap that is visually appealing (list of them here: https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html). Show the colorbar. 
# 
# C. Plot the X-ray data as contours above the optical image. Make the contours spring green with 80% opacity and dotted lines. Make the levels go from 2$\sigma$ to 10$\sigma$ in steps of 2$\sigma$. (It might be easier to define the levels array before show_contours, and set levels=levels.)

# In[ ]:


# solution here

