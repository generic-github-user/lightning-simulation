#!/usr/bin/env python
# coding: utf-8

# In[67]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import numba as nb


# In[28]:


y


# In[175]:


resolution = 80
R = resolution
x, y = np.meshgrid(*[np.linspace(0, 1, R)]*2)

u = np.sin(np.pi * x) * np.cos(np.pi * y)+np.random.normal(0, 0.1, [R, R])
v = -np.cos(np.pi * x) * np.sin(np.pi * y)+np.random.normal(0, 0.1, [R, R])

fig = plt.figure(figsize=(10,)*2)
plt.rcParams['figure.figsize'] = [10, 10]
fig, ax = plt.subplots()

# Refer to https://stackoverflow.com/a/40026959/10940584
norm = matplotlib.colors.Normalize()
colors = np.arctan2(u, v)
norm.autoscale(colors)
cmap = matplotlib.cm.inferno(norm(colors))

def Norm(a, b):
    return (a ** 2 + b ** 2) ** 0.5

theta = np.angle(np.apply_along_axis(lambda a: complex(*a), 0, np.stack([u, v])))
ax.quiver(x, y, u, v, theta, cmap='hsv')
plt.axis('off')
plt.show()
