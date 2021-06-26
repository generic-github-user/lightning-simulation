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
