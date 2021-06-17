#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 08:51:24 2021

@author: somnath
This is merely a revision of an example from Chapter 6. The example 
deals with the influence of grandparents on the educational achievement 
of children. It is natural to assume that grandparents influence their 
own children and so the causal DAG of influences looks like:
    
    G ---> P ----> C
    |              ^      
    |______________|
    
But there might be unobserved variables U that influence both parents 
and children that are not shared with grandparents. An example of such 
an unobserved variable is the neighborhood in which parents and children 
live. In this setting, the causal DAG looks like:

                   -----U-----
                   |         |
                   v         v
            G----->P-------->C
            |                ^ 
            |________________|
        
However in this setting, the variable P is a collider and conditioning 
on it creates an association between the variables G and U. Since 
G and U each influence C, conditioning on P biases the inference 
of G--->C.    
"""
import arviz as az
import pymc3 as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm, bernoulli

N = 200
b_GP = 1 # direct influence of grandparents on parents
b_GC = 0 # direct influence of grandparents on children
b_PC = 1 # direct influence of parents on children
b_U = 2  # direct influence of neighborhood on parents and children

RANDOM_SEED = 12345678

U = 2 * bernoulli.rvs(p=0.5, 
                      loc=0, 
                      size=N, 
                      random_state=RANDOM_SEED) - 1 

G = norm.rvs(loc=0, 
             scale=1, 
             size=N, 
             random_state=RANDOM_SEED)

P = norm.rvs(loc=b_GP * G + b_U * U, 
             scale=1, 
             size=N, 
             random_state=RANDOM_SEED)

C = norm.rvs(loc=b_GC * G + b_U * U + b_PC * P, 
             scale=1, 
             size=N, 
             random_state=RANDOM_SEED)


df = pd.DataFrame({'C': C, 'P': P, 'G': G, 'U': U})

with pm.Model() as m_6_11:
    a = pm.Normal('a', mu=0, sigma=0.25)
    b_GC = pm.Normal('b_GC', mu=0, sigma=0.5)
    b_PC = pm.Normal('b_PC', mu=1, sigma=0.5)
    sigma = pm.Exponential('sigma', lam=10)
    
    mu = a + b_GC * df['G'] + b_PC * df['P']
    C = pm.Normal('C', mu=mu, sigma=sigma, observed=df['C'])
    
    trace_6_11 = pm.sample(2000, init='advi', tune=2000, return_inferencedata=False)

print(az.summary(trace_6_11))
