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

from scipy.stats import norm, bernoulli

N = 200
b_GP = 1  # direct influence of grandparents on parents
b_GC = 0  # direct influence of grandparents on children
b_PC = 1  # direct influence of parents on children
b_U = 2  # direct influence of neighborhood on parents and children

U = 2 * bernoulli.rvs(p=0.5,
                      loc=0,
                      size=N) - 1

G = norm.rvs(loc=0,
             scale=1,
             size=N)

P = norm.rvs(loc=b_GP * G + b_U * U,
             scale=1,
             size=N)

C = norm.rvs(loc=b_GC * G + b_U * U + b_PC * P,
             scale=1,
             size=N)

df = pd.DataFrame({'C': C, 'P': P, 'G': G, 'U': U})

with pm.Model() as m_6_11:
    a = pm.Normal('a', mu=0, sigma=0.25)
    b_GC = pm.Normal('b_GC', mu=0, sigma=0.5)
    b_PC = pm.Normal('b_PC', mu=1, sigma=0.5)
    sigma = pm.Exponential('sigma', lam=10)

    mu = a + b_GC * df['G'] + b_PC * df['P']
    C = pm.Normal('C', mu=mu, sigma=sigma, observed=df['C'])

    trace_6_11 = pm.sample(2000, init='advi', tune=2000, return_inferencedata=False)

print('******************Summary for Trace 6.11*******************************')
print(az.summary(trace_6_11,
                 var_names=['a', 'b_GC', 'b_PC', 'sigma'],
                 hdi_prob=0.89))


with pm.Model() as m_6_12:
    a = pm.Normal('a', mu=0, sigma=1)
    b_GC = pm.Normal('b_GC', mu=0, sigma=1)
    b_PC = pm.Normal('b_PC', mu=0, sigma=1)
    b_U = pm.Normal('b_U', mu=0, sigma=1)
    sigma = pm.Exponential('sigma', lam=1)

    mu = pm.Deterministic('mu', a + b_GC * df['G'] + b_PC * df['P'] + b_U * df['U'])
    C = pm.Normal('C', mu=mu, sigma=sigma, observed=df['C'])

    trace_6_12 = pm.sample(2000, init='advi', tune=6000)

print('******************Summary for Trace 6.12*******************************')
print(az.summary(trace_6_12,
                 var_names=['a', 'b_GC', 'b_PC', 'b_U', 'sigma'],
                 hdi_prob=0.89))

with pm.Model() as m_6_13:
    a = pm.Normal('a', mu=0, sigma=0.25)
    b_GC = pm.Normal('b_GC', mu=0, sigma=0.25)
    b_PC = pm.Normal('b_PC', mu=1, sigma=0.25)
    b_GP = pm.Normal('b_GP', mu=1, sigma=0.25)
    b_U = pm.Normal('b_U', mu=2, sigma=0.25)
    sigma = pm.Exponential('sigma', lam=1)

    mu_P = b_GP * df['G'] + b_U * df['U']
    P = pm.Normal('P', mu=mu_P, sigma=sigma, observed=df['P'])
    mu_C = a + b_GC * df['G'] + b_PC * P + b_U * df['U']

    C = pm.Normal('C', mu=mu_C, sigma=sigma, observed=df['C'])

    trace_6_13 = pm.sample(2000, init='advi', tune=6000, return_inferencedata=False)

print('******************Summary for Trace 6.13 *******************************')
print(az.summary(trace_6_13,
                 var_names=['a', 'b_GC', 'b_PC', 'b_GP', 'b_U', 'sigma'],
                 hdi_prob=0.89))
