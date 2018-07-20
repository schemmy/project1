# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-07-17 18:09:08
# @Last Modified by:   chenxinma
# @Last Modified at:   2018-07-20 13:48:29


from pyscipopt import Model, quicksum, multidict
import numpy as np
np.random.seed(1)

# 1-RDC, 1-FDC, 1-Sku 

I = 130
alpha = 0.1
theta = 0.2
p = 1
M = p
n = 10
mu, sigma = 120, 30
D0 = np.clip(np.random.normal(mu, sigma, n), 1, None)
mu, sigma = 40, 10
D1 = np.clip(np.random.normal(mu, sigma, n), 1, None)

model = Model('Allocation')


w = {}

w[0] = model.addVar(vtype='C', 
                    lb=0.0,
                    ub=1.0,
                    name='RDC_reserve_ratio')

w[1] = model.addVar(vtype='C', 
                    lb=0.0,
                    ub=1.0,
                    name='FDC_allo_ratio')

x0 = {}
x1 = {}
s0 = {}

for j in range(n):
    x0[(1,j)] = model.addVar(vtype='C',
                        lb=0.0,
                        name='Fulfill_from_RDC')
    x1[(1,j)] = model.addVar(vtype='C',
                        lb=0.0,
                        name='Fulfill_from_FDC')

    s0[j] = model.addVar(vtype='C',
                    lb=0.0,
                    name='Slack')

# RDC capacity
for j in range(n):
    model.addCons( D0[j] + (1-theta)*(1-alpha)*x0[(1,j)] <= w[0]*I + s0[j], name='RDC_capacity')

# RDC stock
for j in range(n):
    model.addCons( x1[(1,j)] <= w[1]*I, name='FDC_stock')

# Demand satisfied
for j in range(n):
    model.addCons( x1[(1,j)]+x0[(1,j)] == D1[j], name='Demand')

# Sum to 1
model.addCons( w[0]+w[1] <= 1, name='Sum1')

# Obj
model.setObjective( quicksum(M*s0[i] for i in range(n)) + \
    (alpha+theta*(1-alpha)) * p * quicksum(x0[(1,j)]  for j in range(n) ), 'minimize')

model.optimize()

print('Optimal value:', model.getObjVal()/n)


print (model.getVal(w[0]), model.getVal(w[1]))
for i in range(10):
    print (D0[i], D1[i], model.getVal(x0[(1,i)]), model.getVal((x1[(1,i)])), model.getVal(s0[i]))