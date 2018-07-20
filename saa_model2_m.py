# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-07-17 18:09:08
# @Last Modified by:   chenxinma
# @Last Modified at:   2018-07-20 14:30:19


from pyscipopt import Model, quicksum, multidict
import numpy as np
np.random.seed(1)

# 1-RDC, m-FDC, 1-Sku 

m = 3
I = {0: 100, 1:0, 2:10, 3:2}  # Inventory
alpha = 0.1
theta = 0.2
p = 1
n = 10

D = {}
SV = {0: [120, 30],
      1: [40, 10],
      2: [40, 10],
      3: [40, 10]
      }
for i in range(m+1):
    D[i] = np.clip(np.random.normal(SV[i][0], SV[i][1], n), 1, None)



model = Model('Allocation')


w = {}
for i in range(m+1):
    w[i] = model.addVar(vtype='C', 
                    lb=0.0,
                    ub=1.0,
                    name='DC%i_reserve_ratio' %i)

x = {}
for i in range(m+1):
    for j in range (n):
        x[(i,j)] = model.addVar(vtype='C',
                        lb=0.0,
                        name='DC%i_stockout' %i)



for j in range(n):
    model.addCons( x[(0,j)] >= (1-alpha) * (1-theta) * quicksum(x[(i,j)]\
                  for i in range(1,m+1)) - w[0] * I[0] + D[0][j], name='RDC_out')

for i in range(1, m+1):
    for j in range(n):
        model.addCons( x[(i,j)] >= D[i][j] - w[i] * I[0] - I[i], name='FDC%i_out' %i)

# Sum to 1
model.addCons( quicksum(w[i] for i in range(m+1)) <= 1, name='Sum1')

# Obj
model.setObjective(
    quicksum( x[(0,j)] * p for j in range(n))
    + quicksum( x[(i,j)] * p * (alpha+theta*(1-alpha)) for i in range(1, m+1) for j in range(n)), 'minimize')

model.optimize()

print('Optimal value:', model.getObjVal()/n)


print('Optimal Solution:')
for i in range(m+1):
    print ('delta_%i: %2.5f,  ' %(i, model.getVal(w[i])), end='')
print('')
print ('----------------------------------------------------------------------------------')
for i in range(m+1):
    print ('{0:9s}, {1:7s}, '.format('  dmnd[%i]' %i, 'dire[%i]' %i), end='')
print('')
print ('----------------------------------------------------------------------------------')
for j in range(n):
    print('{0:9.4f}, {1:7.4f}, '.format(D[0][j], model.getVal(w[0])*I[0]), end='')
    for i in range(1, m+1):
        print('{0:9.4f}, {1:7.4f}, '.format(D[i][j], model.getVal(x[(i,j)])*(1-alpha)), end='')
    print ('')

