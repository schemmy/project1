# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-07-17 18:09:08
# @Last Modified by:   chenxinma
# @Last Modified at:   2018-07-18 15:40:29


from pyscipopt import Model, quicksum, multidict
import random

# 1-RDC, 2-FDC, 5-Sku 

I = {1:30, 2:60, 3:65, 4:10, 5:200}  # Inventory
L = 6 # Sku capacity upper bound
m = 2
n = 5
alpha = 0.1

d = {(0,1):30,    (0,2):80,    (0,3):60,    (0,4):10,    (0,5):100,
     (1,1):6,     (1,2):9,     (1,3):30,    (1,4):2,     (1,5):120,    
     (2,1):10,    (2,2):7,     (2,3):26,    (2,4):7,     (2,5):70,    
    }

c0 = {(0,1):4,   (0,2):3,     (0,3):10,    (0,4):14,    (0,5):2,
     (1,1):4,     (1,2):3,     (1,3):10,    (1,4):14,     (1,5):2,    
     (2,1):4,     (2,2):3,     (2,3):10,    (2,4):14,     (2,5):2,    
     }

c1 = {(1,1):2,     (1,2):1,     (1,3):2,    (1,4):5,     (1,5):1,    
      (2,1):3,     (2,2):2,     (2,3):3,    (2,4):7,     (2,5):2,    
     }

c2 = {(1,1):1,     (1,2):1,     (1,3):1,    (1,4):1,     (1,5):1,    
      (2,1):1,     (2,2):1,     (2,3):1,    (2,4):1,     (2,5):1,    
     }


# Create variables


m = 5 # Number of FDC
n = 1000 # Number of Sku
L = 400 # Sku capacity upper bound
alpha = 0.1
random.seed(1)

I, d, c0, c1, c2 = {}, {}, {}, {}, {}

for i in range(m+1):
    for j in range(n):
        if i == 0:
            d[(i,j)] = random.random() * random.randint(500, 1000)
            c0[(i,j)] = random.random() * random.randint(20, 100)
        d[(i,j)] = random.random() * d[(0,j)]
        c0[(i,j)] = c0[(0,j)]*0.6
        c1[(i,j)] = random.random() * random.randint(10, 20)
        c2[(i,j)] = 2

for j in range(n):
    I[j] = d[(0,j)] * (random.random()+0.8)

M = list(range(m+1))
N = I.keys() 
w = {}
x = {}
y = {}

model = Model('Allocation')

for i in M:
    for j in N:
        w[i,j] = model.addVar(vtype='C', 
                              lb=0.0,
                              name='w(%s,%s)' % (i,j))

for i in M[1:]:
    for j in N:
        x[i,j] = model.addVar(vtype='B', name='x(%s,%s)' % (i,j))

for i in M:
    for j in N:
        y[i,j] = model.addVar(vtype='C', 
                              lb=0.0,
                              name='y(%s,%s)' % (i,j))

# Capacity constraints
model.addCons(sum(x[i,j] for i in M[1:] for j in N) <= L, name='Capacity')

# Weights constraints
for j in N:
    model.addCons(sum(w[i,j] for i in M) <= 1, name='Weights(%d) upper bound' % j)

for i in M:
    for j in N:
        model.addCons(w[i,j] >= 0, name='Weight(%d, %d) geq 0' %(i,j))

# Relation between x and w
for i in M[1:]:
    for j in N:
        model.addCons(x[i,j] >= w[i,j], name='Consistent(%s,%s)' % (i,j))

# Max 
for i in M:
    for j in N:
        model.addCons(y[i,j] >= d[i,j]-w[i,j]*I[j], name='Logic max(%s,%s)' % (i,j))

# RDC reserve
for j in N:
    model.addCons(w[0,j] >= 0.2, name='RDC Reserve')

model.data = w,x

# Objective
model.setObjective(\
    quicksum(c0[i,j]*(1-alpha)*y[i,j]  for (i,j) in w)\
    + quicksum(c1[i,j]*alpha*y[i,j] for (i,j) in x)\
    + quicksum(c2[i,j]*x[i,j] for (i,j) in x)\
     , 'minimize')

model.optimize()

print('Optimal value:', model.getObjVal())


# for i in M:
#     for j in N:
#         print ('%.2f  ' %abs(model.getVal(w[i,j])), end='')
#     print ('')