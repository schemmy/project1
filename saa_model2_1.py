# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-07-17 18:09:08
# @Last Modified by:   chenxinma
# @Last Modified at:   2018-07-20 13:47:42


from pyscipopt import Model, quicksum, multidict
import numpy as np
np.random.seed(1)

# 1-RDC, 1-FDC, 1-Sku 

I = 130
alpha = 0.1
theta = 0.2
p = 1
n = 10

D = {}
mu, sigma = 120, 30
D[0] = np.random.normal(mu, sigma, n)
mu, sigma= 40, 10
D[1] = np.random.normal(mu, sigma, n)

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

x = {}

for j in range (n):
    x[(0,j)] = model.addVar(vtype='C',
                        lb=0.0,
                        name='RDC_stock_out')
    x[(1,j)] = model.addVar(vtype='C',
                        lb=0.0,
                        name='FDC_stock_out')


for j in range(n):
    model.addCons( x[(0,j)] >= (1-alpha) * (1-theta) * x[(1,j)] - w[0] * I + D[0][j], name='RDC_out')

for j in range(n):
    model.addCons( x[(1,j)] >= D[1][j] - w[1] * I, name='FDC_out')

# Sum to 1
model.addCons( w[0] + w[1] <= 1, name='Sum1')

# Obj
model.setObjective(
	quicksum( x[(0,j)] * p for j in range(n))
    + quicksum(x[(1,j)] * p * (alpha+theta*(1-alpha)) for j in range(n) ), 'minimize')

model.optimize()

print('Optimal value:', model.getObjVal()/n)


print ('Optimal Solution: RDC_reserve: %.4f, FDC_allo: %.4f' %(model.getVal(w[0]), model.getVal(w[1])))
print ('-----------------------------------------------------------------')
# print ('  ')
for i in range(n):
    print (D[0][i], D[1][i], model.getVal(x[(0,i)]), model.getVal(x[(1,i)]))

