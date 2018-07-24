# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-07-17 18:09:08
# @Last Modified by:   chenxinma
# @Last Modified at:   2018-07-24 10:25:19


from pyscipopt import Model, quicksum, multidict
import numpy as np
import matplotlib.pyplot as plt

# 1-RDC, m-FDC, 1-Sku 


class RDC_FDC:

    def __init__(self, I, SV, n=500): 
        self.I = I
        self.m = len(I) - 1
        self.alpha = 0.1
        self.theta = 0.2
        self.p = 1
        self.M = self.p*1.
        self.D = {}
        self.SV = SV
        self.n = n
        for i in range(self.m+1):
            self.D[i] = np.clip(np.random.normal(SV[i][0], SV[i][1], n), 1, None)

    def build_1(self):

        self.model = Model('Allocation')


        self.w = {}
        for i in range(self.m+1):
            self.w[i] = self.model.addVar(vtype='C', 
                            lb=0.0,
                            ub=1.0,
                            name='DC%i_reserve_ratio' %i)

        self.x0 = {}
        self.x1 = {}
        self.s0 = {}

        for i in range(1,self.m+1):
            for j in range(self.n):
                self.x0[(i,j)] = self.model.addVar(vtype='C',
                                lb=0.0,
                                name='Fulfill_from_RDC_to_DC%i' %i)
                self.x1[(i,j)] = self.model.addVar(vtype='C',
                                lb=0.0,
                                name='Fulfill_from_FDC%i' %i)

        for j in range(self.n):
            self.s0[j] = self.model.addVar(vtype='C', lb=0.0, name='Slack')


        # RDC capacity
        for j in range(self.n):
            self.model.addCons(self.D[0][j] + (1-self.theta)*(1-self.alpha)*quicksum(self.x0[(i,j)] \
                                            for i in range(1,self.m+1))\
                            <= self.w[0]*self.I[0] + self.s0[j], name='RDC_capacity')

        # RDC stock
        for i in range(1,self.m+1):
            for j in range(self.n):
                self.model.addCons( self.x1[(i,j)] <=self.w[i]*self.I[0], name='FDC_stock')

        # Demand satisfied
        for i in range(1, self.m+1):
            for j in range(self.n):
                self.model.addCons( self.I[i]+self.x0[(i,j)]+self.x1[(i,j)] >= self.D[i][j], name='Demand')

        # Sum to 1
        self.model.addCons( quicksum(self.w[i] for i in range(self.m+1)) <= 1, name='Sum1')

        # Obj
        self.model.setObjective( quicksum(self.M*self.s0[i] for i in range(self.n)) + \
            (self.alpha+self.theta*(1-self.alpha)) * self.p * quicksum(self.x0[(i,j)]  \
                for i in range(1, self.m+1) for j in range(self.n) ), 'minimize')

        self.model.optimize()



    def build_2(self):

        self.model = Model('Allocation')


        self.w = {}
        for i in range(self.m+1):
            self.w[i] = self.model.addVar(vtype='C', 
                            lb=0.0,
                            ub=1.0,
                            name='DC%i_reserve_ratio' %i)

        self.x = {}
        for i in range(self.m+1):
            for j in range (self.n):
                self.x[(i,j)] = self.model.addVar(vtype='C',
                                lb=0.0,
                                name='DC%i_stockout' %i)



        for j in range(self.n):
            self.model.addCons( self.x[(0,j)] >= (1-self.alpha) * (1-self.theta) * quicksum(self.x[(i,j)]\
                          for i in range(1,self.m+1)) - self.w[0] * self.I[0] + self.D[0][j], name='RDC_out')

        for i in range(1, self.m+1):
            for j in range(self.n):
                self.model.addCons( self.x[(i,j)] >= self.D[i][j] - self.w[i] * self.I[0] - self.I[i],
                name='FDC%i_out' %i)

        # Sum to 1
        self.model.addCons( quicksum(self.w[i] for i in range(self.m+1)) <= 1, name='Sum1')

        # Obj
        self.model.setObjective(
            quicksum( self.x[(0,j)] * self.p for j in range(self.n))
            + quicksum( self.x[(i,j)] * self.p * (self.alpha+self.theta*(1-self.alpha))
            for i in range(1, self.m+1) for j in range(self.n)) , 'minimize')

        self.model.optimize()


    def get_sol(self):
        return [self.model.getVal(self.w[i]) for i in range(self.m+1)]

    def print_model(self, full=False):

        print('Optimal value:', self.model.getObjVal()/self.n)
        print('Optimal Solution:')
        for i in range(self.m+1):
            print ('delta_%i: %2.5f,  ' %(i, self.model.getVal(self.w[i])), end='')
        print('')

        if full==True:
            print ('-----------------------------------------------------------------')
            for i in range(self.m+1):
                print ('{0:9s}, {1:7s}, '.format('  dmnd[%i]' %i, 'dire[%i]' %i), end='')
            print('')
            print ('----------------------------------------------------------------------------------')
            for j in range(self.n):
                print('{0:9.4f}, {1:7.4f}, '.format(self.D[0][j], self.model.getVal(self.w[0])*self.I[0]), end='')
                for i in range(1, self.m+1):
                    print('{0:9.4f}, {1:7.4f}, '.format(self.D[i][j], \
                        self.model.getVal(self.x0[(i,j)])*(1-self.alpha)), end='')
                print('')




def exp1(I, SV, n):

    exp_I0 = list(range(100,1000,50))
    w0, w1 = [], []
    for e in exp_I0:
        np.random.seed(0)
        I[0] = e
        m = RDC_FDC(I, SV, n)
        m.build_2()
        m.print_model()
        sol = m.get_sol()
        w0.append(sol[0])
        w1.append(1 - sum([i for i in sol[1:]] ))

    plt.plot(exp_I0, w0, label='w[0]')
    plt.plot(exp_I0, w1, label='1-w[1:]')
    plt.xlabel('RDC Inventory')
    plt.ylabel('RDC 保有率')
    plt.title('RDC保有率随自身库存下降')
    plt.legend()
    plt.grid()
    plt.savefig('fig1_0/RDC_I.png', dpi=150)
    plt.close()
    # plt.show()




def exp2(I, SV, n):

    exp_I0 = list(range(0,100,10))
    w0 = []
    for e in exp_I0:
        # np.random.seed(1)
        SV[0][1] = e
        m = RDC_FDC(I, SV, n)
        m.build_2()
        m.print_model()
        sol = m.get_sol()
        w0.append(sol[0] )

    plt.plot(exp_I0, w0)
    plt.xlabel('RDC Demand Variance')
    plt.ylabel('RDC 保有率')
    plt.title('RDC保有率随RDC需求方差下降')
    plt.grid()
    plt.savefig('fig1_0/RDC_D_var.png', dpi=150)
    plt.close()
    # plt.show()


def exp3(I, SV, n):


    exp_I0 = list(range(0,200,20))
    w0 = []
    for e in exp_I0:
        np.random.seed(2)
        SV[1][0] = e
        m = RDC_FDC(I, SV, n)
        m.build_2()
        m.print_model()
        sol = m.get_sol()
        w0.append(sol[0])

    plt.plot(exp_I0, w0)
    plt.xlabel('FDC1 Demand Mean')
    plt.ylabel('RDC 保有率')
    plt.title('RDC保有率随FDC需求下降')
    plt.grid()
    plt.savefig('fig1_0/FDC_D_mean.png', dpi=150)
    plt.close()
    # plt.show()



def exp4(I, SV, n):
    I = {0: 100, 1:0, 2:10, 3:2}  # Inventory
    SV = {0: [120, 30],
          1: [40, 10],
          2: [40, 10],
          3: [40, 10]
          } 
    n = 500

    exp_I0 = list(range(0,22,2))
    w0 = []
    for e in exp_I0:
        np.random.seed(1)
        SV[1][1] = e
        m = RDC_FDC(I, SV, n)
        m.build_2()
        m.print_model()
        sol = m.get_sol()
        w0.append(sol[0])

    plt.plot(exp_I0, w0)
    plt.xlabel('FDC1 Demand Var')
    plt.ylabel('RDC 保有率')
    plt.title('RDC保有率随FDC需求方差上升')
    plt.grid()
    plt.savefig('fig1_0/FDC_D_var.png', dpi=150)
    plt.close()
    # plt.show()


def exp5(I, SV, n):

    exp_I0 = list(range(0,300,15))
    w0 = []
    for e in exp_I0:
        np.random.seed(1)
        SV[0][0] = e
        m = RDC_FDC(I, SV, n)
        m.build_1()
        m.print_model()
        sol = m.get_sol()
        w0.append(sol[0])

    plt.plot(exp_I0, w0)
    plt.xlabel('RDC Demand Mean')
    plt.ylabel('RDC 保有率')
    plt.title('RDC保有率随RDC需求上升')
    plt.grid()
    plt.savefig('fig1_0/RDC_D_mean.png', dpi=150)
    plt.close()
    # plt.show()



def exp0(I, SV, n):

    exp_I0 = list(range(50,1000,50))
    d_mean_sum = sum([SV[i][0] for i in SV])
    ratio_I_d = [exp_I0[i]/d_mean_sum for i in range(len(exp_I0))]
    w0, w1, w2, w3 = [], [], [], []
    for e in exp_I0:
        np.random.seed(0)
        I[0] = e
        m = RDC_FDC(I, SV, n)
        m.build_2()
        m.print_model()
        sol = m.get_sol()
        w0.append(1- sum([i for i in sol[1:]]))
        w1.append(e * sum([i for i in sol[1:]] ))
        w2.append(e * sum([i for i in sol] ))
        w3.append(e)

    fig, ax1 = plt.subplots()
    ax1.plot(ratio_I_d, w0, color='r', label='RDC_reserve_ratio')    
    ax1.set_xlabel('库存/需求均值总和')
    ax1.set_ylabel('RDC_reserve_ratio')
    ax1.set_ylim((0,1))
    ax1.set_xlim((0,ratio_I_d[-1]))

    ax2 = ax1.twinx()    
    ax2.fill_between(ratio_I_d, 0, w1, label='FDC_Alloc', alpha=0.8)
    ax2.fill_between(ratio_I_d, w1, w2, label='RDC_Alloc', alpha=0.8)
    ax2.fill_between(ratio_I_d, w2, w3, label='Buffer', alpha=0.8)
    ax2.legend(loc=6)
    ax2.set_ylabel('Quantity')
    ax2.set_ylim(ymin=0)
    plt.xlabel('RDC Inventory')
    plt.title('库存变化如何影响分配数量')
    # plt.grid()
    plt.savefig('fig2_0/Q_I_Fvar_%i.png' %SV[1][1])
    plt.close()
    # plt.show()


if __name__ == "__main__":
    I = {0: 200, 1:0, 2:10, 3:2}  # Inventory
    SV = {0: [120, 30],
          1: [40, 10],
          2: [40, 10],
          3: [40, 10]
          } 
    n = 500

    # np.random.seed(0)
    # exp0(I, SV, n)
    # exp1(I, SV, n)
    # exp2(I, SV, n)  
    # exp3(I, SV, n)
    # exp4(I, SV, n)
    # exp5(I, SV, n)

    var = [0,5,10,15,20]

    for v in var:
        SV[1][1] = v
        SV[2][1] = v
        SV[3][1] = v
        exp0(I, SV, n)
       