import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal
from scipy import linalg as la
from scipy import special as ss
import math

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
sys.path.append('C:/Users/faran/Documents_nuova/Documenti/Federico/ETH/in context learning/Git_Hub/In-Context-SSM/model/')
from model.op import transition


"""
The HiPPO_LegT and HiPPO_LegS modules satisfy the HiPPO interface:

The forward() method takes an input sequence f of length L to an output sequence c of shape (L, N) where N is the order of the HiPPO operator.
c[k] can be thought of as representing all of f[:k] via coefficients of a polynomial approximation.

The reconstruct() method takes the coefficients and turns each coefficient into a reconstruction of the original input.
Note that each coefficient c[k] turns into an approximation of the entire input f, so this reconstruction has shape (L, L),
and the last element of this reconstruction (which has shape (L,)) is the most accurate reconstruction of the original input.

Both of these two methods construct approximations according to different measures, defined in the HiPPO paper.
The first one is the "Translated Legendre" (which is up to scaling equal to the LMU matrix),
and the second one is the "Scaled Legendre".
Each method comprises an exact recurrence c_k = A_k c_{k-1} + B_k f_k, and an exact reconstruction formula based on the corresponding polynomial family.
"""

class HiPPO_LegT(nn.Module):
    def __init__(self, N, dt=1.0, discretization='bilinear', trainable=False, teacher_ratio: float = 1.0, device: str = "cpu"):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        """
        super().__init__()
        self.N = N
        self.dt = dt
        self.teacher_ratio = teacher_ratio
        A, B = transition('legt', N)

        if trainable:
            C = np.ones((N,))
            D = np.zeros((1,))
            self.C_discr = torch.nn.Parameter(torch.Tensor(C).requires_grad_())
            self.D_discr = torch.nn.Parameter(torch.Tensor(D).requires_grad_())

        else:
            Q = np.arange(N, dtype=np.float64)
            evals = (2*Q + 1)** .5  #The Lengendre Polinomyals satisfy P_n(1) = 1 for all n
            evals = np.reshape(evals, (self.N, ))

            C = np.dot(evals, A)
            D = np.sum(evals*B.squeeze(-1)).reshape(1,)

            self.C = torch.Tensor(np.reshape(C, (N, )))
            self.D = torch.Tensor(np.reshape(D, (1,)))

            self.C_discr = 1/(1-0.5*self.D*self.dt)*0.5*dt*self.C
            self.D_discr = 1/(1-0.5*self.D*self.dt)*(1+0.5*dt*self.D)


        # dt, discretization options
        A, B, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)

        B = B.squeeze(-1)
    
        self.register_buffer('A', torch.Tensor(A)) # (N, N)
        self.register_buffer('B', torch.Tensor(B)) # (N,)

        # vals = np.linspace(0.0, 1.0, 1./dt)
        vals = np.arange(0.0, 1.0, dt)
        self.eval_matrix = torch.Tensor(ss.eval_legendre(np.arange(N)[:, None], 1 - 2 * vals).T)

    def forward(self, inputs):
        """
        inputs : (batch size, length)
        output : (batch size, length) where N is the order of the HiPPO projection
        """
        if len(inputs.shape) == 1:
            #then batch size is one and we unsqueeze
            inputs = inputs.unsqueeze(0)
            
        inputs = inputs.unsqueeze(-1)
        u = inputs * self.B # (length, ..., N)

        c = torch.zeros((u.shape[0], u.shape[-1])).to(torch.float32)

        cs = []
        next_step_pred = torch.zeros_like(inputs)
    
        steps_teacherforcing = range(int(inputs.shape[1] * self.teacher_ratio))
        steps_autoregressive = range(len(steps_teacherforcing),inputs.shape[1], 1)
        for i in steps_teacherforcing:
            c = F.linear(c, self.A) + self.B * inputs[:,i,:]
            if len(cs)==0:
                pred = (c @ self.C_discr).reshape(-1,1) + self.D_discr * inputs[:,i,:]
            else:
                pred = (2*c @ self.C_discr).reshape(-1,1) + self.D_discr * inputs[:,i,:]
            cs.append(c)
            next_step_pred[:,i,:]=pred

        for i in steps_autoregressive:
            c = F.linear(c, self.A) + self.B * next_step_pred[:,i-1,:]
            pred = (2*c @ self.C_discr).reshape(-1,1) + self.D_discr * next_step_pred[:,i-1,:]
            cs.append(c)
            next_step_pred[:,i,:]=pred
        
        return next_step_pred.view(inputs.shape[0],inputs.shape[1])
    

    def reconstruct(self, c):
        return (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)



class HiPPO_LegS(nn.Module):
    """ Vanilla HiPPO-LegS model (scale invariant instead of time invariant) """
    def __init__(self, N, max_length=1024, measure='legs', discretization='bilinear', dt=1.0):
        """
        max_length: maximum sequence length
        """
        super().__init__()
        self.N = N
        self.dt = 1
        A, B = transition(measure, N)
        B = B.squeeze(-1)
        A_stacked = np.empty((max_length, N, N), dtype=A.dtype)
        B_stacked = np.empty((max_length, N), dtype=B.dtype)
        C_stacked = np.empty((max_length, N))
        D_stacked = np.empty((max_length, 1))
        C_discr_stacked = np.empty((max_length, N))
        D_discr_stacked = np.empty((max_length, 1))

        Q = np.arange(N, dtype=np.float64)
        evals = (2*Q + 1)** .5  #The Lengendre Polinomyals satisfy P_n(1) = 1 for all n
        evals = np.reshape(evals, (self.N, ))

        for t in range(1, max_length + 1):
            At = A / t
            Bt = B / t

            C_stacked[t-1] = np.dot(evals, At)
            D_stacked[t-1] = np.sum(evals*Bt)
            C_discr_stacked[t-1] = 1/(1-0.5*D_stacked[t-1]*self.dt)*0.5*self.dt*C_stacked[t-1]
            D_discr_stacked[t-1] = 1/(1-0.5*D_stacked[t-1]*self.dt)*(1+0.5*self.dt*D_stacked[t-1])


            if discretization == 'forward':
                A_stacked[t - 1] = np.eye(N) + At
                B_stacked[t - 1] = Bt
            elif discretization == 'backward':
                A_stacked[t - 1] = la.solve_triangular(np.eye(N) - At, np.eye(N), lower=True)
                B_stacked[t - 1] = la.solve_triangular(np.eye(N) - At, Bt, lower=True)
            elif discretization == 'bilinear':
                A_stacked[t - 1] = la.solve_triangular(np.eye(N) - At / 2, np.eye(N) + At / 2, lower=True)
                B_stacked[t - 1] = la.solve_triangular(np.eye(N) - At / 2, Bt, lower=True)
            else: # ZOH
                A_stacked[t - 1] = la.expm(A * (math.log(t + 1) - math.log(t)))
                B_stacked[t - 1] = la.solve_triangular(A, A_stacked[t - 1] @ B - B, lower=True)
        
        self.A_stacked = torch.Tensor(A_stacked) # (max_length, N, N)
        self.B_stacked = torch.Tensor(B_stacked) # (max_length, N)
        self.C_stacked = torch.Tensor(C_stacked) # (max_length, N, N)
        self.D_stacked = torch.Tensor(D_stacked)
        self.C_discr_stacked = torch.Tensor(C_discr_stacked) 
        self.D_discr_stacked = torch.Tensor(D_discr_stacked)
        # print("B_stacked shape", B_stacked.shape)

        vals = np.linspace(0.0, 1.0, max_length)
        self.eval_matrix = torch.Tensor((B[:, None] * ss.eval_legendre(np.arange(N)[:, None], 2 * vals - 1)).T)

    
    def forward(self, inputs):
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """

        inputs = inputs.unsqueeze(-1)
        u = inputs * self.B_stacked[0] # (length, ..., N)

        c = torch.zeros(u.shape[1:])
        cs = []
        next_step_pred = []
    
        for i in range(len(inputs)):
            c = F.linear(c, self.A_stacked[i]) + self.B_stacked[i] * inputs[i]

            if len(cs)==0:
                pred = torch.dot(c, self.C_discr_stacked[i])+self.D_discr_stacked[i]*inputs[i]
            else:
                #pred = 1/(1-0.5*self.D*self.dt)*((1+0.5*self.D*self.dt)*f+0.5*self.dt*torch.dot((c+torch.Tensor(cs[-1])), self.C))
                pred = torch.dot(2*c, self.C_discr_stacked[i])+self.D_discr_stacked[i]*inputs[i]
            cs.append(c)
            next_step_pred.append(pred)
        
        return torch.stack(next_step_pred, dim=0)
        #return torch.stack(cs, dim=0)

    
    '''def forward(self, inputs, fast=False):
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """

        L = inputs.shape[0]

        inputs = inputs.unsqueeze(-1)
        u = torch.transpose(inputs, 0, -2)
        u = u * self.B_stacked[:L]
        u = torch.transpose(u, 0, -2) # (length, ..., N)

        if fast:
            result = unroll.variable_unroll_matrix(self.A_stacked[:L], u)
        else:
            result = unroll.variable_unroll_matrix_sequential(self.A_stacked[:L], u)
        return result'''

    def reconstruct(self, c):
        a = self.eval_matrix @ c.unsqueeze(-1)
        return a.squeeze(-1)
    

class HiPPO_FouT(nn.Module):
    def __init__(self, N, dt=1.0, discretization='bilinear', trainable=False, teacher_ratio: float = 1.0, device: str = "cpu"):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        """
        super().__init__()
        if N%2==0:
            raise ValueError('N must be odd')
        self.N = N
        self.dt = dt
        self.teacher_ratio = teacher_ratio
        A, B = transition('fout', N)

        if trainable:
            C = np.ones((N,))
            D = np.zeros((1,))
            self.C_discr = torch.nn.Parameter(torch.Tensor(C).requires_grad_())
            self.D_discr = torch.nn.Parameter(torch.Tensor(D).requires_grad_())

        else:
            C=np.zeros(N)
            for i in range(int((N-1)/2)):
                C[2*i+2]=-2*np.pi*(i+1)
            
            D=np.ones((1,))

            self.C_discr=torch.Tensor(dt*C)
            self.D_discr=torch.Tensor(D)    
            
        # dt, discretization options
        A, B, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)
    
        self.register_buffer('A', torch.Tensor(A)) # (N, N)
        self.register_buffer('B', torch.Tensor(B)) # (N,)

        # vals = np.linspace(0.0, 1.0, 1./dt)
        vals = np.arange(0.0, 1.0, dt)
        self.eval_matrix = torch.Tensor(ss.eval_legendre(np.arange(N)[:, None], 1 - 2 * vals).T)

    '''def forward(self, inputs):
        """
        inputs : (batch size, length)
        output : (batch size, length) where N is the order of the HiPPO projection
        """
        if len(inputs.shape) == 1:
            #then batch size is one and we unsqueeze
            inputs = inputs.unsqueeze(0)
            
        inputs = inputs.unsqueeze(-1)
        u = inputs * self.B # (length, ..., N)

        c = torch.zeros((u.shape[0], u.shape[-1])).to(torch.float32)

        cs = []
        next_step_pred = torch.zeros_like(inputs)
    
        steps_teacherforcing = range(int(inputs.shape[1] * self.teacher_ratio))
        steps_autoregressive = range(len(steps_teacherforcing),inputs.shape[1], 1)
        for i in steps_teacherforcing:
            c = F.linear(c, self.A) + self.B * inputs[:,i,:]
            if len(cs)==0:
                pred = (c @ self.C_discr).reshape(-1,1) + self.D_discr * inputs[:,i,:]
            else:
                pred = (2*c @ self.C_discr).reshape(-1,1) + self.D_discr * inputs[:,i,:]
            cs.append(c)
            next_step_pred[:,i,:]=pred

        for i in steps_autoregressive:
            c = F.linear(c, self.A) + self.B * next_step_pred[:,i-1,:]
            pred = (2*c @ self.C_discr).reshape(-1,1) + self.D_discr * next_step_pred[:,i-1,:]
            cs.append(c)
            next_step_pred[:,i,:]=pred
        
        return next_step_pred.view(inputs.shape[0],inputs.shape[1])'''
    
    def forward(self, inputs):
            
        inputs = inputs.unsqueeze(-1)
        u = inputs * self.B # (length, ..., N)

        c = torch.zeros(u.shape[-1]).to(torch.float32)

        cs = []
        next_step_pred = torch.zeros_like(inputs)
    
        steps_teacherforcing = range(int(inputs.shape[0] * self.teacher_ratio))
        
        for i in steps_teacherforcing:
            c = F.linear(c, self.A) + self.B * inputs[i]
            if len(cs)==0:
                pred = torch.dot(c, self.C_discr) + self.D_discr * inputs[i]
            else:
                pred = torch.dot(c, self.C_discr) + self.D_discr * inputs[i]
            cs.append(c)
            next_step_pred[i]=pred

        return next_step_pred

    

    def reconstruct(self, c):
        return (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)


