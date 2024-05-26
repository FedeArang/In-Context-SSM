import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal
from scipy import linalg as la
from scipy import special as ss

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
    def __init__(self, N, dt=1.0, discretization='bilinear', trainable=False, teacher_ratio: float = 1.0, device: str = "cpu", full: bool = False):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        """
        super().__init__()
        self.N = N
        self.dt = dt
        self.teacher_ratio = teacher_ratio
        self.device = device
        self.full = full
        self.trainable = trainable
        A, B = transition('legt', N)

        if trainable and not full:
            # Q = np.arange(N, dtype=np.float64)
            # evals = (2*Q + 1)** .5  #The Lengendre Polinomyals satisfy P_n(1) = 1 for all n
            # evals = np.reshape(evals, (self.N, ))

            # C = np.dot(evals, A)
            # D = np.sum(evals*B.squeeze(-1)).reshape(1,)
            # CC = torch.Tensor(np.reshape(C, (N, )))
            # DD = torch.Tensor(np.reshape(D, (1,)))
            # C_discr = 1/(1-0.5*DD*self.dt)*0.5*dt*CC
            # D_discr = 1/(1-0.5*DD*self.dt)*(1+0.5*dt*DD)
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

        if full:
            self.A = torch.nn.Parameter(torch.Tensor(A).requires_grad_())
            self.B = torch.nn.Parameter(torch.Tensor(B).requires_grad_())
        else:
            self.register_buffer('A', torch.Tensor(A)) # (N, N)
            self.register_buffer('B', torch.Tensor(B)) # (N,)
        

        # vals = np.linspace(0.0, 1.0, 1./dt)
        vals = np.arange(0.0, 1.0, dt)
        self.eval_matrix = torch.Tensor(ss.eval_legendre(np.arange(N)[:, None], 1 - 2 * vals).T)

    def generate_C_D(self, A, B):
        Q = np.arange(self.N, dtype=np.float32)
        evals = (2*Q + 1)** .5  # The Lengendre Polinomyals satisfy P_n(1) = 1 for all n
        evals = torch.tensor(np.reshape(evals, (self.N, )))

        C = evals @ A
        D = torch.sum(evals*B.squeeze(-1)).reshape(1,)

        CC = torch.Tensor(torch.reshape(C, (self.N, )))
        DD = torch.Tensor(torch.reshape(D, (1,)))

        self.C_discr = 1/(1-0.5*DD*self.dt)*0.5*self.dt*CC
        self.D_discr = 1/(1-0.5*DD*self.dt)*(1+0.5*self.dt*DD)

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
        if self.full:
            self.generate_C_D(self.A, self.B)
        
        #next_step_pred = []
        next_step_pred = torch.zeros_like(inputs)

        steps_teacherforcing = range(int(inputs.shape[1] * self.teacher_ratio))
        steps_autoregressive = range(len(steps_teacherforcing),inputs.shape[1], 1)
        for i in steps_teacherforcing:
            if i==0:
                c=self.B * inputs[:,i,:]
            else:
                c=(c @ self.A + self.B * inputs[:,i,:])
            if i==0:
                pred = (c @ self.C_discr).reshape(-1,1) + self.D_discr * inputs[:,i,:]
            else:
                pred = (2 * c @ self.C_discr).reshape(-1,1) + self.D_discr * inputs[:,i,:]
            cs.append(c)
            next_step_pred[:,i,:]=pred

        for i in steps_autoregressive:
            c=F.linear(c, self.A) + self.B * next_step_pred[:,i-1,:]
            pred = (2*c@ self.C_discr).reshape(-1,1) + self.D_discr * next_step_pred[:,i-1,:]
            next_step_pred[:,i,:]=pred
            cs.append(c)
        
        return next_step_pred.view(inputs.shape[0],inputs.shape[1])
    

    def reconstruct(self, c):
        return (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)





'''class HiPPO_LegS(nn.Module):
    """ Vanilla HiPPO-LegS model (scale invariant instead of time invariant) """
    def __init__(self, N, max_length=1024, measure='legs', discretization='bilinear'):
        """
        max_length: maximum sequence length
        """
        super().__init__()
        self.N = N
        A, B = transition(measure, N)
        B = B.squeeze(-1)
        A_stacked = np.empty((max_length, N, N), dtype=A.dtype)
        B_stacked = np.empty((max_length, N), dtype=B.dtype)
        for t in range(1, max_length + 1):
            At = A / t
            Bt = B / t
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
        # print("B_stacked shape", B_stacked.shape)

        vals = np.linspace(0.0, 1.0, max_length)
        self.eval_matrix = torch.Tensor((B[:, None] * ss.eval_legendre(np.arange(N)[:, None], 2 * vals - 1)).T)

    def forward(self, inputs, fast=False):
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
        return result

    def reconstruct(self, c):
        a = self.eval_matrix @ c.unsqueeze(-1)
        return a.squeeze(-1)'''