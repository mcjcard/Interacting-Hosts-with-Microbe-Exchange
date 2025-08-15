import networkx as nx
from scipy.integrate import solve_ivp
from scipy.optimize import fixed_point
from scipy.integrate import RK45
from scipy import spatial
import numpy.random as rand
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import pandas as pd
import seaborn as sns
import pickle as pkl
from joblib import Parallel, delayed
from einsumt import einsumt as einsum
import time
import gc
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

class NetDiff:
    def __init__(self, G: nx.Graph, fun: callable) -> None:
        self.G = G
        self.fun = fun
        self.interactions = nx.get_edge_attributes(self.G, 'interaction')
        if len(self.interactions) == 0:
            if len(nx.get_edge_attributes(self.G,'weight'))==0:
                nx.set_edge_attributes(self.G, 1, 'interaction')
            else:
                nx.set_edge_attributes(self.G, nx.get_edge_attributes(self.G,'weight'),'interaction')
            self.interactions = nx.get_edge_attributes(self.G, 'interaction')
    
    def get_interaction_times(self, tspan: list , num_interactions = 0, fixed=None) -> list:
        
        if fixed is not None:
            return fixed
        
        keys = []
        interaction_strength = []

        for x in self.interactions:
            keys.append(x)
            interaction_strength.append(self.interactions[x])

        interaction_sum = self.G.size('interaction')
        interaction_strength = np.array(interaction_strength) / interaction_sum
        t = tspan[0]
        interaction_list = []
        
        if num_interactions < 1:
            while t <= tspan[1]:
                eps = rand.default_rng().exponential(1 / interaction_sum)
                t += eps
                interaction_list.append(t)
        
            interaction_list = interaction_list[:-1]
        
        else:
            for i in range(num_interactions):
                eps = rand.default_rng().exponential(1 / interaction_sum)
                t += eps
                interaction_list.append(t)
            
            tspan[1] = interaction_list[-1] * (num_interactions + 1) / num_interactions

        edge_index = rand.default_rng().choice(np.arange(0,len(keys),dtype=int), len(interaction_list), p = interaction_strength)
        edge_list = [keys[i] for i in edge_index]
        
        return [interaction_list, edge_list, tspan]
    
    def solve(self, tspan: list, y0_list: list, gamma = 0.1, num_interactions = 0, args_list = None, steps = 10_000, fixed = None):
        
        m = self.G.order()
        
        if args_list == None:
            args_list = [None] * m
        
        t_sol = np.linspace(tspan[0], tspan[1], steps+1)
        tl = steps+1
        
        [interaction_list, edge_list, temp] = self.get_interaction_times(tspan, 0, fixed=fixed)
        t_ind_current = [0] * m
        t_current = [t_sol[0]] * m

        state_current = y0_list
            
        t_sol = np.linspace(tspan[0], tspan[1], steps+1)
        y_sol = np.empty((m,len(y0_list[0]),len(t_sol)))
        y_sol[:,:,0] = np.array(y0_list)

        step = t_sol[1]-t_sol[0]

        for i in range(len(interaction_list)):
            a = edge_list[i][0]
            b = edge_list[i][1]
            ta = t_ind_current[a]
            tb = t_ind_current[b]
            t = interaction_list[i]
            tf = int(np.floor((t-tspan[0])/step))

            if t>t_sol[tf]:
            
                sola = solve_ivp(self.fun, [t_current[a],t+1e-10], state_current[a],
                                t_eval=np.append(t_sol[ta+1:tf+1],t), args=args_list[a],rtol=1e-6,atol=1e-8)
                
                
                solb = solve_ivp(self.fun, [t_current[b],t+1e-10], state_current[b],
                                t_eval=np.append(t_sol[tb+1:tf+1],t), args=args_list[b],rtol=1e-6,atol=1e-8)
                
                y_sol[a,:,ta+1:tf+1] = sola.y[:,:-1]
                y_sol[b,:,tb+1:tf+1] = solb.y[:,:-1]
            
            elif t==t_sol[tf]:
            
                sola = solve_ivp(self.fun, [t_current[a],t+1e-10], state_current[a],
                                t_eval=t_sol[ta+1:tf+1], args=args_list[a],rtol=1e-6,atol=1e-8)
                
                
                solb = solve_ivp(self.fun, [t_current[b],t+1e-10], state_current[b],
                                t_eval=t_sol[tb+1:tf+1], args=args_list[b],rtol=1e-6,atol=1e-8)
                
                y_sol[a,:,ta+1:tf+1] = sola.y
                y_sol[b,:,tb+1:tf+1] = solb.y
            
            else:
                raise Exception("t < t_sol[tf]")

            state_current[a] = (1-gamma)*sola.y[:,-1] + gamma*solb.y[:,-1]
            state_current[b] = (1-gamma)*solb.y[:,-1] + gamma*sola.y[:,-1]

            t_ind_current[a] = tf
            t_ind_current[b] = tf

            t_current[a] = t
            t_current[b] = t
        
        for i in range(m):
            ti = t_ind_current[i]
            tf = len(t_sol)-1
            
            soli = solve_ivp(self.fun,[t_current[i], t_sol[tf]+1e-10], state_current[i],
                                 t_eval=t_sol[ti+1:], args=args_list[i],rtol=1e-6,atol=1e-8)
            y_sol[i,:,ti+1:] = soli.y
        
        return [t_sol,y_sol]
    
    def find_basin(self,points,basins,basins_range):
        m = points.shape[0]
        n = len(np.unique(basins))
        n2 = points.shape[1]
        tl = points.shape[2]

        basin_starts = np.empty(n2)
        dx = np.empty(n2)

        for i in range(n2):
            basin_starts[i] = basins_range[i][0]
            dx[i] = (basins.shape[i]-1)/(basins_range[i][1]-basins_range[i][0])
    
        points = points - np.transpose(np.full((m,tl,n2),basin_starts),axes=(0,2,1))
        points = np.transpose(einsum('abc,b->abc',points,dx).astype(int),axes=(1,0,2))

        out = np.zeros((m,n,tl))
        bs = (basins[tuple(points)]-1).astype(int)
        [t_mat,h_mat] = np.meshgrid(np.arange(tl),np.arange(m))

        out[h_mat,bs,t_mat] = 1.
    
        return out

    def solve_hils(self, tspan: list, y0_list: np.array, gamma = 0.1, steps=1000, args_list = None):
        
        m = self.G.order()

        t_sol = np.linspace(tspan[0], tspan[1], steps+1)
        tl = steps+1
        
        if args_list == None:
            args_list = [None] * m
            
        n = len(y0_list[0])

        lam = nx.to_numpy_array(self.G)
        lap = gamma*(lam - np.diag(lam.sum(axis=0)))

        step = t_sol[1]-t_sol[0]
        
        if args_list == None:
            args_list = [None] * m
        
        def fun2(t,ys_flat):
            ys_temp = ys_flat.reshape((n,m))
            dys = np.array(self.fun(t,ys_temp,*args_list[0]))
            dys_exch = ys_temp@lap
            return (dys+dys_exch).flatten()

        sol = solve_ivp(fun2, tspan, np.transpose(np.array(y0_list)).flatten(),
                            t_eval=t_sol,rtol=1e-7,atol=1e-7)
            
        y_sol = np.transpose(sol.y.reshape(n,m,tl),[1,0,2])
        
        return [t_sol,y_sol]
    
    def solve_hils_basins(self, tspan: list, states0: np.array, crits, basins, basins_range, gamma = 0.1, steps=1000, args_list = None):
        
        m = self.G.order()
        n2 = len(np.unique(basins))

        t_sol = np.linspace(tspan[0], tspan[1], steps+1)
        tl = steps+1
        
        if args_list == None:
            args_list = [None] * m
    
            
        inds = np.arange(n2)
        y0_list = []
        for i in range(m):
            y0_list.append(crits[rand.default_rng().choice(inds,p=states0[i])])

        n = len(y0_list[0])

        lam = nx.to_numpy_array(self.G)
        lap = gamma*(lam - np.diag(lam.sum(axis=0)))

        step = t_sol[1]-t_sol[0]
        
        if args_list == None:
            args_list = [None] * m
        
        def fun2(t,ys_flat):
            ys_temp = ys_flat.reshape((n,m))
            dys = np.array(self.fun(t,ys_temp,*args_list[0]))
            dys_exch = ys_temp@lap
            return (dys+dys_exch).flatten()

        t_sol = np.arange(tspan[0], tspan[1]+step, step)
        tl = len(np.arange(tspan[0], tspan[1]+step, step))

        sol = solve_ivp(fun2, tspan, np.transpose(np.array(y0_list)).flatten(),
                            t_eval=t_sol,rtol=1e-7,atol=1e-7)
        
            
        y_sol = np.transpose(sol.y.reshape(n,m,tl),[1,0,2])

        basins_data = self.find_basin(y_sol,basins,basins_range)
        
        return basins_data
    
    def p_solve_hils(self, tspan: list, states0: np.array, crits, basins, basins_range, gamma = 0.1, steps=1000, args_list = None, trials = 100,n_jobs=48):

        t_sol = np.linspace(tspan[0], tspan[1], steps+1)
        step = t_sol[1]-t_sol[0]
        y_sol = np.array(Parallel(n_jobs=n_jobs,backend='loky')(delayed(self.solve_hils_basins)(tspan, states0, crits, basins, basins_range,gamma, 
                                                                                                steps, args_list) for i in range(trials)))
        avg = np.mean(y_sol,axis=0)
        return [t_sol,avg]
    
    def experiment(self, tspan: list, states0: np.array, crits, basins, basins_range, trials = 100, gamma = 0.1, digits = 5, args_list = None):
        
        m = self.G.order()
        n = len(np.unique(basins))

        step = 10**(-digits)
        
        if args_list == None:
            args_list = [None] * m
        
        t_sol = np.arange(tspan[0], tspan[1]+step, step)
        tl = len(np.arange(tspan[0], tspan[1]+step, step))

        basins_data = np.zeros((trials,m,n,tl)) 

        for j in range(trials):
            [interaction_list, edge_list, temp] = self.get_interaction_times(tspan, digits, 0)
            t_current = [0] * m

            
            inds = np.arange(n)
            y0 = []
            for i in range(m):
                y0.append(crits[np.random.choice(inds,p=states0[i])])
            
            t_sol = np.arange(tspan[0], tspan[1]+step, step)
            y_sol = np.empty((m,len(y0[0]),len(t_sol)))
            y_sol[:,:,0] = np.array(y0)


            for i in range(len(interaction_list)):
                a = edge_list[i][0]
                b = edge_list[i][1]
                ta = t_current[a]
                tb = t_current[b]
                t = interaction_list[i]
                tf = round((t-tspan[0])/step)
            
                sola = solve_ivp(self.fun, [t_sol[ta],t+step], y_sol[a,:,ta],
                                 t_eval=t_sol[ta+1:tf+1], args=args_list[a])
            
            
                solb = solve_ivp(self.fun, [t_sol[tb],t+step], y_sol[b,:,tb],
                                t_eval=t_sol[tb+1:tf+1], args=args_list[b])
            
                y_sol[a,:,ta+1:tf+1] = sola.y
                y_sol[b,:,tb+1:tf+1] = solb.y

                y_sol[a,:,tf] = (1-gamma)*sola.y[:,-1] + gamma*solb.y[:,-1]
                y_sol[b,:,tf] = (1-gamma)*solb.y[:,-1] + gamma*sola.y[:,-1]

                t_current[a] = tf
                t_current[b] = tf
        
            for i in range(m):
                ti = t_current[i]
                tf = len(t_sol)-1
            
                soli = solve_ivp(self.fun,[t_sol[ti], t_sol[tf]], y_sol[i,:,ti],
                                 t_eval=t_sol[ti+1:], args=args_list[i])
                y_sol[i,:,ti+1:] = soli.y

            basins_data[j] = self.find_basin(y_sol,basins,basins_range)
        
        avg = np.mean(basins_data,axis=0)
        
        return [t_sol,avg]
    
    def s_experiment(self, tspan: list, states0: np.array, crits, basins, basins_range, gamma = 0.1, steps=10_000, args_list = None, ret_all = False):
        
        m = self.G.order()
        n = len(crits)
        
        if args_list == None:
            args_list = [None] * m
        
        t_sol = np.linspace(tspan[0], tspan[1], steps+1)
        tl = steps+1
        
        [interaction_list, edge_list, temp] = self.get_interaction_times(tspan, 0)
        t_ind_current = [0] * m
        t_current = [t_sol[0]] * m

            
        inds = np.arange(n)
        y0 = []
        for i in range(m):
            y0.append(crits[rand.default_rng().choice(inds,p=states0[i])])

        state_current = y0
            
        t_sol = np.linspace(tspan[0], tspan[1], steps+1)
        y_sol = np.empty((m,len(y0[0]),len(t_sol)))
        y_sol[:,:,0] = np.array(y0)

        step = t_sol[1]-t_sol[0]

        for i in range(len(interaction_list)):
            a = edge_list[i][0]
            b = edge_list[i][1]
            ta = t_ind_current[a]
            tb = t_ind_current[b]
            t = interaction_list[i]
            tf = int(np.floor((t-tspan[0])/step))

            if t>t_sol[tf]:
            
                sola = solve_ivp(self.fun, [t_current[a],t+1e-10], state_current[a],
                                t_eval=np.append(t_sol[ta+1:tf+1],t), args=args_list[a],rtol=1e-6,atol=1e-8)
                
                
                solb = solve_ivp(self.fun, [t_current[b],t+1e-10], state_current[b],
                                t_eval=np.append(t_sol[tb+1:tf+1],t), args=args_list[b],rtol=1e-6,atol=1e-8)
                
                y_sol[a,:,ta+1:tf+1] = sola.y[:,:-1]
                y_sol[b,:,tb+1:tf+1] = solb.y[:,:-1]
            
            elif t==t_sol[tf]:
            
                sola = solve_ivp(self.fun, [t_current[a],t+1e-10], state_current[a],
                                t_eval=t_sol[ta+1:tf+1], args=args_list[a],rtol=1e-6,atol=1e-8)
                
                
                solb = solve_ivp(self.fun, [t_current[b],t+1e-10], state_current[b],
                                t_eval=t_sol[tb+1:tf+1], args=args_list[b],rtol=1e-6,atol=1e-8)
                
                y_sol[a,:,ta+1:tf+1] = sola.y
                y_sol[b,:,tb+1:tf+1] = solb.y
            
            else:
                raise Exception("t < t_sol[tf]")

            state_current[a] = (1-gamma)*sola.y[:,-1] + gamma*solb.y[:,-1]
            state_current[b] = (1-gamma)*solb.y[:,-1] + gamma*sola.y[:,-1]

            t_ind_current[a] = tf
            t_ind_current[b] = tf

            t_current[a] = t
            t_current[b] = t
        
        for i in range(m):
            ti = t_ind_current[i]
            tf = len(t_sol)-1
            
            soli = solve_ivp(self.fun,[t_current[i], t_sol[tf]+1e-10], state_current[i],
                                 t_eval=t_sol[ti+1:], args=args_list[i],rtol=1e-6,atol=1e-8)
            y_sol[i,:,ti+1:] = soli.y

        basins_data = self.find_basin(y_sol,basins,basins_range)

        if ret_all:
            return [y_sol,basins_data]
        else:
            return basins_data
    
    def p_experiment(self, tspan: list, states0: np.array, crits, basins, basins_range, trials = 100, gamma = 0.1, steps = 10_000, args_list = None,n_jobs=48, ret_all=False):

        t_sol = np.linspace(tspan[0], tspan[1], steps)
        y_sol = Parallel(n_jobs=n_jobs,backend='loky')(delayed(self.s_experiment)(tspan=tspan, states0=states0, crits=crits, basins=basins, basins_range=basins_range, 
                                                                                            gamma=gamma, steps=steps, args_list=args_list, ret_all=ret_all) for i in range(trials))

        if ret_all:
            y_full = np.array([y[0] for y in y_sol])
            y_basins = np.array([y[1] for y in y_sol])
            avg = np.mean(y_basins,axis=0)
            return [t_sol,avg,y_full]

        else:
            avg = np.mean(y_sol,axis=0)
            return [t_sol,avg]

    def s_experiment_ys(self, tspan: list, y0_list: np.array, gamma = 0.1, steps=10_000, args_list = None):
        
        m = self.G.order()
        
        if args_list == None:
            args_list = [None] * m
        
        t_sol = np.linspace(tspan[0], tspan[1], steps+1)
        tl = steps+1
        
        [interaction_list, edge_list, temp] = self.get_interaction_times(tspan, 0)
        t_ind_current = [0] * m
        t_current = [t_sol[0]] * m

        state_current = y0_list
            
        t_sol = np.linspace(tspan[0], tspan[1], steps+1)
        y_sol = np.empty((m,len(y0_list[0]),len(t_sol)))
        y_sol[:,:,0] = np.array(y0_list)

        step = t_sol[1]-t_sol[0]

        for i in range(len(interaction_list)):
            a = edge_list[i][0]
            b = edge_list[i][1]
            ta = t_ind_current[a]
            tb = t_ind_current[b]
            t = interaction_list[i]
            tf = int(np.floor((t-tspan[0])/step))

            if t>t_sol[tf]:
            
                sola = solve_ivp(self.fun, [t_current[a],t+1e-10], state_current[a],
                                t_eval=np.append(t_sol[ta+1:tf+1],t), args=args_list[a],rtol=1e-6,atol=1e-8)
                
                
                solb = solve_ivp(self.fun, [t_current[b],t+1e-10], state_current[b],
                                t_eval=np.append(t_sol[tb+1:tf+1],t), args=args_list[b],rtol=1e-6,atol=1e-8)
                
                y_sol[a,:,ta+1:tf+1] = sola.y[:,:-1]
                y_sol[b,:,tb+1:tf+1] = solb.y[:,:-1]
            
            elif t==t_sol[tf]:
            
                sola = solve_ivp(self.fun, [t_current[a],t+1e-10], state_current[a],
                                t_eval=t_sol[ta+1:tf+1], args=args_list[a],rtol=1e-6,atol=1e-8)
                
                
                solb = solve_ivp(self.fun, [t_current[b],t+1e-10], state_current[b],
                                t_eval=t_sol[tb+1:tf+1], args=args_list[b],rtol=1e-6,atol=1e-8)
                
                y_sol[a,:,ta+1:tf+1] = sola.y
                y_sol[b,:,tb+1:tf+1] = solb.y
            
            else:
                raise Exception("t < t_sol[tf]")

            state_current[a] = (1-gamma)*sola.y[:,-1] + gamma*solb.y[:,-1]
            state_current[b] = (1-gamma)*solb.y[:,-1] + gamma*sola.y[:,-1]

            t_ind_current[a] = tf
            t_ind_current[b] = tf

            t_current[a] = t
            t_current[b] = t
        
        for i in range(m):
            ti = t_ind_current[i]
            tf = len(t_sol)-1
            
            soli = solve_ivp(self.fun,[t_current[i], t_sol[tf]+1e-10], state_current[i],
                                 t_eval=t_sol[ti+1:], args=args_list[i],rtol=1e-6,atol=1e-8)
            y_sol[i,:,ti+1:] = soli.y

        return y_sol
    
    def p_experiment_ys(self, tspan: list, y0_list: np.array, trials = 100, gamma = 0.1, steps = 10_000, args_list = None,n_jobs=48):

        t_sol = np.linspace(tspan[0], tspan[1], steps+1)
        y_sol = Parallel(n_jobs=n_jobs,backend='loky')(delayed(self.s_experiment_ys)(tspan=tspan, y0_list=y0_list,
                                                                                            gamma=gamma, steps=steps, args_list=args_list) for i in range(trials))

        return [t_sol,np.array(y_sol)]

    def s_experiment_hfcs(self, tspan: list, states0: np.array, crits, basins, basins_range, gamma = 0.1, digits = 5, args_list = None):
        
        m = self.G.order()
        n = len(np.unique(basins))

        step = 10**(-digits)
        
        if args_list == None:
            args_list = [None] * m
        
        t_sol = np.arange(tspan[0], tspan[1]+step, step)
        tl = len(np.arange(tspan[0], tspan[1]+step, step))
        
        [interaction_list, edge_list, temp] = self.get_interaction_times(tspan, digits, 0)
        t_current = [0] * m

            
        inds = np.arange(n)
        y0 = []
        for i in range(m):
            y0.append(crits[rand.default_rng().choice(inds,p=states0[i])])
            
        t_sol = np.arange(tspan[0], tspan[1]+step, step)
        y_sol = np.empty((m,len(y0[0]),len(t_sol)))
        y_sol[:,:,0] = np.array(y0)


        for i in range(len(interaction_list)):
            a = edge_list[i][0]
            b = edge_list[i][1]
            ta = t_current[a]
            tb = t_current[b]
            t = interaction_list[i]
            tf = round((t-tspan[0])/step)
            
            y_sol[a,:,ta+1:tf+1] = y_sol[a,:,ta][:, None]
            y_sol[b,:,tb+1:tf+1] = y_sol[b,:,tb][:, None]

            ya_temp = y_sol[a,:,tf]
            yb_temp = y_sol[b,:,tf]

            y_sol[a,:,tf] = (1-gamma)*ya_temp + gamma*yb_temp
            y_sol[b,:,tf] = (1-gamma)*yb_temp + gamma*ya_temp

            t_current[a] = tf
            t_current[b] = tf
        
        for i in range(m):
            ti = t_current[i]
            tf = len(t_sol)-1
            y_sol[i,:,ti+1:] = y_sol[i,:,ti][:, None]

        basins_data = self.find_basin(y_sol[:,:,::10],basins,basins_range)
        
        return basins_data
    
    def p_experiment_hfcs(self, tspan: list, states0: np.array, crits, basins, basins_range, trials = 100, gamma = 0.1, digits = 5, args_list = None,n_jobs=48):

        step = 10**(-digits)
        t_sol = np.arange(tspan[0], tspan[1]+step, step)
        y_sol = np.array(Parallel(n_jobs=n_jobs,backend='loky')(delayed(self.s_experiment_hfcs)(tspan=tspan, states0=states0, crits=crits, basins=basins, basins_range=basins_range, 
                                                                                            gamma=gamma, digits=digits, args_list=args_list) for i in range(trials)))
        avg = np.mean(y_sol,axis=0)
        return [t_sol,avg]

def find_basin(points,basins,basins_range):
    m = points.shape[0]
    n = len(np.unique(basins))
    n2 = points.shape[1]
    tl = points.shape[2]

    basin_starts = np.empty(n2)
    dx = np.empty(n2)

    for i in range(n2):
        basin_starts[i] = basins_range[i][0]
        dx[i] = (basins.shape[i]-1)/(basins_range[i][1]-basins_range[i][0])
    
    points = points - np.transpose(np.full((m,tl,n2),basin_starts),axes=(0,2,1))
    points = np.transpose(einsum('abc,b->abc',points,dx).astype(int),axes=(1,0,2))

    out = np.zeros((m,n,tl))
    bs = (basins[tuple(points)]-1).astype(int)
    [t_mat,h_mat] = np.meshgrid(np.arange(tl),np.arange(m))

    out[h_mat,bs,t_mat] = 1.
    
    return out


def inter_ops(crits: np.array, basins: np.array, basins_range: list, gamma: float) -> np.array:

    n = len(crits)
    n2 = len(crits[0])
    ind = np.empty((2,n2,n,n),dtype=int)

    for i in range(n2):
        crits_temp = crits[:,i]
        dxi = (basins_range[i][1]-basins_range[i][0])/(basins.shape[i]-1)

        x1 = np.outer(crits_temp,(1-gamma)*np.ones(n))+np.outer(gamma*np.ones(n),crits_temp)
        x2 = np.outer(crits_temp,gamma*np.ones(n))+np.outer((1-gamma)*np.ones(n),crits_temp)
        ind[0,i] = np.round((x1-basins_range[i][0])/dxi).astype(int)
        ind[1,i] = np.round((x2-basins_range[i][0])/dxi).astype(int)
    
    b_after = np.zeros((n,n,n,n),dtype=int)
    [j_mat,i_mat] = np.meshgrid(np.arange(n),np.arange(n))
    i_mat = i_mat.flatten()
    j_mat = j_mat.flatten()
    k_mat = basins[tuple(ind[0])].flatten()-1
    l_mat = basins[tuple(ind[1])].flatten()-1

    b_after[i_mat,j_mat,k_mat,l_mat] = 1


    return b_after

def get_s0(states0):
    s0 = einsum('ak,bl->abkl',states0,states0)
    m = len(states0)
    n = len(states0[0])
    [k_inds,a_inds] = np.meshgrid(np.arange(n,dtype=int),np.arange(m,dtype=int))
    a_inds = a_inds.flatten()
    k_inds = k_inds.flatten()

    s0[np.arange(m),np.arange(m)] = 0
    s0[a_inds,a_inds,k_inds,k_inds] = states0[a_inds,k_inds]

    return s0


def make_r(s,lam):
    m = s.shape[0]
    n = s.shape[-1]
    num = einsum('ac,abkl,ackm->abcklm',lam,s,s)
    den = np.transpose(np.full((m,m,n,n,m,n),einsum("aakk->ak",s)),[4,0,1,5,2,3])
    den[den<1e-8] = 1
    return num/den

def make_sig(s,lam):
    m = s.shape[0]
    n = s.shape[-1]
    return np.transpose(np.full((n,n,m,m),lam),[2,3,0,1])*s

def pair_step(t,s,lam,phi,n,m,a_inds,k_inds):
    
    s = s.reshape(n,n,m,m)
    r = make_r(s,lam)
    sig = make_sig(s,lam)
    
    out = (-sig + einsum('abij,ijkl->abkl',sig,phi) - einsum('abcklm->abkl',r) - einsum('baclkm->abkl',r)
            + einsum('abcilm,imkn->abkl',r,phi) + einsum('bacjkm,jmln->abkl',r,phi)
            + einsum('baalkm->abkl',r) - einsum('baajkm,jmln->abkl',r,phi) + einsum('abbklm->abkl',r)
            - einsum('abbilm,imkn->abkl',r,phi) )
    out[np.arange(n),np.arange(n)] = 0
    out[a_inds,a_inds,k_inds,k_inds] = (-einsum('abkj->ak',sig) + einsum('abij,ijkl->ak',sig,phi)).flatten()
    return out.flatten()

    
def pair_evolve(s0,lam,phi,tspan, digits = 5):

    n = s0.shape[0]
    m = s0.shape[-1]
    [k_inds,a_inds] = np.meshgrid(np.arange(m,dtype=int),np.arange(n,dtype=int))
    a_inds = a_inds.flatten()
    k_inds = k_inds.flatten()

    step = 10**(-digits)
        
    t_sol = np.arange(tspan[0], tspan[1]+step, step)

    sol = solve_ivp(pair_step,tspan,s0.flatten(),args=(lam,phi,n,m,a_inds,k_inds),t_eval=t_sol)
    return sol

def find_basin(points,basins,basins_range):
    m = points.shape[0]
    n = len(np.unique(basins))
    n2 = points.shape[1]
    tl = points.shape[2]

    basin_starts = np.empty(n2)
    dx = np.empty(n2)

    for i in range(n2):
        basin_starts[i] = basins_range[i][0]
        dx[i] = (basins.shape[i]-1)/(basins_range[i][1]-basins_range[i][0])
    
    points = points - np.transpose(np.full((m,tl,n2),basin_starts),axes=(0,2,1))
    points = np.transpose(einsum('abc,b->abc',points,dx).astype(int),axes=(1,0,2))

    out = np.zeros((m,n,tl))
    bs = (basins[tuple(points)]-1).astype(int)
    [t_mat,h_mat] = np.meshgrid(np.arange(tl),np.arange(m))

    out[h_mat,bs,t_mat] = 1.
    
    return out


def inter_ops(crits: np.array, basins: np.array, basins_range: list, gamma: float) -> np.array:

    n = len(crits)
    n2 = len(crits[0])
    ind = np.empty((2,n2,n,n),dtype=int)

    for i in range(n2):
        crits_temp = crits[:,i]
        dxi = (basins_range[i][1]-basins_range[i][0])/(basins.shape[i]-1)

        x1 = np.outer(crits_temp,(1-gamma)*np.ones(n))+np.outer(gamma*np.ones(n),crits_temp)
        x2 = np.outer(crits_temp,gamma*np.ones(n))+np.outer((1-gamma)*np.ones(n),crits_temp)
        ind[0,i] = np.round((x1-basins_range[i][0])/dxi).astype(int)
        ind[1,i] = np.round((x2-basins_range[i][0])/dxi).astype(int)
    
    b_after = np.zeros((n,n,n,n),dtype=int)
    [j_mat,i_mat] = np.meshgrid(np.arange(n),np.arange(n))
    i_mat = i_mat.flatten()
    j_mat = j_mat.flatten()
    k_mat = basins[tuple(ind[0])].flatten()-1
    l_mat = basins[tuple(ind[1])].flatten()-1

    b_after[i_mat,j_mat,k_mat,l_mat] = 1


    return b_after

def get_s0(states0):
    s0 = einsum('ak,bl->abkl',states0,states0)
    m = len(states0)
    n = len(states0[0])
    [k_inds,a_inds] = np.meshgrid(np.arange(n,dtype=int),np.arange(m,dtype=int))
    a_inds = a_inds.flatten()
    k_inds = k_inds.flatten()

    s0[np.arange(m),np.arange(m)] = 0
    s0[a_inds,a_inds,k_inds,k_inds] = states0[a_inds,k_inds]

    return s0

def state_diff(ys1,ys2):
    return np.average(np.sqrt(np.sum((ys1-ys2)**2,axis=(0,1))))

def make_r(s,lam):
    m = s.shape[0]
    n = s.shape[-1]
    num = einsum('ac,abkl,ackm->abcklm',lam,s,s)
    den = np.transpose(np.full((m,m,n,n,m,n),einsum("aakk->ak",s)),[4,0,1,5,2,3])
    den[den<1e-7] = 1e-7
    return num/den

def make_sig(s,lam):
    m = s.shape[0]
    n = s.shape[-1]
    return np.transpose(np.full((n,n,m,m),lam),[2,3,0,1])*s

def pair_step(t,s,lam,phi,n,m,a_inds,k_inds):
    
    s = s.reshape(n,n,m,m)
    r = make_r(s,lam)
    sig = make_sig(s,lam)
    
    out = (-sig + einsum('abij,ijkl->abkl',sig,phi) - einsum('abcklm->abkl',r) - einsum('baclkm->abkl',r)
            + einsum('abcilm,imkn->abkl',r,phi) + einsum('bacjkm,jmln->abkl',r,phi)
            + einsum('baalkm->abkl',r) - einsum('baajkm,jmln->abkl',r,phi) + einsum('abbklm->abkl',r)
            - einsum('abbilm,imkn->abkl',r,phi) )
    out[np.arange(n),np.arange(n)] = 0
    out[a_inds,a_inds,k_inds,k_inds] = (-einsum('abkj->ak',sig) + einsum('abij,ijkl->ak',sig,phi)).flatten()
    return out.flatten()

    
def pair_evolve(s0,lam,phi,tspan, steps=1_000):

    n = s0.shape[0]
    m = s0.shape[-1]
    [k_inds,a_inds] = np.meshgrid(np.arange(m,dtype=int),np.arange(n,dtype=int))
    a_inds = a_inds.flatten()
    k_inds = k_inds.flatten()
    
    t_sol = np.linspace(tspan[0], tspan[1], steps+1)

    sol = solve_ivp(pair_step,[tspan[0],tspan[1]],s0.flatten(),args=(lam,phi,n,m,a_inds,k_inds),t_eval=t_sol)
    return sol