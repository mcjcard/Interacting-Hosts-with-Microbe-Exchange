from Setup import *

crits = np.load('crits.npy')
basins = np.load('basins.npy')
num_basins = len(crits)
basins_range = [[0,14],[0,14]]

def fun(t,y,a,b,c,d,e,f,l):
    return [-y[0]*(y[0]-a)*(y[0]-b)*(y[0]-c)/l,-y[1]*(y[1]-d)*(y[1]-e)*(y[1]-f)/l]

gammas = np.linspace(0.,0.5,61)
#gammas = [0.1,0.4]
lam0s = np.logspace(3,-2,21)[:2]

#gammas_valid = (gammas != 0.1)*(gammas != 0.4)
#gammas = gammas[gammas_valid]

#gammas_done = np.linspace(0.,0.5,31)
#lam0s_done = np.logspace(2,1,3)

#gammas = np.logspace(-1,-5,5)
#lam0s = np.logspace(1,5,5)

#exps = len(gammas)*len(lam0s) - len(gammas_done)*len(lam0s_done)
exps = len(lam0s)*len(gammas)

steps = 100

trials = 200

n_hosts = 10
#p_hosts = 0.5
#lam = 0.1*rand.choice(2,size=n_hosts*n_hosts,p=[1-p_hosts,p_hosts]).reshape((n_hosts,n_hosts))
#lam = np.tri(n_hosts,k=-1)*lam
#lam = lam + lam.T
#np.save(f"Experiment1Results/lam.npy",lam)

lam = 10*np.load(f"lam.npy",allow_pickle=True)

exp_count = 0
for lam0 in lam0s:
    for gamma in gammas:
        tspan = [0,1.]
        run_exp = True
        #if not (np.isin(gamma,gammas_done) and np.isin(lam0,lam0s_done)):
            #run_exp = True

        if run_exp:
            start = time.time()
        
            y0_list = np.load('y0_list.npy')
        
            G = nx.from_numpy_array(lam*lam0)
            net = NetDiff(G,fun)
            [ts_sim,ys_sim] = net.p_experiment_ys(tspan, y0_list, steps=steps, args_list=[[2,8,12,2,11,12,10.]]*n_hosts, trials=trials,
                                           gamma=gamma, n_jobs=48)
            np.save(f"HFCSA_Simulations/HFCSA_sim_gamma{gamma:.8f}_lam{lam0:.8f}.npy",ys_sim)
            
            end = time.time()
            exp_count += 1
            print(f'Completed {exp_count}/{exps} in {(end-start)/60:.3f} minutes')