from Setup import *

crits = np.load('crits.npy')
basins = np.load('basins.npy')
num_basins = len(crits)
basins_range = [[0,14],[0,14]]

def fun(t,y,a,b,c,d,e,f,l):
    return [-y[0]*(y[0]-a)*(y[0]-b)*(y[0]-c)/l,-y[1]*(y[1]-d)*(y[1]-e)*(y[1]-f)/l]

gammas = np.linspace(0.,0.5,61)
lam0s = np.logspace(3,-3,13)

gammas_valid = (gammas != 0.1)*(gammas != 0.4)
gammas = gammas[gammas_valid]

#gammas_done = np.linspace(0.,0.5,31)
#lam0s_done = np.logspace(2,1,3)

#gammas = np.logspace(-1,-5,5)
#lam0s = np.logspace(1,5,5)

#exps = len(gammas)*len(lam0s) - len(gammas_done)*len(lam0s_done)
exps = len(gammas)*len(lam0s)

steps = 1_000

trials = 1_000

n_hosts = 10
#p_hosts = 0.5
#lam = 0.1*rand.choice(2,size=n_hosts*n_hosts,p=[1-p_hosts,p_hosts]).reshape((n_hosts,n_hosts))
#lam = np.tri(n_hosts,k=-1)*lam
#lam = lam + lam.T
#np.save(f"Experiment1Results/lam.npy",lam)

lam = 10*np.load(f"Experiment1Results/lam.npy",allow_pickle=True)

exp_count = 0
for gamma in gammas:
    for lam0 in lam0s:
        tspan = [0,2./lam0]
        run_exp = True
        #if not (np.isin(gamma,gammas_done) and np.isin(lam0,lam0s_done)):
            #run_exp = True

        if run_exp:
            start = time.time()
        
            states0 = rand.dirichlet(np.ones(num_basins),size=n_hosts)
        
            G = nx.from_numpy_array(lam*lam0)
            net = NetDiff(G,fun)
            [ts_sim,ys_sim] = net.p_experiment(tspan, states0, crits, basins, basins_range, steps=steps, args_list=[[2,8,12,2,11,12,10.]]*n_hosts, trials=trials,
                                           gamma=gamma, n_jobs=48,ret_all=False)
            np.save(f"Experiment4Results/Exp4_gamma{gamma:.4f}_lam{lam0:.4f}.npy",ys_sim)
            
            end = time.time()
            exp_count += 1
            print(f'Completed {exp_count}/{exps} in {(end-start)/60:.3f} minutes')