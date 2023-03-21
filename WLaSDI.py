import numpy as np
import numpy.linalg as LA
import WLaSDI.wsindy as ws
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, Rbf
import scipy.integrate as integrate
from scipy.integrate import solve_ivp
from itertools import combinations_with_replacement
import time

class WLaSDI:
    """
    WLaSDI class for data-driven ROM. Functions: train_dynamics approximates dynamical systems of the latent-space. 
                                                generate_FOM uses an initial condition and parameter values to generate a new model
    NOTE: To avoid errors, make sure to set NN = True for use with autoencoder.
    
    Inputs:
       encoder: either neural network (with pytorch) or matrix (LS-ROM)
       decoder: either neural network (with pytorch) or matrix (LS-ROM)
       NN: Boolean on whether a NN is used
       device: device NN is on. Default 'cpu', use 'cuda' if necessary
       Local: Boolean. Determines Local or Global DI (still in progress)
       Coef_interp: Boolean. Determines method of Local DI
       nearest_neigh: Number of nearest neigh in Local DI
       Coef_interp_method: Either interp2d or Rbf method for coefficient interpolation.
    """
    
    def __init__(self, encoder, decoder, NN = False, device = 'cpu', Local = False, Coef_interp = False, nearest_neigh = 4, Coef_interp_method = None, plot_fname = 'latent_space_dynamics.png'):
        
        self.Local = Local
        self.Coef_interp = Coef_interp
        self.nearest_neigh = nearest_neigh
        self.NN = NN
        self.plot_fname = plot_fname
        if Coef_interp == True:
            if Coef_interp_method == None:
                print('WARNING: Please specify an interpolation method either interp2d or Rbf')
            else:
                self.Coef_interp_method = Coef_interp_method
            if nearest_neigh <4:
                print('WARNING: More minimum 4 nearest neighbors required for interpolation')
                return
        if NN == False:
            self.IC_gen = lambda params: np.matmul(encoder, params)
            self.decoder = lambda traj: np.matmul(decoder, traj.T)
            
        else:
            import torch
            self.IC_gen = lambda IC: encoder(torch.tensor(IC).to(device)).cpu().detach().numpy()
            self.decoder = lambda traj: decoder(torch.tensor(traj.astype('float32')).to(device)).cpu().detach().numpy()
            
        return
    
    def train_dynamics(self, ls_trajs, training_values, dt, normal = 1, degree = 1, include_interaction=False, LS_vis = True, gamma = 0, threshold = 0, L = 30, overlap = 0.5, opt_tfsupp = False, sampling_rate = 1):
        """
        Approximates the dynamical system for the latent-space. Local == True, use generate_FOM. 
        
        Inputs:
           ls_trajs: latent-space trajectories in a list of arrays formatted as [time, space]
           training_values: list/array of corresponding parameter values to above
           dt: time-step used in FOM
           normal: normalization constant. Default as 1
           LS_vis: Boolean to visulaize a trajectory and discovered dynamics in the latent-space. Default True

           WSINDy parameters
           L: test function support
           overlap: how much 2 consecutive test functions overlap. 
           opt_tfsupp: toggle the use of optimal test function support
        """
        self.dt = dt
        self.LS_vis = LS_vis
        self.normal = normal
        self.degree = degree
        self.threshold = threshold
        self.opt_tfsupp = opt_tfsupp
        self.L = L
        self.overlap = overlap
        self.gamma = gamma
        data_LS = []
        for traj in ls_trajs:
            data_LS.append(traj[0: len(traj)-1: sampling_rate]/normal)
            
        #print(data_LS[0].shape)
    
        if self.Local == False:
            model = ws.wsindy(polys=np.arange(0, degree+1), multiple_tracjectories=True, ld = threshold, gamma = gamma)

            #generating t
            t = [np.linspace(0, sampling_rate*dt*(len(data_LS[-1])-1), int(len(data_LS[-1])))]
            #print(t[0].shape)
            #print(t[0])
            for i in range(1, len(data_LS)):
                t.append(np.linspace(0, sampling_rate*dt*(len(data_LS[-1])-1), int(len(data_LS[-1]))))

            #using uniform grid
            model.getWSindyUniform(data_LS, t, L = L, overlap=overlap, opt_tfsupp=opt_tfsupp)


            self.model = model
            if LS_vis == True:
                if self.NN == True:
                    DcTech = 'LaSDI-NM Latent-Space Visualization'
                    DcTech = 'Latent-Space Dynamics by Nonlinear Compression'
                else:
                    DcTech = 'LaSDI-LS Latent-Space Visualization'
                    DcTech = 'Latent-Space Dynamics by Linear Compression'
                time = np.linspace(0, dt*(len(data_LS[-1])-1), len(data_LS[-1]))
                fig = plt.figure()
                fig.set_size_inches(9,6)
                ax = plt.axes()
                ax.set_title(DcTech)
                labels = {'orig': 'Latent-Space Trajectory', 'new': 'Approximated Dynamics'}
                for dim in range(data_LS[-1].shape[1]):
                    plt.plot(time[:-1], data_LS[-1][:-1,dim], alpha = .5, label = labels['orig'])
                    labels['orig'] = '_nolegend_'
                plt.gca().set_prop_cycle(None)

                new = model.simulate(x0 = data_LS[-1][0], t_span = np.array([0, dt*len(data_LS[-1])]), t_eval = np.linspace(0, dt*(len(data_LS[-1])-1), len(data_LS[-1])))
                #print(data_LS[-1][0])
                #print(np.array([0, dt*len(data_LS[-1])]))
                #print(np.linspace(0, dt*(len(data_LS[-1])-1), len(data_LS[-1])).shape)
                #print(new.shape)
                #print(model.coef)

                for dim in range(data_LS[-1].shape[1]):
                    plt.plot(time, new[:,dim], '--', label = labels['new'])
                    labels['new'] = '_nolegend_'
                ax.legend()
                ax.set_xlabel('Time')
                ax.set_ylabel('Magnitude')
                #plt.savefig(self.plot_fname)
            return model.coef 
        elif self.Coef_interp == True:
            #("Local approach WITH SINDy coefficient interpolation")
            if self.Coef_interp_method == None:
                print('WARNING: Please specify an interpolation method either interp2d or Rbf')
            self.model_list = []
            self.training_values = training_values
            self.dt = dt
            self.degree = degree
            self.length = len(data_LS[0])
            for i, _ in enumerate(training_values):
                model = ws.wsindy(polys=np.arange(0, self.degree+1), multiple_tracjectories=False, ld=threshold, gamma=gamma )
                #generating t
                t = np.linspace(0, dt*(len(data_LS[i])-1), len(data_LS[i]))

                #using Adaptive grid
                #model.getWsindyAdaptive(data_LS[i], t)

                #using uniform grid
                model.getWSindyUniform(data_LS[i], t, opt_tfsupp=opt_tfsupp, L = L, overlap=overlap)
                self.model_list.append(model.coef)
                self.tags = model.tags
                if LS_vis == True:
                    if self.NN == True:
                        DcTech = 'LaSDI-NM Latent-Space Visualization'
                        DcTech = 'Latent-Space Dynamics by Nonlinear Compression'
                    else:
                        DcTech = 'LaSDI-LS Latent-Space Visualization'
                        DcTech = 'Latent-Space Dynamics by Linear Compression'
                    time = np.linspace(0, dt*(len(data_LS[-1])-1), len(data_LS[-1]))
                    fig = plt.figure()
                    fig.set_size_inches(9,6)
                    ax = plt.axes()
                    ax.set_title(DcTech)
                    labels = {'orig': 'Latent-Space Trajectory', 'new': 'Approximated Dynamics'}
                    for dim in range(data_LS[-1].shape[1]):
                        plt.plot(time[:-1], data_LS[-1][:-1,dim], alpha = .5, label = labels['orig'])
                        labels['orig'] = '_nolegend_'
                    plt.gca().set_prop_cycle(None)

                    new = model.simulate(data_LS[-1][0], np.array([0, dt*len(data_LS[-1])]), np.linspace(0, dt*(len(data_LS[-1])-1), len(data_LS[-1])))
                    for dim in range(data_LS[-1].shape[1]):
                        plt.plot(time, new[:,dim], '--', label = labels['new'])
                        labels['new'] = '_nolegend_'
                    ax.legend()
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Magnitude')
                    #plt.savefig(self.plot_fname)

            return self.model_list
        else:
            #print("Local approach WITHOUT SINDy coefficient interpolation")
            self.ls_trajs = ls_trajs
            self.training_values = training_values
            self.dt = dt
            self.degree = degree
            self.include_interaction = include_interaction
            self.data_LS = data_LS
            
            return 
        
    
        
    def generate_ROM(self,pred_IC,pred_value,t):
        """
        Takes initial condition in full-space and associated parameter values and generates forward in time using the trained dynamics from above.
        Inputs:
            pred_IC: Initial condition of the desired simulation
            pred_value: Associated parameter values
            t: time stamps corresponding to training FOMs
        """
        IC = self.IC_gen(pred_IC)
        if self.Local == False: # Global approach
            latent_space_recon = self.normal*self.model.simulate(IC/self.normal, np.array([t[0], t[-1]]), t)
            FOM_recon = self.decoder(latent_space_recon)
            if self.NN == False:
                return FOM_recon.T
            else:
                return FOM_recon
        else: # Local approach
            training_time_start = time.time()
            dist = np.empty(len(self.training_values))
            for iii,P in enumerate(self.training_values):
                dist[iii]=(LA.norm(P-pred_value))

            k = self.nearest_neigh
            dist_index = np.argsort(dist)[0:k]
            self.dist_index = dist_index


            if self.Coef_interp == False: # WITHOUT SINDy coefficient interpolation
                local = []
                for iii in dist_index:
                    local.append(self.data_LS[iii])
                model = ws.wsindy(polys=np.arange(0, self.degree+1), multiple_tracjectories=True, ld=self.threshold, gamma=self.gamma)
                t_val = [np.linspace(0, self.dt*(len(local[-1])-1), len(local[-1]))]
                for i in range(1, len(local)):
                    t_val.append(np.linspace(0, self.dt*(len(local[-1])-1), len(local[-1])))

                #using uniform grid
                model.getWSindyUniform(local, t_val, opt_tfsupp = self.opt_tfsupp, L = self.L, overlap = self.overlap )

                #using adaptive grid
                #model.getWsindyAdaptive(local, t_val)

                self.training_time = time.time()-training_time_start
                latent_space_recon = self.normal*model.simulate(IC/self.normal, np.array([t[0], t[-1]]), t)
                FOM_recon = self.decoder(latent_space_recon)
                
                if self.LS_vis == True:
                    if self.NN == True:
                        DcTech = 'LaSDI-NM Latent-Space Visualization'
                        DcTech = 'Latent-Space Dynamics by Nonlinear Compression'
                    else:
                        DcTech = 'LaSDI-LS Latent-Space Visualization'
                        DcTech = 'Latent-Space Dynamics by Linear Compression'
                    ti = np.linspace(0, self.dt*(len(local[-1])-1), len(local[-1]))
                    fig = plt.figure()
                    fig.set_size_inches(9,6)
                    ax = plt.axes()
                    ax.set_title(DcTech)
                    labels = {'orig': 'Latent-Space Trajectory', 'new': 'Approximated Dynamics'}
                    for dim in range(local[-1].shape[1]):
                        plt.plot(ti[:-1], local[-1][:-1,dim], alpha = .5, label = labels['orig'])
                        labels['orig'] = '_nolegend_'
                    plt.gca().set_prop_cycle(None)

                    new = model.simulate(x0 = local[-1][0], t_span = np.array([0, self.dt*len(local[-1])]), t_eval = np.linspace(0, self.dt*(len(local[-1])-1), len(local[-1])))
                    #print(data_LS[-1][0])
                    #print(np.array([0, dt*len(data_LS[-1])]))
                    #print(np.linspace(0, dt*(len(data_LS[-1])-1), len(data_LS[-1])).shape)
                    #print(new.shape)
                    #print(model.coef)

                    for dim in range(local[-1].shape[1]):
                        plt.plot(ti, new[:,dim], '--', label = labels['new'])
                        labels['new'] = '_nolegend_'
                    ax.legend()
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Magnitude')
                
                if self.NN == False:
                    return FOM_recon.T
                else:
                    return FOM_recon

            else: # WITH SINDy coefficient interpolation
                # print(self.dist_index)
                self.coeff_interp_model = np.empty(self.model_list[0].shape)
                self.training_time = 0
                
                # Compute SINDy coefficients of the testing parameter by interpolation
                for ls_dim in range(self.model_list[0].shape[0]):
                    for func_index in range(self.model_list[0].shape[1]):
                        f = self.Coef_interp_method(self.training_values[dist_index,0], 
                                                    self.training_values[dist_index,1], 
                                                    np.array(self.model_list)[dist_index,ls_dim,func_index])

                        self.coeff_interp_model[ls_dim, func_index] = f(pred_value[0], pred_value[1])
                #print(self.coeff_interp_model) # (ns,nl)
            
                self.time = np.arange(0,self.dt*self.length, self.dt)
                #print(self.tags)
                def simulate(x0, t_span, t_eval, coef):
                    rows, cols = self.tags.shape
                    tol_ode = 10**(-13)
                    def rhs(t, x):
                        term = np.ones(rows)
                        for row in range(rows):
                            for col in range(cols): 
                                term[row] = term[row]*x[col]**self.tags[row, col]
                        return term.dot(coef)
                    sol = solve_ivp(fun = rhs, t_eval=t_eval, t_span=t_span, y0=x0, rtol=tol_ode)
                    return sol.y.T
                self.latent_space_recon = self.normal*simulate(IC/self.normal, t_eval=self.time, t_span = np.array([self.time[0], self.time[-1]]), coef = self.coeff_interp_model)
                
                #plt.plot(self.latent_space_recon)
                # high-dimensional dynamics
                FOM_recon = self.decoder(self.latent_space_recon)
                if self.NN == False:
                    return FOM_recon.T
                else:
                    return FOM_recon
                return
            
        
        

            
