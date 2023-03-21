import numpy as np
from scipy.optimize import brentq
import numpy as np
import itertools
import operator
from scipy.linalg import lstsq
from numpy import matlib as mb
import scipy 
from scipy.integrate import solve_ivp
#from kneed import KneeLocator


class wsindy:
    """
     Inputs:
       polys: monomial powers to include in library
       trigs: sine / cosine frequencies to include in library
       scale_theta: normalize columns of theta. Ex: scale_theta = 0 means no normalization, scale_theta = 2 means l2 normalization.
       ld: sequential thresholding parameter 
       tau_p: test function has value 10^-tau_p at penultimate support point. Or, if tau_p<0, directly sets poly degree p = -tau_p
       gamma: Tikhonoff regularization parameter
    """
    def __init__(self, polys =  np.arange(0, 6), trigs = [], scaled_theta = 0, ld = 0.001, tau_p = 16, gamma = 10**(-np.inf), multiple_tracjectories = False):
        self.polys = polys
        self.trigs = trigs
        self.scale_theta = scaled_theta
        self.ld = ld
        self.tau_p = tau_p
        self.gamma = gamma
        self.coef = None
        self.multiple_trajectories = multiple_tracjectories


    """
       xobs: x values
       tobs: t values
       r_whm: width half max
       s:
       K: # of test function
       p: FD order 
       tau: toggle adaptive grid
    """
    def getWsindyAdaptive(self, xobs, tobs, r_whm = 30, s = 16, K = 200, p = 2, tau = 1, useGLS = 10**(-12)):

        if self.multiple_trajectories == True:
            x_values = xobs[0]
            t_values = tobs[0]
            for i in range(1, len(xobs)):
                x_values = np.vstack((x_values, xobs[i]))
                t_values = np.hstack((t_values, tobs[i]))
            xobs = x_values
            tobs = t_values


        wsindy_params = [s, K, p, tau]
        Theta_0, tags, M_diag = self.buildTheta(xobs)

        n = xobs.shape[1]

        w_sparse = np.zeros((Theta_0.shape[1], n))
        #mats = [] 
        #ps_all = [[]]
        #ts_grids = []  
        #RTs = []  
        #Ys = []  
        #Gs = [] 
        #bs = [] 

        for i in range(n):
            grid_i = self.Adaptive_Grid(tobs, xobs[:, i], wsindy_params)
            V, Vp = self.VVp_build_adaptive_whm(
                tobs, grid_i, r_whm, [0, np.inf, 0])
            #ps_all = ps_all.append(ps)
            #mats.append([V, Vp])
            #ts_grids.append(ab_grid)
            #Ys.append(Y)
            #print("use GLS = ", useGLS)
            if useGLS > 0:
                Cov = Vp.dot(Vp.T) + useGLS*np.identity(V.shape[0])
                RT = np.linalg.cholesky(Cov)
                G = lstsq(RT, V.dot(Theta_0))[0]
                b = lstsq(RT, Vp.dot(xobs[:, i]))[0]
            else:
                RT = 1/np.linalg.norm(Vp, 2, 1)
                RT = np.reshape(RT, (RT.size, 1))
                G = np.multiply(V.dot(Theta_0), RT)
                temp = Vp.dot(xobs[:, i])
                b = RT.T*temp

            #RTs.append(RT)
            #Gs.append(G)
            #bs.append(b)

            if self.scale_theta > 0:
                w_sparse_temp = self.sparsifyDynamics(
                    np.multiply(G, (1/M_diag.T)), b, 1)
                w_sparse[:, i] = np.ndarray.flatten(
                    np.multiply((1/M_diag), w_sparse_temp))
            else:
                # print(gamma)
                #w_sparse[:,i] = np.ndarray.flatten(SparsifyDynamics.sparsifyDynamics(G,b,ld,1,gamma))
                w_sparse_temp = self.sparsifyDynamics(G, b, 1)
                w_sparse[:, i] = np.ndarray.flatten(w_sparse_temp)
            
        self.coef = w_sparse
        self.tags = tags
        return self #w_sparse, ts_grids, mats

    
    """
       xobs: x value
       tobs: T value
       L: test function support
       overlap: 
    """
    
    def getWSindyUniform(self, xobs, tobs, L = 30, overlap = 0.5, useGLS =  10**(-12), opt_tfsupp = False):

        self.opt_tfsupp = opt_tfsupp

        if self.multiple_trajectories == True:
            x_values = xobs[0]
            t_values = tobs[0]
            #print("x", x_values.shape)
            #print("t", t_values.shape)
            for i in range(1, len(xobs)):
                x_values = np.vstack((x_values, xobs[i]))
                t_values = np.hstack((t_values, tobs[i]))
            xobs = x_values
            tobs = t_values

        M = len(tobs)
        Theta_0, tags, M_diag = self.buildTheta(xobs)

        n = xobs.shape[1]
        w_sparse = np.zeros((Theta_0.shape[1], n))
        #res = []
        #mats = []  
        #ps_all = [[]]
        #ts_grids = []  
        #RTs = [] 
        #Gs = [] #[n,1]
        #bs = [] #[n,1]

        if self.opt_tfsupp == False:
            V, Vp = self.Uniform_grid(tobs, L, overlap, [0, np.inf, 0] )

        for i in range(n):
            
            #if self.opt_tfsupp == True:
                #k = self.get_kneepoint(xobs[:, i])
                #print('k', k)
                #L = int(self.get_support(M, 2, k)*2)
                #print('support', L)
                #print("k", k)
                #V, Vp = self.Uniform_grid(tobs, L, overlap, [0, np.inf, 0] )

            #mats.append([V, Vp])
            #ts_grids.append(ab_grid)

            if useGLS > 0:
                Cov = Vp.dot(Vp.T) + useGLS*np.identity(V.shape[0])
                RT = np.linalg.cholesky(Cov)
                G = lstsq(RT, V.dot(Theta_0))[0]
                b = lstsq(RT, Vp.dot(xobs[:, i]))[0]
            else:
                #print("here")
                RT = 1/np.linalg.norm(Vp, 2, 1)
                RT = np.reshape(RT, (RT.size, 1))
                G = np.multiply(V.dot(Theta_0), RT)
                temp = Vp.dot(xobs[:, i])
                b = RT.T*temp


            if self.scale_theta > 0:
                w_sparse_temp = self.sparsifyDynamics(
                    np.multiply(G, (1/M_diag.T)), b, 1)
                temptemp = np.ndarray.flatten(
                    np.multiply((1/M_diag), w_sparse_temp))
                w_sparse[:, i] = temptemp
            else:
                w_sparse_temp = self.sparsifyDynamics(
                    G, b, 1)
                w_sparse[:, i] = np.ndarray.flatten(w_sparse_temp)

            #RTs.append(RT)
            #Gs.append(G)
            #bs.append(b)
        self.coef = w_sparse
        self.tags = tags
        return  self #w_sparse,  ts_grids , mats 
    
    def simulate(self, x0, t_span, t_eval):
        #print(self.tags)
        #print(self.coef)

        rows, cols = self.tags.shape
        tol_ode = 10**(-13)
        def rhs(t, x):
            term = np.ones(rows)
            for row in range(rows):
                for col in range(cols): 
                    term[row] = term[row]*x[col]**self.tags[row, col]
            return term.dot(self.coef)

        #print(len(t_eval))
        sol = solve_ivp(fun = rhs, t_eval=t_eval, t_span=t_span, y0=x0, rtol=tol_ode)
        return sol.y.T

    def Uniform_grid(self, t, L, s, param):
        M = len(t)
        #p = int(np.floor(1/8*((L**2*rho**2 - 1) + np.sqrt((L**2*rho**2 - 1)**2 - 8*L**2*rho**2))))
        p = 16

        overlap = int(np.floor(L*(1 - np.sqrt(1 - s**(1/p)))))
        #print("support and overlap", L, overlap)

        # create grid
        grid = []
        a = 0
        b = L
        grid.append([a, b])
        while b - overlap + L <= M-1:
            a = b - overlap
            b = a + L
            grid.append([a, b])

        grid = np.asarray(grid)
        N = len(grid)
        
        V = np.zeros((N, M))
        Vp = np.zeros((N, M))

        for k in range(N):
            g, gp = self.basis_fcn(p, p)
            a = grid[k][0]
            b = grid[k][1]
            V_row, Vp_row = self.tf_mat_row(g, gp, t, a, b, param)
            V[k, :] = V_row
            Vp[k, :] = Vp_row

        return V, Vp, #grid

    
    def tf_mat_row(self, g, gp, t, t1, tk, param):
        N = len(t)
        if param == None:
            pow = 1
            gap = 1
            nrm = np.inf
            ord = 0
        else:
            pow = param[0]
            nrm = param[1]
            ord = param[2]
            gap = 1

        if t1 > tk:
            tk_temp = tk
            tk = t1
            t1 = tk_temp

        V_row = np.zeros((1, N))
        Vp_row = np.copy(V_row)

        t_grid = t[t1:tk+1:gap]
        dts = np.diff(t_grid)
        w = 1/2*(np.append(dts, [0]) + np.append([0], dts))

        V_row[:, t1:tk+1:gap] = g(t_grid, t[t1], t[tk])*w
        Vp_row[:, t1:tk+1:gap] = -gp(t_grid, t[t1], t[tk])*w
        Vp_row[:, t1] = Vp_row[:, t1] - g(t[t1], t[t1], t[tk])
        Vp_row[:, tk] = Vp_row[:, tk] + g(t[tk], t[t1], t[tk])

        if pow != 0:
            if ord == 0:
                scale_fac = np.linalg.norm(
                    np.ndarray.flatten(V_row[:, t1:tk+1:gap]), nrm)
            elif ord == 1:
                scale_fac = np.linalg.norm(
                    np.ndarray.flatten(Vp_row[:, t1:tk+1:gap]), nrm)
            else:
                scale_fac = np.mean(dts)
            Vp_row = Vp_row/scale_fac
            V_row = V_row/scale_fac
        return V_row, Vp_row


    def Adaptive_Grid(self, t, xobs, params=None):
        if params == None:
            index_gap = 16
            K = max(int(np.floor(len(t))/50), 4)
            p = 2
            tau = 1
        else:
            index_gap = params[0]
            K = params[1]
            p = params[2]
            tau = params[3]

        M = len(t)
        g, gp = self.basis_fcn(p, p)
        o, Vp_row = self.AG_tf_mat_row(g, gp, t, 1, 1+index_gap, [1, 1, 0])
        Vp_diags = mb.repmat(Vp_row[:, 0:index_gap+1], M - index_gap, 1)
        Vp = scipy.sparse.diags(Vp_diags.T, np.arange(
            0, index_gap+1), (M-index_gap, M))
        weak_der = Vp.dot(xobs)
        weak_der = np.append(np.zeros((int(np.floor(index_gap/2)), 1)), weak_der)
        weak_der = np.append(weak_der, np.zeros((int(np.floor(index_gap/2)), 1)))

        Y = np.abs(weak_der)
        Y = np.cumsum(Y)
        Y = Y/Y[-1]

        Y = tau*Y + (1-tau)*np.linspace(Y[0], Y[-1], len(Y)).T

        temp1 = Y[int(np.floor(index_gap/2)) - 1]
        temp2 = Y[int(len(Y) - np.ceil(index_gap/2)) - 1]
        U = np.linspace(temp1, temp2, K+2)

        final_grid = np.zeros((1, K))

        for i in range(K):
            final_grid[0, i] = np.argwhere((Y-U[i+1] >= 0))[0]

        final_grid = np.unique(final_grid)
        #print("length grid", len(final_grid))
        return final_grid #y


    def AG_tf_mat_row(self, g, gp, t, t1, tk, param=None):
        N = len(t)

        if param == None:
            gap = 1
            nrm = np.inf
            ord = 0
        else:
            gap = param[0]
            nrm = param[1]
            ord = param[2]

        if t1 > tk:
            tk_temp = tk
            tk = t1
            t1 = tk_temp

        V_row = np.zeros((1, N))
        Vp_row = np.copy(V_row)

        #print(t1, tk, gap)
        t_grid = t[t1:tk+1:gap]

        dts = np.diff(t_grid)
        w = 1/2*(np.append(dts, [0]) + np.append([0], dts))

        V_row[:, t1:tk+1:gap] = g(t_grid, t[t1], t[tk])*w
        Vp_row[:, t1:tk+1:gap] = -gp(t_grid, t[t1], t[tk])*w
        Vp_row[:, t1] = Vp_row[:, t1] - g(t[t1], t[t1], t[tk])
        Vp_row[:, tk] = Vp_row[:, tk] + g(t[tk], t[t1], t[tk])

        if ord == 0:
            scale_fac = np.linalg.norm(
                np.ndarray.flatten(V_row[:, t1:tk+1:gap]), nrm)
        elif ord == 1:
            scale_fac = np.linalg.norm(
                np.ndarray.flatten(Vp_row[:, t1:tk+1:gap]), nrm)
        else:
            scale_fac = np.mean(dts)
        Vp_row = Vp_row/scale_fac
        V_row = V_row/scale_fac
        return V_row, Vp_row

    def VVp_build_adaptive_whm(self, t, centers, r_whm, param=None):
        if param == None:
            param = [1, 2, 1]

        N = len(t)
        M = len(centers)
        V = np.zeros((M, N))
        Vp = np.zeros((M, N))
        ab_grid = np.zeros((M, 2))
        ps = np.zeros((M, 1))
        p, a, b = self.test_fcn_param(r_whm, t[int(centers[0]-1)], t)

        a = int(a)
        b = int(b)

        if b-a < 10:
            center = (a+b)/2
            a = int(max(0, np.floor(center-5)))
            b = int(min(np.ceil(center+5), len(t)))

        g, gp = self.basis_fcn(p, p)
        V_row, Vp_row = self.tf_mat_row(g, gp, t, a, b, param)

        V[0, :] = V_row
        Vp[0, :] = Vp_row
        ab_grid[0, :] = np.array([a, b])
        ps[0] = p

        for k in range(1, M):
            cent_shift = int(centers[k] - centers[k-1])
            b_temp = min(b + cent_shift, len(t))

            if a > 0 and b_temp < len(t):
                a = a + cent_shift
                b = b_temp
                V_row = np.roll(V_row, cent_shift)
                Vp_row = np.roll(Vp_row, cent_shift)
            else:
                p, a, b = self.test_fcn_param(
                    r_whm, t[int(centers[k]-1)], t)
                a = int(a)
                b = int(b)
                if b-a < 10:
                    center = (a+b)/2
                    b = int(min(np.ceil(center+5), len(t)))
                    a = int(max(0, np.floor(center-5)))
                g, gp = self.basis_fcn(p, p)
                V_row, Vp_row = self.tf_mat_row(g, gp, t, a, b, param)

            V[k, :] = V_row
            Vp[k, :] = Vp_row

            ab_grid[k, :] = np.array([a, b])
            ps[k] = p
        return V, Vp#, ab_grid, ps


    def sparsifyDynamics(self, Theta, dXdt, n, M=None):
        if M is None:
            M = np.ones((Theta.shape[1], 1))

        if self.gamma == 0:
            Theta_reg = Theta
            dXdt_reg = np.reshape(dXdt, (dXdt.size, 1))
        else:
            nn = Theta.shape[1]
            Theta_reg = np.vstack((Theta, self.gamma*np.identity(nn)))
            dXdt = np.reshape(dXdt, (dXdt.size, 1))
            dXdt_reg_temp = np.vstack((dXdt, self.gamma*np.zeros((nn, n))))
            dXdt_reg = np.reshape(dXdt_reg_temp, (dXdt_reg_temp.size, 1))
            #print(nn)
        
        #print("theta", Theta_reg.shape)
        #print("dXdt_reg", dXdt_reg.shape)

        Xi = M*(lstsq(Theta_reg, dXdt_reg)[0])

        for i in range(10):
            smallinds = (abs(Xi) < self.ld)
            while np.argwhere(np.ndarray.flatten(smallinds)).size == Xi.size:
                self.ld = self.ld/2
                smallinds = (abs(Xi) < self.ld)
            Xi[smallinds] = 0
        for ind in range(n):
            biginds = ~smallinds[:, ind]
            temp = dXdt_reg[:, ind]
            temp = np.reshape(temp, (temp.size, 1))
            Xi[biginds, ind] = np.ndarray.flatten(
                M[biginds]*(lstsq(Theta_reg[:, biginds], temp)[0]))
        #residual = np.linalg.norm((Theta_reg.dot(Xi)) - dXdt_reg)
        return Xi

    def buildTheta(self, xobs):
        theta_0, tags = self.poolDatagen(xobs)
        if self.scale_theta > 0:
            M_diag = np.linalg.norm(theta_0, self.scale_theta, 0)
            M_diag = np.reshape(M_diag, (len(M_diag), 1))
            return theta_0, tags, M_diag
        else:
            M_diag = np.array([])
            return theta_0, tags, M_diag

    def poolDatagen(self, xobs):
        # generate monomials
        n, d = xobs.shape
        if len(self.polys) != 0:
            P = self.polys[-1]
        else:
            P = 0
        rhs_functions = {}
        def f(t, x): return np.prod(np.power(list(t), list(x)))
        powers = []
        for p in range(1, P+1):
            size = d + p - 1
            for indices in itertools.combinations(range(size), d-1):
                starts = [0] + [index+1 for index in indices]
                stops = indices + (size,)
                powers.append(tuple(map(operator.sub, stops, starts)))
        for power in powers:
            rhs_functions[power] = [lambda t, x=power: f(t, x), power]

        theta_0 = np.ones((n, 1))
        #print(powers)

        tags = np.array(powers)
        #print('tags', tags)
        # plug in
        for k in rhs_functions.keys():
            func = rhs_functions[k][0]
            new_column = np.zeros((n, 1))
            for i in range(n):
                new_column[i] = func(xobs[i, :])
            theta_0 = np.hstack([theta_0, new_column])

        # trigs:
        for i in range(len(self.trigs)):
            trig_inds = np.array([-self.trigs[i]*1j*np.ones(d), self.trigs[i]*1j*np.ones(d)])
            sin_col = np.zeros((n, 1))
            cos_col = np.zeros((n, 1))
            for m in range(n):
                sin_col[m] = np.sin(self.trigs[i]*xobs[m, :])
                cos_col[m] = np.cos(self.trigs[i]*xobs[m, :])
            theta_0 = np.hstack([theta_0, sin_col, cos_col])
            tags = np.vstack([tags, trig_inds])
        
        if len(tags) != 0:
            tags = np.vstack([np.zeros((1, d)), tags])
        else:
            tags = np.zeros((1, d))
        return theta_0, tags
    
    def basis_fcn(self, p, q):
        def g(t, t1, tk): return (p > 0)*(q > 0)*(t - t1)**max(p, 0)*(tk - t)**max(q, 0) + (p == 0)*(q == 0)*(1 - 2 *
                                                                                                            np.abs(t - (t1+tk)/2)/(tk - t1)) + (p > 0)*(q < 0)*np.sin(p*np.pi/(tk - t1)*(t - t1)) + (p == -1)*(q == -1)

        def gp(t, t1, tk): return (t-t1)**(max(p-1, 0))*(tk-t)**(max(q-1, 0))*((-p-q)*t+p*tk+q*t1)*(q > 0)*(p > 0) + -2*np.sign(t -
                                                                                                                                (t1+tk)/2)/(tk-t1)*(q == 0)*(p == 0) + p*np.pi/(tk-t1)*np.cos(p*np.pi/(tk-t1)*(t-t1))*(q < 0)*(p > 0) + 0*(p == -1)*(q == -1)

        if p > 0 and q > 0:
            def normalize(t, t1, tk): return (
                t - t1)**max(p, 0)*(tk - t)**max(q, 0)

            def g(t, t1, tk): return ((p > 0)*(q > 0)*(t - t1)**max(p, 0)*(tk - t)**max(q, 0) + (p == 0)*(q == 0)*(1 - 2*np.abs(t - (t1+tk)/2) /
                                                                                                                (tk - t1)) + (p > 0)*(q < 0)*np.sin(p*np.pi/(tk - t1)*(t - t1)) + (p == -1)*(q == -1))/(np.abs(normalize((q*t1+p*tk)/(p+q), t1, tk)))

            def gp(t, t1, tk): return ((t-t1)**(max(p-1, 0))*(tk-t)**(max(q-1, 0))*((-p-q)*t+p*tk+q*t1)*(q > 0)*(p > 0) + -2*np.sign(t-(t1+tk)/2)/(tk-t1)*(q == 0)
                                    * (p == 0) + p*np.pi/(tk-t1)*np.cos(p*np.pi/(tk-t1)*(t-t1))*(q < 0)*(p > 0) + 0*(p == -1)*(q == -1))/(np.abs(normalize((q*t1+p*tk)/(p+q), t1, tk)))

        return g, gp

    def test_fcn_param(self, r, c, t, p=None):
        if self.tau_p < 0:
            self.tau_p = - self.tau_p
        else:
            p = self.tau_p
            self.tau_p = 16
        dt = t[1]-t[0]
        r_whm = r*dt
        A = np.log2(10)*self.tau_p
        def gg(s): return -s**2*((1-(r_whm/s)**2)**A-1)
        def hh(s): return (s-dt)**2
        def ff(s): return hh(s)-gg(s)

        s = brentq(ff, r_whm, r_whm*np.sqrt(A)+dt)

        if p == None:
            p = min(np.ceil(max(-1/np.log2(1-(r_whm/s)**2), 1)), 200)

        a = np.argwhere((t >= (c-s)))
        if len(a) != 0:
            a = a[0]
        else:
            a = []

        if c+s > t[-1]:
            b = len(t)-1
        else:
            b = np.argwhere((t >= (c+s)))[0]
        return p, a, b

    '''''
    def get_kneepoint(self, x):
        M = len(x)
        N = int(np.ceil(M/2))
        #print(M)
        Ufft = abs(np.fft.fftshift(np.fft.fft(x)))
        Ufft = np.cumsum(Ufft)
        Ufft = Ufft[0:N]
        kn = KneeLocator(np.arange(N), Ufft, curve='convex',
                        direction='increasing')
        #kn.plot_knee()
        return N - kn.knee + 1
    '''''

    #Code by Dan Messenger
    def get_support(self, N, tauhat, k):
        ks = (np.arange(0, N) - np.floor(N/2))**2
        err = np.zeros((np.floor(N/2).astype(int)-1, 1))
        m = 1
        err[0] = np.abs((k/tauhat)**2-np.sum(ks))
        check = 0
        while check == False and m <= (N-3)/2:
            m = m + 1
            x = np.linspace(-1, 1, 2*m + 1)
            phigrid = self.get_tf(16)(x)
            phigrid = np.hstack((phigrid, np.zeros((N-2*m-1))))
            phi_fft = np.fft.fftshift(np.abs(np.fft.fft(phigrid)))
            phi_fft = phi_fft/np.sum(phi_fft)
            err[m-1] = np.abs((k/tauhat)**2-np.sum(phi_fft*ks))
            check = err[m-1] > err[m-2] 
        return m

    #Test function
    def get_tf(self, p):
        def g(t): return (1-t**2)**p
        return g

