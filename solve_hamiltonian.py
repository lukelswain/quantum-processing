import numpy as np
import matplotlib.pyplot as plt
from scipy import constants, linalg
import matplotlib as mpl

plt.rcParams['lines.linewidth'] = 1
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

n1 = 400
n2 = 50
ts = np.linspace(0, 1e-6, n1)
dt = 1e-6/n1
qs = np.linspace(0, 35, n2)
colormap = mpl.colormaps["cool"].resampled(n2)(range(n2))



a = 1.2
constant_fn = lambda t, max: max + 0*t
t_fn_1 = lambda t, max: 8*max*((t-(ts[-1]/2))/ts[-1])**3
t_fn_2 = lambda t, max: 4*max*((t-(ts[-1]/2))/ts[-1])**2
t_fn_3 = lambda t, max: max*((2/ts[-1])*(t-(ts[-1]/2)))**5
t_fn_4 = lambda t, max: max*((2/ts[-1])*(t-(ts[-1]/2)))
t_sine_1 = lambda t, max: max*np.sin(2*np.pi*t/ts[-1])
q_fn_1 = lambda q, max: max*10**(-(1/a)*np.log(q+1))
q_fn_2 = lambda q, max: max*q/20


class Hamiltonian:
    def __init__(self, omega, delta_func, delta_max, phi, v):
        self.omega = omega
        self.delta_func = delta_func
        self.delta_max = delta_max
        self.phi = phi
        self.v = v

    def v_update(self, v_func, v_max, q):
        self.v = v_func(q, v_max)

    def matrix(self, t):
        h = (constants.hbar/2)*np.array([

            [2*self.delta_func(t, self.delta_max), self.omega*np.exp(-1j*self.phi), self.omega*np.exp(-1j*self.phi), 0],
            [self.omega*np.exp(1j*self.phi), 0, 0, self.omega*np.exp(-1j*self.phi)],
            [self.omega*np.exp(1j*self.phi), 0, 0, self.omega*np.exp(-1j*self.phi)],
            [0, self.omega*np.exp(1j*self.phi), self.omega*np.exp(1j*self.phi), -2*self.delta_func(t, self.delta_max) + 2*self.v]

            ])

        return h

    def eig(self, t):
        h = self.matrix(t)
        return np.linalg.eig(h)

    def evolve_analytic(self, wavefunction, t):
        eigenvalues, eigenvectors = self.eig(t)
        eigenbasis = np.transpose(eigenvectors)
        inverse_basis = np.linalg.inv(eigenbasis)
        eig_wavefunction = np.dot(eigenbasis, wavefunction)
        time_evolved =  np.array([y*np.exp((-1j*t*eigenvalues[x])/(constants.hbar)) for x,y in enumerate(eig_wavefunction)])
        return np.dot(inverse_basis, time_evolved)
    
    def trotter_evolve(self, wavefunction, t):
        flag = False
        for i in ts:
            if t == i:
                flag = True
        if flag == False:
            return "t value must be within the ts array"
        trotter_evolved = list(np.zeros(len(ts)))
        for i, j in enumerate(ts):
            time_operator = linalg.expm((-1j/constants.hbar)*(self.matrix(j))*dt)
            if i == 0:
                trotter_evolved[i] = np.dot(time_operator, wavefunction)
            else:
                trotter_evolved[i] = np.dot(time_operator, trotter_evolved[i-1])
        return trotter_evolved

class System:
    def __init__(self, initial_wavefunction, hamiltonian):
        self.initial_state = (1/np.sqrt(sum(np.array(initial_wavefunction)**2)))*np.array(initial_wavefunction)
        self.hamiltonian = hamiltonian

    def state(self, t):
        if self.hamiltonian.delta_func == constant_fn:
            return self.hamiltonian.evolve_analytic(self.initial_state, t)
        else:
            index = list(ts).index(t)
            return self.hamiltonian.trotter_evolve(self.initial_state, t)[index]

    def probability(self, desired_state, evolved_state):
        state = (1/np.sqrt(sum(np.array(desired_state)**2)))*np.array(desired_state)
        inner_product = np.dot(np.conjugate(state), evolved_state)
        return np.real(np.conjugate(inner_product)*inner_product)
    
    def graph_probability(self, states, v_func, v_max):
        normalised_states = np.array(list(map(lambda x: (1/np.sqrt(sum(np.array(x)**2)))*np.array(x), states)))
        if self.hamiltonian.delta_func == constant_fn:
            fig, axs = plt.subplots(len(normalised_states))
            for i in range(len(normalised_states)):
                for r, q in enumerate(qs):
                    self.hamiltonian.v_update(v_func, v_max, q)
                    axs[i].plot(ts, np.array([self.probability(normalised_states[i], self.state(t)) for t in ts]), color = colormap[r])
            plt.show()
        else:
            fig, axs = plt.subplots(len(normalised_states)+1)
            axs[0].plot(ts, np.array([self.hamiltonian.delta_func(t, self.hamiltonian.delta_max) for t in ts]), 'gold')
            for i in range(len(normalised_states)):
                for r, q in enumerate(qs):
                    self.hamiltonian.v_update(v_func, v_max, q)
                    trotter_states = self.hamiltonian.trotter_evolve(self.initial_state, ts[-1])
                    axs[i+1].plot(ts, np.array([self.probability(normalised_states[i], trotter_states[r]) for r, t in enumerate(ts)]), color = colormap[r])
            plt.show()


h1 = Hamiltonian(2*np.pi*10**6, constant_fn, 0, 0, 2*np.pi*10**8)
h2 = Hamiltonian(2*np.pi*10**6, t_fn_1, 2*np.pi*50*10**6, 0, 2*np.pi*10**8)


# system_1 = System([1,0,0,0], h1)
# system_1.graph_probability([[1,0,0,0], [0,1,1,0]], q_fn_1, 2*np.pi*10**8)


system_2 = System([1,0,0,0], h2)
system_2.graph_probability([[1,0,0,0], [0,1,1,0]], q_fn_1, 2*np.pi*10**8)