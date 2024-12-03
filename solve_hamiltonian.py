import numpy as np
import matplotlib.pyplot as plt
from scipy import constants, linalg

plt.rcParams['lines.linewidth'] = 1

n = 400
ts = np.linspace(0, 1e-6, n)
dt = 1e-6/n
qs = np.linspace(0, 20, 40)


a = 1.2
constant_fn = lambda t, max: max + 0*t
t_fn_1 = lambda t, max: 8*max*((t-(ts[-1]/2))/ts[-1])**3
t_fn_2 = lambda t, max: 4*max*((t-(ts[-1]/2))/ts[-1])**2
t_fn_3 = lambda t, max: max*((2/ts[-1])*(t-(ts[-1]/2)))**5
t_fn_4 = lambda t, max: max*((2/ts[-1])*(t-(ts[-1]/2)))
t_sine_1 = lambda t, max: max*np.sin(2*np.pi*t/ts[-1])
q_fn_1 = lambda q, max: max*10**(-(1/a)*np.log(q+1))


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
        index = list(ts).index(t)
        trotter_evolved = np.zeros(len(ts) + 1)
        trotter_evolved[0] = wavefunction
        for i, t in enumerate(ts):
            time_operator = linalg.expm((-1j/constants.hbar)*(self.matrix(t))*dt)
            trotter_evolved[i+1] = np.dot(time_operator, trotter_evolved[i])
        return trotter_evolved[index]

class System:
    def __init__(self, initial_wavefunction, hamiltonian):
        self.initial_state = (1/np.sqrt(sum(np.array(initial_wavefunction)**2)))*np.array(initial_wavefunction)
        self.hamiltonian = hamiltonian

    def state(self, t):
        return self.hamiltonian.evolve(self.initial_state, t)

    def probability(self, desired_state, t):
        state = (1/np.sqrt(sum(np.array(desired_state)**2)))*np.array(desired_state)
        inner_product = np.dot(np.conjugate(state), self.state(t))
        return np.real(np.conjugate(inner_product)*inner_product)
    
    def graph_probability(self, states, v_func, v_max):
        normalised_states = np.array(list(map(lambda x: (1/np.sqrt(sum(np.array(x)**2)))*np.array(x), states)))
        if self.hamiltonian.delta_func == constant_fn:
            fig, axs = plt.subplots(len(normalised_states))
            for i in range(len(normalised_states)):
                axs[i].set_ylim(0, 1)
                axs[i].set_xlim(ts[0], ts[-1])
                self.hamiltonian.v_update(v_func, 1e-9, 0)
                axs[i].plot(ts, np.array([self.probability(normalised_states[i], t) for t in ts]), 'b', alpha = 1)
                self.hamiltonian.v_update(v_func, v_max, 0)
                axs[i].plot(ts, np.array([self.probability(normalised_states[i], t) for t in ts]), 'black', alpha = 1)
                for r, q in enumerate(qs):
                    self.hamiltonian.v_update(v_func, v_max, q)
                    axs[i].plot(ts, np.array([self.probability(normalised_states[i], t) for t in ts]), 'r', alpha = (1/2)*(1-(r/len(qs))))
            plt.show()
        else:
            fig, axs = plt.subplots(len(normalised_states)+1)
            axs[0].plot(ts, np.array([self.hamiltonian.delta_func(t, self.hamiltonian.delta_max) for t in ts]), 'gold')
            for i in range(len(normalised_states)):
                self.hamiltonian.v_update(v_func, 0, 0)
                p1 = []
                wavefunction = self.initial_state
                for t in ts:
                    p1.append(np.real(np.conjugate(np.dot(np.conjugate(normalised_states[i]), wavefunction))*np.dot(np.conjugate(normalised_states[i]), wavefunction)))
                    time_operator = linalg.expm((-1j/constants.hbar)*(self.hamiltonian.matrix(t))*dt)
                    wavefunction = np.dot(time_operator, wavefunction)
                axs[i+1].plot(ts, p1, 'b', alpha = 1)
                for r, q in enumerate(qs):
                    self.hamiltonian.v_update(v_func, v_max, q)
                    p2 = []
                    wavefunction = self.initial_state
                    for t in ts:
                        p2.append(np.real(np.conjugate(np.dot(np.conjugate(normalised_states[i]), wavefunction))*np.dot(np.conjugate(normalised_states[i]), wavefunction)))
                        time_operator = linalg.expm((-1j/constants.hbar)*(self.hamiltonian.matrix(t))*dt)
                        wavefunction = np.dot(time_operator, wavefunction)
                    axs[i+1].plot(ts, p2, 'r', alpha = (1/2)*(1-(r/len(qs))))
                self.hamiltonian.v_update(v_func, v_max, 0)
                p1 = []
                wavefunction = self.initial_state
                for t in ts:
                    p1.append(np.real(np.conjugate(np.dot(np.conjugate(normalised_states[i]), wavefunction))*np.dot(np.conjugate(normalised_states[i]), wavefunction)))
                    time_operator = linalg.expm((-1j/constants.hbar)*(self.hamiltonian.matrix(t))*dt)
                    wavefunction = np.dot(time_operator, wavefunction)
                axs[i+1].plot(ts, p1, 'black', alpha = 1)
            plt.show()


h1 = Hamiltonian(2*np.pi*10**6, constant_fn, 0, 0, 2*np.pi*10**8)
h2 = Hamiltonian(2*np.pi*10**6, t_fn_1, 2*np.pi*50*10**6, 0, 2*np.pi*10**8)


# system_1 = System([1,0,0,0], h1)
# system_1.graph_probability([[1,0,0,0], [0,1,1,0]], q_fn_1, 2*np.pi*10**8)


# system_2 = System([1,0,0,0], h2)
# system_2.graph_probability([[1,0,0,0], [0,1,1,0]], q_fn_1, 2*np.pi*10**8)