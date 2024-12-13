import numpy as np
import matplotlib.pyplot as plt
from scipy import constants, linalg
import matplotlib as mpl
import time
import inspect

# sets some initial parameters globally for plotting graphs
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['axes.labelpad'] = 5
mpl.rc('font', family='palatino linotype', size=15)


def timer(func):
    def wrapper(*args):
        start = time.time()
        func_output = func(*args)
        end = time.time()
        return func_output, end - start
    return wrapper


# sets up a variety of functions that take as an input the linear space of t and v values.
# t functions are to produce a varying detuning to alter the Hamiltonian's value as a function of time
# v functions are aim to produce a spread of plots of differing V values that look clean and smooth.
a = 1.2
constant_fn = lambda t, max: max + 0*t
t_fn_1 = lambda t, max: 8*max*((t-(max/2))/max)**3
t_fn_2 = lambda t, max: 4*max*((t-(max/2))/max)**2
t_fn_3 = lambda t, max: max*((2/max)*(t-(max/2)))**5
t_fn_4 = lambda t, max: max*((2/max)*(t-(max/2)))
t_sine_1 = lambda t, max: max*np.sin(2*np.pi*t/max)
v_fn_1 = lambda v, max: max*10**(-(1/a)*np.log(v+1))
v_fn_2 = lambda v, max: max*v/20

# Hamiltonian object, which has methods constructing a 4x4 matrix representation, and giving its eigenvalues and eigenvectors.
class Hamiltonian:
    def __init__(self, omega, delta_func, delta_max, phi, v):
        self.omega = omega
        self.delta_func = delta_func
        self.delta_max = delta_max
        self.phi = phi
        self.v = v

    def v_update(self, v_func, v_max, v):
        self.v = v_func(v, v_max)

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

# Object that contains an instance of a whole system, with an initial state given and a Hamiltonian for the system.
# Can compute the state at any time given in ts array, as well as the probability of measurement collapsing said state into any other given state.
# Can solve this analytically if time independent Hamiltonian and via Trotter-Suzuki decomposition if not.
# Also contains graphing functions of the probability of the system being in any given state through ts array.
class System:
    def __init__(self, initial_wavefunction, hamiltonian, n_t=400, t_max=1e-6):
        self.n_t = n_t
        self.t_max = t_max
        self.dt = t_max/n_t
        self.ts = np.arange(0, self.t_max, self.dt)
        self.initial_state = (1/np.sqrt(sum(np.array(initial_wavefunction)**2)))*np.array(initial_wavefunction)
        self.hamiltonian = hamiltonian

    def n_t_update(self, n_t):
        self.n_t = n_t
        self.dt = self.t_max/self.n_t
        self.ts = np.arange(0, self.t_max, self.dt)

    def t_max_update(self, t_max):
        self.t_max = t_max
        self.dt = self.t_max/self.n_t
        self.ts = np.arange(0, self.t_max, self.dt)

    def hamiltonian_update(self, hamiltonian):
        self.hamiltonian = hamiltonian

    # solves analytically for the state at any time t. Only possible for H constant in time. Returns single value
    def evolve_analytic(self, wavefunction, t):
        eigenvalues, eigenvectors = self.hamiltonian.eig(t)
        eigenbasis = np.transpose(eigenvectors)
        inverse_basis = np.linalg.inv(eigenbasis)
        eig_wavefunction = np.dot(eigenbasis, wavefunction)
        time_evolved =  np.array([y*np.exp((-1j*t*eigenvalues[x])/(constants.hbar)) for x,y in enumerate(eig_wavefunction)])
        return np.dot(inverse_basis, time_evolved)

    # solves numerically for the state at any time t within ts array for any given H. Returns array of every value through ts
    def trotter_evolve(self, wavefunction):
        trotter_evolved = list(np.zeros(len(self.ts)))
        for i, j in enumerate(self.ts):
            time_operator = linalg.expm((-1j/constants.hbar)*(self.hamiltonian.matrix(j))*self.dt)
            if i == 0:
                trotter_evolved[i] = np.dot(time_operator, wavefunction)
            else:
                trotter_evolved[i] = np.dot(time_operator, trotter_evolved[i-1])
        return trotter_evolved

    # returns state at a time t. For time independent H uses analytical solution unless specified otherwise with "method" parameter.
    # if using numerical method input t is required to be within ts array.
    @timer
    def state(self, t, *method):
        if (self.hamiltonian.delta_func != constant_fn) or (method == "numerical"):
            flag = False
            for i in self.ts:
                if t == i:
                    flag = True
            if flag == False:
                return "t value must be within the ts array"
            index = list(self.ts).index(t)
            return self.trotter_evolve(self.initial_state)[index] 
        else:
            return self.evolve_analytic(self.initial_state, t)

    def probability(self, desired_state, evolved_state):
        state = (1/np.sqrt(sum(np.array(desired_state)**2)))*np.array(desired_state)
        inner_product = np.dot(np.conjugate(state), evolved_state)
        return np.real(np.conjugate(inner_product)*inner_product)
    
    def graph_probability(self, states, v_func, v_max, n_v = 50):
        vs = np.linspace(0, 35, n_v) # linear range of inputs for a function to output desired V values to plot
        colormap = mpl.colormaps["cool"].resampled(n_v)(range(n_v))   # defines colour of each different V value plot
        normalised_states = np.array(list(map(lambda x: (1/np.sqrt(sum(np.array(x)**2)))*np.array(x), states)))
        if self.hamiltonian.delta_func == constant_fn:
            fig, axs = plt.subplots(len(normalised_states), sharex=True)
            fig.subplots_adjust(hspace=0.1)
            fig.set_size_inches(8, 3*len(normalised_states))
            for i in range(len(normalised_states)):
                for r, v in enumerate(vs):
                    self.hamiltonian.v_update(v_func, v_max, v)
                    axs[i].plot(self.ts*10**6, [self.probability(normalised_states[i], self.state(t)) for t in self.ts], color = colormap[r])
                self.hamiltonian.v_update(constant_fn, v_max, 0)
                axs[i].plot(self.ts*10**6, [self.probability(normalised_states[i], self.state(t)) for t in self.ts], color = "#10a610", linewidth=2)
                axs[i].set_yticks(np.arange(0, 1.2, 0.2), [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                axs[i].tick_params(axis='y', labelsize=15)
                axs[i].set_ylabel(f"P($\\psi$ = {str(states[i])})")
                axs[i].tick_params(axis='y', labelsize=15)
            plt.xlabel("t ($\\mu s$)")
            plt.show()
        else:
            fig, axs = plt.subplots(len(normalised_states)+1, sharex=True)
            fig.subplots_adjust(hspace=0.1)
            fig.set_size_inches(8, 2*(len(normalised_states)+1))
            axs[0].plot(self.ts*10**6, np.array([self.hamiltonian.delta_func(t, self.hamiltonian.delta_max) for t in self.ts])*10**(-8), color='#ffc000', linewidth=1.5)
            axs[0].set_ylabel("$\\Delta$ ($10^8$ Hz)")
            axs[0].tick_params(axis='y', labelsize=15)
            axs[0].grid(visible=True, alpha=0.5)
            axs[0].grid(which='both', axis='x', alpha=0.05)
            for i in range(len(normalised_states)):
                for r, v in enumerate(v):
                    self.hamiltonian.v_update(v_func, v_max, v)
                    trotter_states = self.trotter_evolve(self.initial_state)
                    axs[i+1].plot(self.ts*10**6, [self.probability(normalised_states[i], trotter_states[r]) for r, t in enumerate(self.ts)], color = colormap[r])
                self.hamiltonian.v_update(constant_fn, v_max, 0)
                trotter_states = self.trotter_evolve(self.initial_state)
                axs[i+1].plot(self.ts*10**6, [self.probability(normalised_states[i], trotter_states[r]) for r, t in enumerate(self.ts)], color = "#00852a", linewidth=2)
                axs[i+1].set_yticks(np.arange(0, 1.2, 0.2), [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                axs[i+1].set_ylabel(f"P($\\psi$ = {str(states[i])})")
                axs[i+1].tick_params(axis='y', labelsize=15)
            axs[len(normalised_states)].tick_params(axis='x', labelrotation=35, labelsize=15)
            plt.xlabel("t ($\\mu s$)")
            plt.show()

# Instantiates two cases of Hamiltonian
h_time_independent = Hamiltonian(2*np.pi*10**6, constant_fn, 0, 0, 2*np.pi*10**8)
h_time_dependent = Hamiltonian(2*np.pi*10**6, t_fn_1, 2*np.pi*50*10**6, 0, 2*np.pi*10**8)

# separate function that graphically compares the numerical and analytical methods for a time independent Hamiltonian, and an accompanying residual plot.
def compare_numerical_analytic(initial_wavefunction, n_v):
    colormap_residuals = mpl.colormaps["winter"].resampled(n_v)(range(n_v)) # defines colour for residuals plot
    compare_system = System(initial_wavefunction, h_time_independent)
    fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 2]})
    fig.subplots_adjust(hspace=0.1)
    fig.set_size_inches(10, 7)
    for r, v in enumerate(v):
        compare_system.hamiltonian.v_update(v_fn_1, 2*np.pi*10**8, v)
        analytic_probabilities = np.array([compare_system.probability(initial_wavefunction, x) for x in [compare_system.state(t) for t in compare_system.ts]])
        numerical_probabilities = np.array([compare_system.probability(initial_wavefunction, x) for x in compare_system.trotter_evolve(initial_wavefunction)])
        residuals = analytic_probabilities - numerical_probabilities
        axs[0].plot(compare_system.ts*10**6, analytic_probabilities, color='b', alpha=0.5, label="analytic")
        axs[0].plot(compare_system.ts*10**6, numerical_probabilities, color='r', alpha=0.25, label="numeric")
        axs[1].plot(compare_system.ts*10**6, residuals*10**2, 'o', markersize=0.5, color=colormap_residuals[::-1][r])
    axs[1].tick_params(axis='x', labelrotation=35, labelsize=15)
    axs[0].tick_params(axis='y', labelsize=15)
    axs[1].tick_params(axis='y', labelsize=15)
    axs[0].set_ylabel("P{$\\psi$(t=0)}")
    axs[1].set_ylabel("Residuals ($10^{-2}$)")
    axs[0].legend(["analytic", "numeric"], loc='best')
    plt.xlabel("t ($\\mu s$)")
    # plt.show()

def timer_graph(system, method, *args):
    n_array = np.arange(1, 201000, 100)
    t_array = []
    for n in n_array:
        system.n_t_update(n)
        t_array.append(method(*args)[1])
    plt.plot(n_array, t_array)
    plt.show()
    return method(*args)[0]


def main():

    # system_1 = System([1,0,0,0], h_time_independent)
    # system_1.graph_probability([[1,0,0,0], [0,1,1,0]], v_fn_1, 2*np.pi*10**8)

    system_2 = System([1,0,0,0], h_time_dependent)
    # system_2.graph_probability([[1,0,0,0], [0,1,1,0]], v_fn_1, 2*np.pi*10**8)

    # compare_numerical_analytic([1,0,0,0])
    
    # system_2.n_t_update(625)
    # print(sum([system_2.state(system_2.ts[-1], "numerical")[1] for i in range(200)])/200)
    # timer_graph(system_2, system_2.state, system_2.ts[-1], "numerical")
    # timer_graph(system_1, system_1.state, system_1.ts[-1])


if __name__ == "__main__":
    main()