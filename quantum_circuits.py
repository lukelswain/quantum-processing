import itertools as ite  # noqa: D100, INP001

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants, linalg

plt.rc("text", usetex=True)
plt.rc("font", family="times")
plt.rc("text.latex", preamble=r"\usepackage{physics}")
plt.rc("xtick", labelsize="large")
plt.rc("ytick", labelsize="large")

class System:
    """Creates a class for of the quantum system itself.

    Includes properties of physical system such as the number of qubits and their positions in space, the laser parameters,
    the decoherence parameters due to interaction with the external environment; includes also the number timesteps.

    Generates a Hamiltonian matrix and Lindblad operator for the system due to the physical properties.S
    """

    def __init__(self, qubit_array:tuple=((0, 0), (2.5,0), (1.25,0)), n_steps:int = 1000) -> None:
        """Define all properties of the system."""
        self.n_steps = n_steps
        self.qubit_array = np.array(qubit_array)
        self.n_qubits = len(qubit_array)
        self.hyperfine_detuning = [(lambda t: 0*t) for qubit in qubit_array]
        self.hyperfine_rabi = [(lambda t: 0*t) for qubit in qubit_array]
        self.hyperfine_phase = [(lambda t: 0*t) for qubit in qubit_array]
        self.rydberg_detuning = lambda t: 0*t
        self.rydberg_rabi = lambda t: 0*t
        self.rydberg_phase = lambda t: 0*t
        self.gamma_dephasing = 1*np.pi*10**4
        self.gamma_relaxation = 1*np.pi*10**4

    def distance(self, qubit_one:int, qubit_two:int) -> float:
        """Calculate distances between qubit pairs."""
        displacement = self.qubit_array[qubit_one] - self.qubit_array[qubit_two]
        return np.sqrt(np.dot(displacement, displacement))

    def rabi(self, qubit_no:int, t:float) -> np.ndarray:
        """Create 3x3 matrix of Rabi frequency for a single qubit."""
        return np.array([[0, self.hyperfine_rabi[self.n_qubits-1-qubit_no](t)*np.exp(-1j*self.hyperfine_phase[self.n_qubits-1-qubit_no](t)), 0],
                         [self.hyperfine_rabi[self.n_qubits-1-qubit_no](t)*np.exp(1j*self.hyperfine_phase[self.n_qubits-1-qubit_no](t)), 0, self.rydberg_rabi(t)*np.exp(-1j*self.rydberg_phase(t))],  # noqa: E501
                         [0, self.rydberg_rabi(t)*np.exp(1j*self.rydberg_phase(t)), 0]])

    def detuning(self, qubit_no:int, t:float) -> np.ndarray:
        """Create 3x3 matrix of detuning for a single qubit."""
        return np.array([[self.hyperfine_detuning[self.n_qubits-1-qubit_no](t), 0, 0],
                         [0, -self.hyperfine_detuning[self.n_qubits-1-qubit_no](t)+self.rydberg_detuning(t), 0],
                         [0, 0, -self.rydberg_detuning(t)]])

    def rydberg(self, qubit_pair:int) -> np.ndarray:
        """Create 3x3 matrix of the Rydberg interaction between a pair of qubits."""
        qubit_one, qubit_two = qubit_pair[0], qubit_pair[1]
        return np.array([[0, 0, 0],
                         [0, 0, 0],
                         [0, 0, ((5420503*10**6)/(2*np.pi))/(self.distance(self.n_qubits-1-qubit_one, self.n_qubits-1-qubit_two)**6)]])

    def rabi_generator(self, qubit_no:int, t:float) -> np.ndarray:
        """Generate the contribution to the total Hamiltonian of the Rabi frequency terms by means of a tensor product sum."""
        i = 0
        identity = np.eye(3)
        product = identity
        while i < self.n_qubits:
            if i == 0:
                if i == qubit_no:
                    product = self.rabi(i, t)
                    i += 1
                else:
                    i += 1
            elif i == qubit_no:
                product = np.tensordot(self.rabi(i, t), product, axes=0)
                i += 1
            else:
                product = np.tensordot(identity, product, axes=0)
                i += 1
        return product

    def detuning_generator(self, qubit_no:int, t:float) -> np.ndarray:
        """Generate the contribution to the total Hamiltonian of the detuning terms by means of a tensor product sum."""
        i = 0
        identity = np.eye(3)
        product = identity
        while i < self.n_qubits:
            if i == 0:
                if i == qubit_no:
                    product = self.detuning(i, t)
                    i += 1
                else:
                    i += 1
            elif i == qubit_no:
                product = np.tensordot(self.detuning(i, t), product, axes=0)
                i += 1
            else:
                product = np.tensordot(identity, product, axes=0)
                i += 1
        return product

    def rydberg_generator(self, qubit_pair:np.ndarray) -> np.ndarray:
        """Generate the contribution to the total Hamiltonian of the Rydberg interaction terms between a qubit pair."""
        i = 0
        qubit_one, qubit_two = qubit_pair[0], qubit_pair[1]
        identity = np.eye(3)
        product = identity
        while i < self.n_qubits:
            if i == 0:
                if i in (qubit_one, qubit_two):
                    product = np.sqrt(self.rydberg(qubit_pair))
                    i += 1
                else:
                    i += 1
            elif i in (qubit_one, qubit_two):
                product = np.tensordot(np.sqrt(self.rydberg(qubit_pair)), product, axes=0)
                i += 1
            else:
                product = np.tensordot(identity, product, axes=0)
                i += 1
        return product

    def total_hamiltonian(self, t: float) -> np.ndarray:
        """Generate Hamiltonian matrix from the values given in class at a given time, t."""
        h = np.zeros((3, 3))
        for i in range(self.n_qubits-1):  # noqa: B007
            h = np.tensordot(h, np.zeros((3,3)), axes=0)
        self.t = t

        for i in range(self.n_qubits):
            h = h + self.rabi_generator(i, self.t) + self.detuning_generator(i, self.t)
        for i in ite.combinations(range(self.n_qubits), 2):
            h = h + self.rydberg_generator(i)

        elements = np.arange(len(np.shape(h)))
        evens = []
        odds = []
        for i in range(len(elements)):
            if i%2 == 0:
                evens.append(elements[i])
            else:
                odds.append(elements[i])
        elements = evens + odds
        return (constants.hbar/2)*h.transpose(elements).reshape((3**self.n_qubits, 3**self.n_qubits))

    def gamma_generator(self, qubit_no:int) -> np.ndarray:
            """Generate gamma matrix."""
            gamma_matrix = np.array([[self.gamma_relaxation, -self.gamma_dephasing, -self.gamma_dephasing],
                                     [-self.gamma_dephasing, -self.gamma_relaxation, -self.gamma_dephasing],
                                     [-self.gamma_dephasing, -self.gamma_dephasing, -0*self.gamma_relaxation]])
            i = 0
            identity = np.eye(3)
            product = identity
            while i < self.n_qubits:
                if i == 0:
                    if i == qubit_no:
                        product = gamma_matrix
                        i += 1
                    else:
                        i += 1
                elif i == qubit_no:
                    product = np.tensordot(gamma_matrix, product, axes=0)
                    i += 1
                else:
                    product = np.tensordot(identity, product, axes=0)
                    i += 1
            return product

    def lindblad_operator(self, t:float, rho:np.ndarray) -> np.ndarray:
        """Generate lindblad operator from Hamiltonian at time t, system density matrix, and gamma matrix."""
        commutator_term = (-1j/constants.hbar)*(np.matmul(self.total_hamiltonian(t), rho) - np.matmul(rho, self.total_hamiltonian(t)))

        decay_term = np.zeros((3, 3))
        for i in range(self.n_qubits-1):  # noqa: B007
            decay_term = np.tensordot(decay_term, np.zeros((3,3)), axes=0)

        elements = np.arange(len(np.shape(decay_term)))
        evens = []
        odds = []
        for i in range(len(elements)):
            if i%2 == 0:
                evens.append(elements[i])
            else:
                odds.append(elements[i])
        elements = evens + odds

        decay_term = decay_term.transpose(elements).reshape((3**self.n_qubits, 3**self.n_qubits))
        for i in range(self.n_qubits):
            rho_transposed = np.empty(np.shape(rho))
            rho_transposed = rho_transposed + rho
            for j in range(3**self.n_qubits):
                if (self.gamma_generator(i).transpose(elements).reshape((3**self.n_qubits, 3**self.n_qubits))[j][j] > 0):
                    rho_transposed[j][j] = rho_transposed[j+(3**i)][j+(3**i)]
            decay_term = decay_term + (self.gamma_generator(i).transpose(elements).reshape((3**self.n_qubits, 3**self.n_qubits))*rho_transposed)
        return (commutator_term + decay_term)

class Circuit:
    """Create a circuit class that takes the system and an initial state as input, with methods that evolve this system."""

    def __init__(self, system:System, initial_state:np.ndarray) -> None:
        """Declare the aforementioned inputs for the Circuit instantiation."""
        self.system = system
        self.states = np.zeros(3**self.system.n_qubits)
        if len(initial_state) != 3**self.system.n_qubits:
            self.states = np.zeros(3**self.system.n_qubits)
            for count,value in enumerate([0, 1, 3, 4, 9, 10, 12, 13]):
                self.states[value] = initial_state[count]
        else:
            self.states = np.array(initial_state)
        self.states = (1/np.sqrt(sum((self.states)**2)))*self.states.reshape(1, 3**self.system.n_qubits)

    def evolve_trotter(self, evolve_time:float) -> np.ndarray:
        """Evolve the state via Trotter-Suzuki Decomposition."""
        trotter_evolved = list(np.zeros(len(self.hamiltonian)))
        for i in range(len(self.hamiltonian)):
            time_operator = linalg.expm((-1j/constants.hbar)*(self.hamiltonian[i])*((evolve_time*10**-6)/self.system.n_steps))
            if i == 0:
                trotter_evolved[i] = np.dot(time_operator, self.states[-1])
            else:
                trotter_evolved[i] = np.dot(time_operator, trotter_evolved[i-1])
        return np.array(trotter_evolved)

    def evolve_rk4(self, initial_state:np.ndarray, evolve_time:float) -> np.ndarray:
        """Evolve the state via RK4."""
        ts = np.arange(0, evolve_time, (evolve_time)/self.system.n_steps)
        rho_array = [initial_state]

        for t in ts:
            rho = np.array(rho_array[-1])
            step = (evolve_time/self.system.n_steps)*10**-6
            k1 = step*self.system.lindblad_operator(t, rho)
            k2 = step*(1/2)*(self.system.lindblad_operator(t, rho + (1/2)*k1) + self.system.lindblad_operator(t+(step*10**6), rho + (1/2)*k1))
            k3 = step*(1/2)*(self.system.lindblad_operator(t, rho + (1/2)*k2) + self.system.lindblad_operator(t+(step*10**6), rho + (1/2)*k2))
            k4 = step*self.system.lindblad_operator(t+(step*10**6), rho + k3)
            rho_array.append((rho + (1/6)*(k1 + 2*k2 + 2*k3 + k4)).tolist())

        return rho_array

    def output(self) -> np.ndarray:
        """Return most recent state of system."""
        final_state = self.states[-1]
        length_with_intermediate_state = 27
        if len(final_state) == length_with_intermediate_state:
            output = np.delete(np.round(self.states[-1], 3), [2,5,6,7,8,11,14,15,16,17,18,19,20,21,22,23,24,25,26])
        else:
            output = np.round(self.states[-1], 3)
        self.states[-1] = final_state
        return output

    def density_operator(self, state:np.ndarray) -> np.ndarray:
        """Generate density operator for a given state of system."""
        length_with_intermediate_state = 27
        reduced_state = np.delete(state, [2, 5, 6, 7, 8, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]) if len(state) == length_with_intermediate_state else state  # noqa: E501
        return np.tensordot(reduced_state, np.conjugate(reduced_state), axes=0)

    def trace_out(self, total_rho:np.ndarray, qubit:int) -> np.ndarray:
        """Trace out an individual qubit from a given density operator."""
        elements_total = np.arange(int(2*np.log2(len(total_rho))))

        # creates array containing numbered axes representing the indices for each qubit whose information is contained in the input
        # density matrix

        placeholder_matrix_total = np.eye(2,2)
        for i in range(int(np.log2(len(total_rho))-1)):  # noqa: B007
            placeholder_matrix_total = np.tensordot(placeholder_matrix_total, np.eye(2,2), axes=0)
        shape_total = np.shape(placeholder_matrix_total)

        # finds the tensor product structure of the density matrix; each qubit has a 2x2 matrix, so for 3 qubits, shape = (2,2,2,2,2,2),
        # which is the result of the repeated tensor product of three 2x2 matrices. I use the identity, but the value doesn't matter,
        # it's only the shape we want.

        evens = []
        odds = []
        for i in range(len(elements_total)):
            if i%2 == 0:
                evens.append(elements_total[i])
            else:
                odds.append(elements_total[i])
        permutation = evens + odds

        # finds the permutation of the axes that orders them so that when the reshape function in numpy merges axes of the above
        # tensor product shape into a standard matrix shape (that is, 2**n_qubits by 2**n_qubits, where n_qubits is the number
        # of qubits whose information is contained in the input density matrix), the indices are grouped, and thus the elements
        # are organised in the correct way. In the case (2,2,2,2,2,2), the indices for the tensors (say they are all I(ij) for
        # simplicity) are (i,j,k,l,m,n). To correctly group these indices into a matrix with two indices, I(ab), we must merge
        # the first and second indices from each component, ([i,k,m], [j,l,n]). But numpy's reshape function applied to our
        # array would group the indices as follows: ([i,j,k], [l,m,n]). The permutation is therefore [0,2,4,1,3,5].

        qubits_removed = int(self.system.n_qubits - np.log2(len(total_rho)))

        # finds the number of qubits that have already been traced out from the total system to result in the input matrix.

        inverse_permutation = np.argsort(permutation)
        rho = total_rho.reshape(shape_total).transpose(inverse_permutation)

        # since our input density matrix is already of the form ([j,k,m], [j,l,n]), once we reshape it by splitting these axes
        # we want to return to our original configuration, so we apply the inverse permutation to the one discussed above. We
        # do this because the form (i,j,k,l,m,n) is more workable for tracing out a qubit; in this configuration the indices
        # for each qubit are separated neatly from each other in pairs. The reason for this will become clearer in the next
        # sections. Recall that qubit one has indices (m,n), then we tensor it with qubit two (k,l) to get (k,l,m,n), etc.

        elements_total[[2*qubit, -2]] = elements_total[[-2, 2*qubit]]
        elements_total[[(2*qubit)+1, -1]] = elements_total[[-1, (2*qubit)+1]]
        rho = rho.transpose(elements_total)

        # Here take the indices of the qubit we want to trace out and move it to the back; if in our example we want to trace out
        # qubit 3, we have (i,j,k,l,m,n) --> (m,n,k,l,i,j). The indices are still grouped in their qubit pairs, but the pair
        # belonging to the qubit we want to trace out, we move to the back: (1,2,3) --> (3,2,1)

        rho = rho.reshape((2**(self.system.n_qubits-1-qubits_removed)),(2**(self.system.n_qubits-1-qubits_removed)), 2,2)
        reduced_rho = np.zeros((2**(self.system.n_qubits-1-qubits_removed),2**(self.system.n_qubits-1-qubits_removed)), dtype = "complex_")
        for i in range(len(reduced_rho)):
            for j in range(len(reduced_rho[0])):
                reduced_rho[i][j] = np.trace(rho[i][j])


        # Now we group the indices not being traced over into two dimensions, so that we form a matrix of 2x2 matrices. This is
        # to make it easier to formulate the above repeated application of the trace on each of these 2x2 matrices. The indices
        # being summed over to perform the trace are the matrices at the back of the structure, which pertain to the qubit we
        # are trying to trace over. We construct a new reduced density matrix (of the correct shape) and fill its entries
        # accordingly with the results of each trace. We have now traced out this qubit.

        elements_reduced = np.arange(int(2*np.log2(len(reduced_rho))))
        placeholder_matrix_reduced = np.eye(2,2)
        for i in range(int(np.log2(len(reduced_rho))-1)):  # noqa: B007
            placeholder_matrix_reduced = np.tensordot(placeholder_matrix_reduced, np.eye(2,2), axes=0)
        shape_reduced = np.shape(placeholder_matrix_reduced)
        evens = []
        odds = []
        for i in range(len(elements_reduced)):
            if i%2 == 0:
                evens.append(elements_reduced[i])
            else:
                odds.append(elements_reduced[i])
        permutation = evens + odds
        reduced_rho = reduced_rho.reshape(shape_reduced)

        # Now performs the procedure we implemented at the start in reverse, but for the reduced dimension of the new density operator.
        # Indices are grouped as ([m,n], [k,l]). We first split this grouping to yield (m,n,k,l) according to the new shape
        # variable, which in our example is now (2,2,2,2).

        permuted_axes = np.delete(np.array(elements_total), [-1,-2])
        reorder_permutation = np.argsort(permuted_axes)
        reduced_rho = reduced_rho.transpose(reorder_permutation)

        # We now reorder the qubits so that they are in their original order. We had (1,2,3) --> (3,2,1) --> (3,2). If we later want
        # to trace out qubit two, we would need to keep track of its position as we shuffled around the qubits to move the desired one
        # to the back. This would be annoying for large numbers of qubits and large numbers of traces. It is thus convenient to reorder
        # them, and scale the labels down to reflect their positions: (3,2) --> (2,3) --> (1,2); (m,n,k,l) --> (k,l,m,n) --> (i,j,k,l).

        reduced_rho = reduced_rho.transpose(permutation).reshape((2**(self.system.n_qubits-1-qubits_removed)), (2**(self.system.n_qubits-1-qubits_removed)))  # noqa: E501

        # This applies the aforementioned permutation to group the indices correctly for reshaping in to a 4x4 matrix: (i,j,k,l)
        # --> (i,k,j,l) --> ([i,k], [j,l]).

        return reduced_rho  # noqa: RET504

    def probability(self, state:np.ndarray, compare_state:np.ndarray) -> float:
        """Return probability that upon measurement one state will be found to be in another."""
        if len(state) != 3**self.system.n_qubits:
            total_state = np.zeros(3**self.system.n_qubits)
            for count,value in enumerate([0, 1, 3, 4, 9, 10, 12, 13]):
                total_state[value] = state[count]
        else:
            total_state = state
        if len(compare_state) != 3**self.system.n_qubits:
            total_compare = np.zeros(3**self.system.n_qubits)
            for count,value in enumerate([0, 1, 3, 4, 9, 10, 12, 13]):
                total_compare[value] = compare_state[count]
        else:
            total_compare = compare_state
        compare_normalised = (1/np.sqrt(np.dot((total_compare), np.conjugate(total_compare))))*np.array(total_compare)
        return np.real(np.dot(compare_normalised, np.conjugate(total_state))*np.conjugate(np.dot(compare_normalised, np.conjugate(total_state))))

    def graph_probability(self, compare_states:np.ndarray) -> None:
        """Graph probability of system being measured to be in any number of input states over time."""
        ts_array = np.linspace(0, (1/self.system.n_steps)*len(self.states), len(self.states))
        for compare_state in compare_states:
            plt.plot(ts_array, [self.probability(self.states[i], compare_state) for i in range(len(ts_array))])
        plt.show()

    def graph_bloch(self, qubit:int) -> None:  # noqa: PLR0915
        """Graph the evolution of a single qubit's state on the bloch sphere over time."""
        bloch_vectors = []
        for state in self.states:
            rho = self.density_operator(state)
            j = 1
            qubit_pos = qubit
            tracer_pos = 0
            while j < self.system.n_qubits:
                if (qubit_pos == 0):
                    tracer_pos = 1
                rho = self.trace_out(rho, tracer_pos)
                qubit_pos = qubit_pos - 1
                j = j + 1
            u = 2*np.real(rho[0][1])
            v = 2*np.imag(rho[1][0])
            w = np.real(rho[0][0] - rho[1][1])
            bloch_vectors.append([0, 0, 0, u, v, w])
        X, Y, Z, U, V, W = zip(*bloch_vectors)  # noqa: N806

        theta = np.linspace(0, 2 * np.pi, 100)
        phi = np.linspace(0, np.pi, 50)
        theta, phi = np.meshgrid(theta, phi)
        r = 1
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        axis = np.linspace(-1.5,0,500)
        axis_z = np.linspace(-1.25,0,500)
        zeroes = np.zeros(500)

        figure = plt.figure(figsize=(3,3))
        ax = figure.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, z, cmap="binary", alpha=0.2)
        ax.plot_wireframe(x, y, z, color="gray", alpha=0.25, rstride=8, cstride=8, linewidth=0.8)
        ax.quiver(X[0::25], Y[0::25], Z[0::25], U[0::25], V[0::25], W[0::25], color=mpl.colormaps["cool"](np.linspace(0,1,len(W[0::25]))), arrow_length_ratio=0.05, linewidth=1.5)  # noqa: E501
        ax.quiver(X[0], Y[0], Z[0], U[0], V[0], W[0], arrow_length_ratio=0.05, linewidth=2, color="turquoise")
        ax.quiver(X[-1], Y[-1], Z[-1], U[-1], V[-1], W[-1], arrow_length_ratio=0.05, linewidth=2, color="purple")
        ax.scatter(U, V, W, c = np.arange(len(W)), cmap="cool", s=1)
        ax.scatter(axis, zeroes, zeroes, c="#525250", s=0.07)
        ax.scatter(zeroes, axis, zeroes, c="#525250", s=0.07)
        ax.scatter(zeroes, zeroes, axis_z, c="#525250", s=0.07)
        ax.quiver(0, 0, 0, 1.5, 0, 0, arrow_length_ratio=0.08, linewidth=1, color="#525250")
        ax.quiver(0, 0, 0, 0, 1.5, 0, arrow_length_ratio=0.08, linewidth=1, color="#525250")
        ax.quiver(0, 0, 0, 0, 0, 1.25, arrow_length_ratio=0.08, linewidth=1, color="#525250")
        ax.text(0.2,0,1.3,r"$\ket{0}$", size="x-large")
        ax.text(0.2,0,-1.4,r"$\ket{1}$", size="x-large")
        ax.text(1.3, 0, -0.2, r"$x$", size="large")
        ax.text(0, 1.3, -0.2, r"$y$", size="large")
        ax.text(-0.2, 0, 1.3, r"$z$", size="large")
        ax.text(-1, -1, 1.5, r"(d)", size="x-large")
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_box_aspect((1, 1, 1))
        ax.grid(visible=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        plt.show()

    def measure_qubits(self, qubits:list) -> None:
        """Print to console the probabilities that any of the given qubits are in the states 0 or 1."""
        for qubit in qubits:
            rho = self.density_operator(self.states[-1])
            j = 1
            qubit_pos = qubit
            tracer_pos = 0
            while j < self.system.n_qubits:
                if (qubit_pos == 0):
                    tracer_pos = 1
                rho = self.trace_out(rho, tracer_pos)
                qubit_pos = qubit_pos - 1
                j = j + 1
            eigenvalues, eigenvectors = np.linalg.eig(rho)
            probabilities_separate = [[eigenvectors[i][j]*np.conjugate(eigenvectors[i][j])*eigenvalues[i] for j in range(len(eigenvectors[i]))]for i in range(len(eigenvectors))]  # noqa: E501
            probabilities_final = [probabilities_separate[1][i] + probabilities_separate[0][i] for i in range(len(probabilities_separate))]
            print(f"Qubit {qubit+1}:\nP(|0>) = {np.real(np.round(probabilities_final[0], 2))}\nP(|1>) = {np.real(np.round(probabilities_final[1], 2))}\n")  # noqa: E501, T201

    def rx_gate(self, targets:list, theta:float) -> np.ndarray:
        """Implement RX gate on given target qubits."""
        evolve_time = 1
        self.ts = np.arange(0, evolve_time, (evolve_time)/self.system.n_steps)
        A = theta/(2*np.pi*0.42*evolve_time)  # noqa: N806
        def blackman_window(t:float) -> float:
            return (2*np.pi*10**6)*(((1-0.16)/2)*A - (A/2)*(np.cos((2*np.pi*t)/evolve_time)) + ((A*0.16)/2)*(np.cos((4*np.pi*t)/evolve_time)))

        self.system.hyperfine_detuning = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.hyperfine_rabi = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.hyperfine_phase = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.rydberg_detuning = lambda t: 0*t
        self.system.rydberg_rabi = lambda t: 0*t
        self.system.rydberg_phase = lambda t: 0*t

        for target in targets:
            self.system.hyperfine_rabi[target] = blackman_window

        self.hamiltonian = [self.system.total_hamiltonian(t) for t in self.ts]

        self.states = np.append(self.states, self.evolve_trotter(evolve_time), axis=0)
        return self.states

    def ry_gate(self, targets:list, theta:float) -> np.ndarray:
        """Implement RY gate on given target qubits."""
        evolve_time = 1
        self.ts = np.arange(0, evolve_time, (evolve_time)/self.system.n_steps)
        A = theta/(2*np.pi*0.42*evolve_time)  # noqa: N806
        def blackman_window(t:float) -> float:
            return (2*np.pi*10**6)*(((1-0.16)/2)*A - (A/2)*(np.cos((2*np.pi*t)/evolve_time)) + ((A*0.16)/2)*(np.cos((4*np.pi*t)/evolve_time)))

        self.system.hyperfine_detuning = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.hyperfine_rabi = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.hyperfine_phase = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.rydberg_detuning = lambda t: 0*t
        self.system.rydberg_rabi = lambda t: 0*t
        self.system.rydberg_phase = lambda t: 0*t

        for target in targets:
            self.system.hyperfine_rabi[target] = blackman_window
            self.system.hyperfine_phase[target] = (lambda t: 0*t + (np.pi/2))

        self.hamiltonian = [self.system.total_hamiltonian(t) for t in self.ts]

        self.states = np.append(self.states, self.evolve_trotter(evolve_time), axis=0)
        return self.states

    def rz_gate(self, targets:list, theta:float) -> np.ndarray:
        """Implement RZ gate on given target qubits."""
        evolve_time = 1
        self.ts = np.arange(0, evolve_time, (evolve_time)/self.system.n_steps)
        detuning_ratio = 1000000000
        alpha = 0.16
        A = theta/(2*np.pi*0.42*evolve_time)  # noqa: N806
        def blackman_window(t:float) -> float:
            return (1/np.sqrt(1+detuning_ratio**2))*(2*np.pi*10**6)*(((1-alpha)/2)*A - (A/2)*(np.cos((2*np.pi*t)/evolve_time)) + ((A*alpha)/2)*(np.cos((4*np.pi*t)/evolve_time)))  # noqa: E501

        self.system.hyperfine_detuning = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.hyperfine_rabi = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.hyperfine_phase = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.rydberg_detuning = lambda t: 0*t
        self.system.rydberg_rabi = lambda t: 0*t
        self.system.rydberg_phase = lambda t: 0*t

        for target in targets:
            self.system.hyperfine_rabi[target] = lambda t: blackman_window(t)
            self.system.hyperfine_detuning[target] = lambda t: detuning_ratio*blackman_window(t)

        self.hamiltonian = [self.system.total_hamiltonian(t) for t in self.ts]

        self.states = np.append(self.states, self.evolve_trotter(evolve_time), axis=0)
        return self.states

    def p_gate(self, targets:list, theta:float) -> np.ndarray:
        """Implement Phase gate on given target qubits."""
        self.rz_gate(targets, theta)
        self.states[-1] = self.states[-1]*((np.exp(1j*0.5*theta))**len(targets))
        # evolve_time = 1  # noqa: ERA001
        # self.ts = np.arange(0, evolve_time, (evolve_time)/self.system.n_steps)  # noqa: ERA001
        # y = 2.4  # noqa: ERA001
        # s = (np.pi*2)/((np.sqrt(y**2 + 2)))  # noqa: ERA001
        # sq = np.sqrt((y**2)+1)  # noqa: ERA001
        # sqs = (s/2)*sq  # noqa: ERA001
        # num = -sq*np.cos(sqs) + 1j*y*np.sin(sqs)  # noqa: ERA001
        # denom = sq*np.cos(sqs) + 1j*y*np.sin(sqs)  # noqa: ERA001
        # phase = 2*np.pi-1j*np.log(num/denom)  # noqa: ERA001

        # self.system.hyperfine_detuning = [(lambda t: 0*t) for i in range(self.system.n_qubits)]  # noqa: ERA001
        # self.system.hyperfine_rabi = [(lambda t: 0*t) for i in range(self.system.n_qubits)]  # noqa: ERA001
        # self.system.hyperfine_phase = [(lambda t: 0*t) for i in range(self.system.n_qubits)]  # noqa: ERA001
        # self.system.rydberg_detuning = lambda t: 0*t  # noqa: ERA001
        # self.system.rydberg_rabi = lambda t: 0*t  # noqa: ERA001
        # self.system.rydberg_phase = lambda t: 0*t  # noqa: ERA001

        # for target in targets:
        #     self.system.hyperfine_rabi[target] = lambda t: (s/evolve_time)*10**6  # noqa: ERA001
        #     self.system.hyperfine_detuning[target] = lambda t: y*(s/evolve_time)*10**6  # noqa: ERA001
        #     self.system.hyperfine_phase[target] = lambda t: 0*t  # noqa: ERA001

        # self.hamiltonian = [self.system.total_hamiltonian(t) for t in self.ts]  # noqa: ERA001

        # self.states = np.append(self.states, self.evolve_trotter(evolve_time), axis=0)  # noqa: ERA001

        # for target in targets:
        #     self.system.hyperfine_phase[target] = lambda t: 0*t - phase  # noqa: ERA001

        # self.hamiltonian = [self.system.total_hamiltonian(t) for t in self.ts]  # noqa: ERA001

        # self.states = np.append(self.states, self.evolve_trotter(evolve_time), axis=0)  # noqa: ERA001
        # print(np.pi+np.angle(self.states[-1][0]))  # noqa: ERA001
        return self.states

    def hadamard_gate(self, targets:list) -> np.ndarray:
        """Implement Hadamard gate on given target qubits."""
        evolve_time = 1
        self.ts = np.arange(0, evolve_time, (evolve_time)/self.system.n_steps)
        detuning_ratio = 1
        alpha = 0.16
        A = np.pi/(2*np.pi*0.42*evolve_time)  # noqa: N806
        def blackman_window(t:float) -> float:
            return (1/np.sqrt(1+detuning_ratio**2))*(2*np.pi*10**6)*(((1-alpha)/2)*A - (A/2)*(np.cos((2*np.pi*t)/evolve_time)) + ((A*alpha)/2)*(np.cos((4*np.pi*t)/evolve_time)))  # noqa: E501
        # plt.figure(figsize=(3.8,2))  # noqa: ERA001
        # plt.plot(self.ts, [blackman_window(t)/(5.25*10**6) for t in self.ts], c='orange', linewidth=2.4)  # noqa: ERA001
        # plt.xlabel(r"$t/T$", fontsize='x-large')  # noqa: ERA001
        # plt.ylabel(r"$\Omega/\Omega_{max}$", fontsize='x-large')  # noqa: ERA001
        # plt.text(0.0,0.8, r"(e)",fontsize='x-large')  # noqa: ERA001
        # plt.tight_layout()  # noqa: ERA001
        # plt.show()  # noqa: ERA001

        self.system.hyperfine_detuning = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.hyperfine_rabi = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.hyperfine_phase = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.rydberg_detuning = lambda t: 0*t
        self.system.rydberg_rabi = lambda t: 0*t
        self.system.rydberg_phase = lambda t: 0*t

        for target in targets:
            self.system.hyperfine_rabi[target] = lambda t: blackman_window(t)
            self.system.hyperfine_detuning[target] = lambda t: detuning_ratio*blackman_window(t)

        self.hamiltonian = [self.system.total_hamiltonian(t) for t in self.ts]

        self.states = np.append(self.states, self.evolve_trotter(evolve_time), axis=0)
        self.states[-1] = self.states[-1]*((np.exp(1j*0.5*np.pi))**len(targets))
        return self.states

    def cz_gate(self, target_pair:list, ratio:float = 0.378233) -> np.ndarray:
        """Implement CZ gate on a given qubit pair via Levine gate protocol."""
        initial_array = self.system.qubit_array
        self.system.qubit_array = 100*(self.system.qubit_array+1.25)
        for count,value in enumerate(target_pair):
            self.system.qubit_array[value] = [10, 10+(2.5*(count))]
        evolve_time = 1
        self.ts = np.arange(0, evolve_time, (evolve_time)/self.system.n_steps)
        y = ratio
        s = (np.pi*2)/(np.sqrt(y**2 + 2))
        sq = np.sqrt((y**2)+1)
        sqs = (s/2)*sq
        num = -sq*np.cos(sqs) + 1j*y*np.sin(sqs)
        denom = sq*np.cos(sqs) + 1j*y*np.sin(sqs)
        phase = 2*np.pi-1j*np.log(num/denom)
        phi = 0.7617829

        self.system.hyperfine_detuning = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.hyperfine_rabi = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.hyperfine_phase = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.rydberg_detuning = lambda t: 0*t
        self.system.rydberg_rabi = lambda t: 0*t
        self.system.rydberg_phase = lambda t: 0*t

        self.system.rydberg_rabi = (lambda t: 0*t + (s/evolve_time)*10**6)
        self.system.rydberg_detuning = (lambda t: 0*t + y*(s/evolve_time)*10**6)
        self.system.rydberg_phase = (lambda t: 0*t)
        # fig, ax = plt.subplots(figsize=(4.3,1.5))  # noqa: ERA001
        # ax.plot(self.ts, [self.system.rydberg_phase(t) for t in self.ts], c='orange', linewidth=2)  # noqa: ERA001

        self.hamiltonian = [self.system.total_hamiltonian(t) for t in self.ts]
        self.states = np.append(self.states, self.evolve_trotter(evolve_time), axis=0)

        self.system.rydberg_phase = (lambda t: (0*t - phase))
        # ax.quiver(1, 0, 0, np.real(self.system.rydberg_phase(0)), color='orange', width=0.01, scale_units='xy', scale=1.45, angles='uv', headwidth=4)  # noqa: E501, ERA001
        # ax.quiver(1, np.real(self.system.rydberg_phase(0))/2, 0, np.real(self.system.rydberg_phase(0)), color='orange', width=0.01, scale_units='xy', scale=1.45, angles='uv', headwidth=0)  # noqa: E501, ERA001
        # ax.plot(self.ts+1, [np.real(self.system.rydberg_phase(t)) for t in self.ts], c='orange', linewidth=2)  # noqa: ERA001
        # ax.text(0.9, -2.3, r"$\xi$", fontsize="x-large", color='orange')  # noqa: ERA001
        # ax.set_ylabel(r"$\phi_r$", fontsize="x-large", color='orange')  # noqa: ERA001
        # ax.set_yticks([0,-2,-4])  # noqa: ERA001
        # ax.text(0,-2, r"(d)", fontsize='x-large')  # noqa: ERA001
        # ax2 = ax.twinx()  # noqa: ERA001
        # ax2.set_ylabel(r"$\Delta/\Omega$", fontsize="x-large", color='b')  # noqa: ERA001
        # ax2.plot(2*self.ts, [y for t in self.ts], c='b')  # noqa: ERA001
        # ax2.set_yticks([-1,0,1])  # noqa: ERA001
        # plt.tight_layout()  # noqa: ERA001
        # plt.show()  # noqa: ERA001

        self.hamiltonian = [self.system.total_hamiltonian(t) for t in self.ts]
        self.states = np.append(self.states, self.evolve_trotter(evolve_time), axis=0)
        self.p_gate(list(range(self.system.n_qubits)), -phi)
        self.system.qubit_array = initial_array

        return self.states

    def cz_time_optimal(self, target_pair:list) -> np.ndarray:
        """Implement CZ gate on a given qubit pair via time-optimal gate protocol."""
        initial_array = self.system.qubit_array
        self.system.qubit_array = 100*(self.system.qubit_array+1.25)
        for count,value in enumerate(target_pair):
            self.system.qubit_array[value] = [10, 10+(2.5*(count))]
        evolve_time = 1
        gate_timing = 7.63407
        rabi = gate_timing*10**6
        detuning = 0*rabi
        def phase_function(t:float) -> float:
            return 2*np.pi*0.1122*np.cos(1.0431*gate_timing*(t) + 0.7318) + detuning*10**(-6)*t
        phi = 2.106894000762855
        self.ts = np.arange(0, evolve_time, (evolve_time)/self.system.n_steps)
        # plt.figure(figsize=(4.1,2.8))  # noqa: ERA001
        # plt.plot(self.ts, [phase_function(t) for t in self.ts], color='orange', linewidth=2.5)  # noqa: ERA001
        # plt.ylabel(r"$\phi_r$", fontsize='x-large')  # noqa: ERA001
        # plt.xlabel(r"$t/T$", fontsize='x-large')  # noqa: ERA001
        # plt.text(0.1,0.5, r"(d)", fontsize='x-large')  # noqa: ERA001
        # plt.tight_layout()  # noqa: ERA001
        # plt.show()  # noqa: ERA001

        self.system.hyperfine_detuning = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.hyperfine_rabi = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.hyperfine_phase = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.rydberg_detuning = lambda t: 0*t
        self.system.rydberg_rabi = lambda t: 0*t
        self.system.rydberg_phase = lambda t: 0*t

        self.system.rydberg_rabi = (lambda t: 0*t + rabi)
        self.system.rydberg_detuning = (lambda t: 0*t)
        self.system.rydberg_phase = (phase_function)

        self.hamiltonian = [self.system.total_hamiltonian(t) for t in self.ts]
        self.states = np.append(self.states, self.evolve_trotter(evolve_time), axis=0)
        self.p_gate(list(range(self.system.n_qubits)), -phi)
        self.system.qubit_array = initial_array

        return self.states

    def cnot_gate(self, target_pair:list) -> np.ndarray:
        """Implement CNOT gate on a given qubit pair."""
        self.hadamard_gate([target_pair[1]])
        self.cz_gate(target_pair)
        self.hadamard_gate([target_pair[1]])
        return self.states

    def ccz_gate(self, target_triplet:np.ndarray, ratio:float=-0.6821) -> None:
        """Implement CCZ gate."""
        initial_array = self.system.qubit_array
        self.system.qubit_array = 100*(self.system.qubit_array+1.25)
        self.system.qubit_array[target_triplet[0]] = [-1,-np.sqrt(3)/2]
        self.system.qubit_array[target_triplet[1]] = [1,-np.sqrt(3)/2]
        self.system.qubit_array[target_triplet[2]] = [0, np.sqrt(3)/2]
        evolve_time = 1
        gate_timing = 11.0018/(evolve_time)
        rabi = gate_timing*10**6
        detuning = -ratio*rabi
        phase_offset = 0
        phi = 0.1749580185919
        def phase_function(t:float) -> float:
            return -(2*np.pi*2.1460*np.sin(0.2101*gate_timing*(t-(evolve_time/2)) - phase_offset) - 2*np.pi*0.0719*np.sin(1.8957*gate_timing*(t-(evolve_time/2)) - phase_offset) - 2*np.pi*0.6432*np.tanh(0.6941*gate_timing*(t-(evolve_time/2)) - phase_offset) + (detuning*10**(-6))*(t-(evolve_time/2)))  # noqa: E501
        self.ts = np.arange(0, evolve_time, (evolve_time)/self.system.n_steps)
        # plt.figure(figsize=(4.1,2.8))  # noqa: ERA001
        # plt.plot(self.ts, [phase_function(t) for t in self.ts], color="orange", linewidth=2.5)  # noqa: ERA001
        # plt.ylabel(r"$\phi_r$ [rad]", fontsize="x-large")  # noqa: ERA001
        # plt.xlabel(r"$t/T$", fontsize="x-large")  # noqa: ERA001
        # plt.text(0.1,0.5, r"(d)", fontsize="x-large")  # noqa: ERA001
        # plt.tight_layout()  # noqa: ERA001
        # plt.show()  # noqa: ERA001

        self.system.hyperfine_detuning = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.hyperfine_rabi = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.hyperfine_phase = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.rydberg_detuning = lambda t: 0*t
        self.system.rydberg_rabi = lambda t: 0*t
        self.system.rydberg_phase = lambda t: 0*t

        self.system.rydberg_rabi = (lambda t: 0*t + rabi)
        self.system.rydberg_detuning = (lambda t: 0*t + detuning)
        self.system.rydberg_phase = (phase_function)

        self.hamiltonian = [self.system.total_hamiltonian(t) for t in self.ts]
        self.states = np.append(self.states, self.evolve_trotter(evolve_time), axis=0)
        phi=np.angle(self.states[-1][1])
        self.p_gate([0,1,2], -phi)

        self.cz_gate([0,2])
        self.cz_gate([0,1])
        self.cz_gate([1,2])

        self.system.qubit_array = initial_array

        return self.states

    def toffoli_gate(self, target_triplet:list) -> np.ndarray:
        """Implement Toffoli gate on a given qubit triplet."""
        self.hadamard_gate([target_triplet[-1]])
        self.ccz_gate(target_triplet)
        self.hadamard_gate([target_triplet[-1]])
        return self.states

    def decoherence_gate_resonant(self) -> np.ndarray:
        """Implement resonant pulse on system with decoherence effects."""
        evolve_time = 1
        rho_array = []

        self.system.gamma_relaxation = 2*np.pi*10**6
        self.system.gamma_dephasing = 2*np.pi*10**6

        self.system.hyperfine_detuning = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.hyperfine_rabi = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.hyperfine_phase = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.rydberg_detuning = lambda t: 0*t
        self.system.rydberg_rabi = lambda t: 0*t
        self.system.rydberg_phase = lambda t: 0*t

        self.system.rydberg_rabi = (lambda t: 0*t + (5*np.pi*10**6))

        rho_array = rho_array + self.evolve_rk4(self.density_operator(self.states[-1]), evolve_time)

        p_00_array = []
        p_psiplus_array = []
        ts = np.arange(0, evolve_time, (evolve_time)/self.system.n_steps)
        for rho in rho_array:
            p_00_array.append(np.real(rho[4][4]))
            p_psiplus_array.append(np.real(2*rho[5][5]))

        return p_00_array, p_psiplus_array, ts

    def decoherence_gate_sweep(self) -> np.ndarray:
        """Implement cubic sweep pulse on system with decoherence effects."""
        evolve_time = 1
        rho_array = []

        def detuning_function(t:float) -> float:
            return 11*(2*np.pi*50*10**6)*((t-(evolve_time/2))/evolve_time)**3

        self.system.gamma_relaxation = 1*np.pi*2*10**6
        self.system.gamma_dephasing = 0*np.pi*10**6

        self.system.hyperfine_detuning = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.hyperfine_rabi = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.hyperfine_phase = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.rydberg_detuning = lambda t: 0*t
        self.system.rydberg_rabi = lambda t: 0*t
        self.system.rydberg_phase = lambda t: 0*t

        self.system.rydberg_detuning = (detuning_function)
        self.system.rydberg_rabi = (lambda t: 0*t + (2*np.pi*10**6))

        rho_array = rho_array + self.evolve_rk4(self.density_operator(self.states[-1]), evolve_time)

        p_00_array = []
        p_psiplus_array = []
        ts = np.arange(0, evolve_time, (evolve_time)/self.system.n_steps)
        for rho in rho_array:
            p_00_array.append(np.real(rho[4][4]))
            p_psiplus_array.append(np.real(2*rho[5][5]))

        return p_00_array, p_psiplus_array, ts

    def cz_decoherence(self, target_pair:list, input_rho:np.ndarray, decay:float) -> np.ndarray:
        """Implement CZ gate with decoherence effects."""
        initial_array = self.system.qubit_array
        self.system.qubit_array = 100*(self.system.qubit_array+1.25)
        for count,value in enumerate(target_pair):
            self.system.qubit_array[value] = [10, 10+(2.5*(count))]
        evolve_time = 1
        self.ts = np.arange(0, evolve_time, (evolve_time)/self.system.n_steps)
        y = 0.378233
        s = (np.pi*2)/(np.sqrt(y**2 + 2))
        sq = np.sqrt((y**2)+1)
        sqs = (s/2)*sq
        num = -sq*np.cos(sqs) + 1j*y*np.sin(sqs)
        denom = sq*np.cos(sqs) + 1j*y*np.sin(sqs)
        phase = 2*np.pi-1j*np.log(num/denom)
        phi = 0.7617829

        phase = 3.90242
        detuning_ratio = 0.377871

        rho_array = []

        self.system.gamma_relaxation = decay*0.1*(np.pi*2*10**4)/((np.sqrt(detuning_ratio**2 + 2))*evolve_time)
        self.system.gamma_dephasing = decay*0.4*(np.pi*2*10**4)/((np.sqrt(detuning_ratio**2 + 2))*evolve_time)

        self.system.hyperfine_detuning = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.hyperfine_rabi = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.hyperfine_phase = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.rydberg_detuning = lambda t: 0*t
        self.system.rydberg_rabi = lambda t: 0*t
        self.system.rydberg_phase = lambda t: 0*t

        self.system.rydberg_rabi = (lambda t: 0*t + (np.pi*2*10**6)/((np.sqrt(detuning_ratio**2 + 2))*evolve_time))
        self.system.rydberg_detuning = (lambda t: 0*t + detuning_ratio*self.system.rydberg_rabi(0))
        self.system.rydberg_phase = (lambda t: 0*t + 1.524)

        rho_array = rho_array + self.evolve_rk4(input_rho, evolve_time)

        self.system.rydberg_phase = (lambda t: (0*t + phase))

        rho_array = rho_array + self.evolve_rk4(rho_array[-1], evolve_time)
        # detuning_ratio = 1000000000  # noqa: ERA001
        # alpha = 0.16  # noqa: ERA001
        # A = -phi/(2*np.pi*0.42*evolve_time)  # noqa: ERA001
        # blackman_window = lambda t: (1/np.sqrt(1+detuning_ratio**2))*(2*np.pi*10**6)*(((1-alpha)/2)*A - (A/2)*(np.cos((2*np.pi*t)/evolve_time)) + ((A*alpha)/2)*(np.cos((4*np.pi*t)/evolve_time)))  # noqa: E501, ERA001

        self.system.gamma_relaxation = decay*0.5*(np.abs(phi)/np.pi)*5.25*10**4
        self.system.gamma_dephasing = decay*2*(np.abs(phi)/np.pi)*5.25*10**4

        self.system.hyperfine_detuning = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.hyperfine_rabi = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.hyperfine_phase = [(lambda t: 0*t) for i in range(self.system.n_qubits)]
        self.system.rydberg_detuning = lambda t: 0*t
        self.system.rydberg_rabi = lambda t: 0*t
        self.system.rydberg_phase = lambda t: 0*t

        # for target in target_pair:
        #     self.system.hyperfine_rabi[target] = lambda t: blackman_window(t)  # noqa: ERA001
        #     self.system.hyperfine_detuning[target] = lambda t: detuning_ratio*blackman_window(t)  # noqa: ERA001

        rho_array = rho_array + self.evolve_rk4(rho_array[-1], evolve_time)

        self.system.qubit_array = initial_array

        return rho_array[-1]

def grover() -> None:
    """Implement grover algorithm for the |111> state."""
    grover_circuit = Circuit(System(), np.ones(8))
    for _i in range(2):
        # apply oracle
        grover_circuit.ccz_gate([0,1,2])

        # reflection, i.e. application of  Grover Diffusion Operator
        grover_circuit.hadamard_gate([0,1,2])
        grover_circuit.rx_gate([0,1,2], np.pi)
        grover_circuit.ccz_gate([0,1,2])
        grover_circuit.rx_gate([0,1,2], np.pi)
        grover_circuit.hadamard_gate([0,1,2])

    data = [np.real(grover_circuit.output()[i]*np.conjugate(grover_circuit.output()[i])) for i in range(len(grover_circuit.output()))]
    plt.figure(figsize=(6,2.5))
    plt.bar(range(len(data)), data, tick_label=[r"$\ket{000}$", r"$\ket{001}$", r"$\ket{010}$", r"$\ket{011}$", r"$\ket{100}$", r"$\ket{101}$", r"$\ket{110}$", r"$\ket{111}$"])  # noqa: E501
    plt.ylim(0,1)
    plt.xlabel(r"$\ket{\phi}$", fontsize="x-large")
    plt.ylabel(r"P($\ket{\phi}; \ket{\psi}$)", fontsize="x-large")
    plt.tight_layout()
    plt.show()

def decoherence_simulation(protocol:str) -> None:
    """Plot decoherence simulation for the specified evolution."""
    d_shift = np.linspace(0, 5, 1)
    colormap = mpl.colormaps["cool"].resampled(1)(range(1))  # noqa: F841
    _fig, axs = plt.subplots(1, sharex=True, figsize=(3, 1))
    for _colour, d in enumerate(d_shift):
        decoherence_circuit = Circuit(System([[0,0], [0, 2.5 + d]]), [0,0,0,0,1,0,0,0,0])
        if protocol == "detuning sweep":
            p_00_array, p_psiplus_array, ts_array = decoherence_circuit.decoherence_gate_sweep()
        if protocol == "resonant":
            p_00_array, p_psiplus_array, ts_array = decoherence_circuit.decoherence_gate_resonant()
        # axs[0].plot(ts_array, p_00_array[0:-1], color=colormap[colour])  # noqa: ERA001
        axs.plot(np.sqrt(2)*2.5*np.array(ts_array), p_00_array[0:-1], color="turquoise", label=r"$\ket{g}$")
        # axs.set_ylabel(r"Population", fontsize='x-large')  # noqa: ERA001
        axs.set_xlabel(r"$\Omega t/(2\pi)$", fontsize="x-large")
        axs.set_yticks([0,0.5,1])
        axs.plot(np.sqrt(2)*2.5*np.array(ts_array), p_psiplus_array[0:-1], color="purple", label=r"$\ket{e}$")
    # plt.legend(frameon=False, fontsize='x-large')  # noqa: ERA001
    plt.tight_layout()
    plt.show()

def decoherence_simulation_cz() -> None:  # noqa: PLR0915
    """Plot evolution of CZ gate implementation on Bloch Sphere with decoherence effects."""
    decoherence_circuit = Circuit(System([[0,0], [0, 2.5]]), [0,1,0,0,0,0,0,0,0])
    rho_array = []

    evolve_time = 1

    phase = 3.90242
    detuning_ratio = 0.377871

    decoherence_circuit.system.gamma_relaxation = 0*np.pi*10**6
    decoherence_circuit.system.gamma_dephasing = 0*np.pi*10**5

    decoherence_circuit.system.hyperfine_detuning = [(lambda t: 0*t) for i in range(decoherence_circuit.system.n_qubits)]
    decoherence_circuit.system.hyperfine_rabi = [(lambda t: 0*t) for i in range(decoherence_circuit.system.n_qubits)]
    decoherence_circuit.system.hyperfine_phase = [(lambda t: 0*t) for i in range(decoherence_circuit.system.n_qubits)]
    decoherence_circuit.system.rydberg_detuning = lambda t: 0*t
    decoherence_circuit.system.rydberg_rabi = lambda t: 0*t
    decoherence_circuit.system.rydberg_phase = lambda t: 0*t

    decoherence_circuit.system.rydberg_rabi = (lambda t: 0*t + (np.pi*2*10**6)/((np.sqrt(detuning_ratio**2 + 2))*evolve_time))
    decoherence_circuit.system.rydberg_detuning = (lambda t: 0*t + detuning_ratio*decoherence_circuit.system.rydberg_rabi(0))
    decoherence_circuit.system.rydberg_phase = (lambda t: 0*t + 1.524)

    rho_array = rho_array + decoherence_circuit.evolve_rk4(decoherence_circuit.density_operator(decoherence_circuit.states[-1]), evolve_time)

    decoherence_circuit.system.rydberg_phase = (lambda t: (0*t + phase))

    rho_array = rho_array + decoherence_circuit.evolve_rk4(rho_array[-1], evolve_time)

    bloch_vectors = []
    for rho in rho_array:
        u = 2*np.real(rho[1][2])
        v = 2*np.imag(rho[2][1])
        w = np.real(rho[1][1] - rho[2][2])
        bloch_vectors.append([0, 0, 0, u, v, w])
    _X, _Y, _Z, U, V, W = zip(*bloch_vectors)  # noqa: N806

    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)
    r = 1
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    figure = plt.figure()
    ax = figure.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, cmap="binary", alpha=0.2)
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.25, rstride=8, cstride=8, linewidths=0.8)
    ax.scatter(U[0:1000], V[0:1000], W[0:1000], c = np.sqrt(np.arange(1000))[-1::-1], cmap="Greens", s = 1)
    ax.scatter(U[1000:2000:8], V[1000:2000:8], W[1000:2000:8], c = np.sqrt(np.arange(1000)[-1::-8]), cmap="Greens", s = 1)
    ax.text(0,0,-1.1,r"$\ket{0r}$", size="x-large")
    ax.text(0,0,1.2,r"$\ket{01}$", size="x-large")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect((2, 2, 2))
    ax.grid(visible=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    plt.show()

    decoherence_circuit = Circuit(System([[0,0], [0, 2.5]]), [0,0,0,0,1,0,0,0,0])

    rho_array = []

    decoherence_circuit.system.gamma_relaxation = 0*np.pi*10**5
    decoherence_circuit.system.gamma_dephasing = 0*np.pi*10**5

    decoherence_circuit.system.hyperfine_detuning = [(lambda t: 0*t) for _i in range(decoherence_circuit.system.n_qubits)]
    decoherence_circuit.system.hyperfine_rabi = [(lambda t: 0*t) for _i in range(decoherence_circuit.system.n_qubits)]
    decoherence_circuit.system.hyperfine_phase = [(lambda t: 0*t) for _i in range(decoherence_circuit.system.n_qubits)]
    decoherence_circuit.system.rydberg_detuning = lambda t: 0*t
    decoherence_circuit.system.rydberg_rabi = lambda t: 0*t
    decoherence_circuit.system.rydberg_phase = lambda t: 0*t

    decoherence_circuit.system.rydberg_rabi = (lambda t: 0*t + (np.pi*2*10**6)/((np.sqrt(detuning_ratio**2 + 2))*evolve_time))
    decoherence_circuit.system.rydberg_detuning = (lambda t: 0*t + detuning_ratio*decoherence_circuit.system.rydberg_rabi(0))
    decoherence_circuit.system.rydberg_phase = (lambda t: 0*t + 1.524)

    rho_array = rho_array + decoherence_circuit.evolve_rk4(decoherence_circuit.density_operator(decoherence_circuit.states[-1]), evolve_time)

    decoherence_circuit.system.rydberg_phase = (lambda t: (0*t + phase))

    rho_array = rho_array + decoherence_circuit.evolve_rk4(rho_array[-1], evolve_time)

    bloch_vectors = []
    for rho in rho_array:
        u = 2*np.sqrt(2)*np.real(rho[4][5])
        v = 2*np.sqrt(2)*np.imag(rho[5][4])
        w = np.real(rho[4][4] - 2*rho[5][5])
        bloch_vectors.append([0, 0, 0, u, v, w])
    _X, _Y, _Z, U, V, W = zip(*bloch_vectors)  # noqa: N806

    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)
    r = 1
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    figure = plt.figure()
    ax = figure.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, cmap="binary", alpha=0.2)
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.25, rstride=8, cstride=8, linewidths=0.8)
    ax.scatter(U[0:1000], V[0:1000], W[0:1000], c = np.sqrt(np.arange(1000))[-1::-1], cmap="Reds", s = 1)
    ax.scatter(U[1000:2000:8], V[1000:2000:8], W[1000:2000:8], c = np.sqrt(np.arange(1000)[-1::-8]), cmap="Reds", s = 1)
    ax.text(0,0,-1.1,r"$\ket{W}$", size="x-large")
    ax.text(0,0,1.2,r"$\ket{11}$", size="x-large")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect((2, 2, 2))
    ax.grid(visible=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    plt.show()

def decoherence_simulation_ccz() -> None:  # noqa: PLR0915
    """Plot evolution of CCZ gate implementation on Bloch Sphere with decoherence effects."""
    decoherence_circuit = Circuit(System([[-1,-np.sqrt(3)/2], [1,-np.sqrt(3)/2], [0, np.sqrt(3)/2]]), [0,1,0,0,0,0,0,0])
    rho_array = []

    evolve_time = 1

    gate_timing = 11.0018/(evolve_time)
    rabi = gate_timing*10**6
    detuning = -0.6821*rabi
    phi = 0.1749580185919
    def phase_function(t:float) -> float:
        return -(2*np.pi*2.1460*np.sin(0.2101*gate_timing*(t-(evolve_time/2))) - 2*np.pi*0.0719*np.sin(1.8957*gate_timing*(t-(evolve_time/2))) - 2*np.pi*0.6432*np.tanh(0.6941*gate_timing*(t-(evolve_time/2))) + (detuning*10**(-6))*(t-(evolve_time/2)))  # noqa: E501

    decoherence_circuit.system.hyperfine_detuning = [(lambda t: 0*t) for i in range(decoherence_circuit.system.n_qubits)]
    decoherence_circuit.system.hyperfine_rabi = [(lambda t: 0*t) for i in range(decoherence_circuit.system.n_qubits)]
    decoherence_circuit.system.hyperfine_phase = [(lambda t: 0*t) for i in range(decoherence_circuit.system.n_qubits)]
    decoherence_circuit.system.rydberg_detuning = lambda t: 0*t
    decoherence_circuit.system.rydberg_rabi = lambda t: 0*t
    decoherence_circuit.system.rydberg_phase = lambda t: 0*t

    decoherence_circuit.system.rydberg_rabi = (lambda t: 0*t + rabi)
    decoherence_circuit.system.rydberg_detuning = (lambda t: 0*t + detuning)
    decoherence_circuit.system.rydberg_phase = (phase_function)

    rho_array = rho_array + decoherence_circuit.evolve_rk4(decoherence_circuit.density_operator(decoherence_circuit.states[-1]), evolve_time)

    bloch_vectors = []
    for rho in rho_array:
        u = 2*np.real(rho[1][2])
        v = 2*np.imag(rho[2][1])
        w = np.real(rho[1][1] - rho[2][2])
        bloch_vectors.append([0, 0, 0, u, v, w])
    _X, _Y, _Z, U, V, W = zip(*bloch_vectors)  # noqa: N806

    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)
    r = 1
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    figure = plt.figure()
    ax = figure.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, cmap="binary", alpha=0.2)
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.25, rstride=8, cstride=8, linewidths=0.8)
    ax.scatter(U[0:1000], V[0:1000], W[0:1000], c = np.sqrt(np.arange(1000)), cmap="Greens", s = 1)
    ax.text(0,0,-1.1,r"$\ket{00r}$", size="x-large")
    ax.text(0,0,1.2,r"$\ket{001}$", size="x-large")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect((2, 2, 2))
    ax.grid(visible=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    plt.show()

    decoherence_circuit_2 = Circuit(System([[-1,-np.sqrt(3)/2], [1,-np.sqrt(3)/2], [0, np.sqrt(3)/2]]), [0,0,0,1,0,0,0,0])
    rho_array = []

    decoherence_circuit_2.system.hyperfine_detuning = [(lambda t: 0*t) for i in range(decoherence_circuit_2.system.n_qubits)]
    decoherence_circuit_2.system.hyperfine_rabi = [(lambda t: 0*t) for i in range(decoherence_circuit_2.system.n_qubits)]
    decoherence_circuit_2.system.hyperfine_phase = [(lambda t: 0*t) for i in range(decoherence_circuit_2.system.n_qubits)]
    decoherence_circuit_2.system.rydberg_detuning = lambda t: 0*t
    decoherence_circuit_2.system.rydberg_rabi = lambda t: 0*t
    decoherence_circuit_2.system.rydberg_phase = lambda t: 0*t

    decoherence_circuit_2.system.rydberg_rabi = (lambda t: 0*t + rabi)
    decoherence_circuit_2.system.rydberg_detuning = (lambda t: 0*t + detuning)
    decoherence_circuit_2.system.rydberg_phase = (phase_function)

    rho_array = rho_array + decoherence_circuit_2.evolve_rk4(decoherence_circuit_2.density_operator(decoherence_circuit_2.states[-1]), evolve_time)

    bloch_vectors = []
    for rho in rho_array:
        u = 2*np.real(np.sqrt(2)*rho[4][5])
        v = 2*np.imag(np.sqrt(2)*rho[5][4])
        w = np.real(rho[4][4] - 2*rho[5][5])
        bloch_vectors.append([0, 0, 0, u, v, w])
    _X, _Y, _Z, U, V, W = zip(*bloch_vectors)  # noqa: N806
    figure = plt.figure()
    ax = figure.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, cmap="Greys", alpha=0.2)
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.25, rstride=8, cstride=8, linewidths=0.8)
    ax.scatter(U[0:1000], V[0:1000], W[0:1000], c = np.sqrt(np.arange(1000)), cmap="Reds", s = 1)
    ax.text(0,0,-1.1,r"$\ket{0} \bigotimes \ket{W}$", size="x-large")
    ax.text(0,0,1.2,r"$\ket{011}$", size="x-large")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect((2, 2, 2))
    ax.grid(visible=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    plt.show()

    decoherence_circuit_3 = Circuit(System([[-1,-np.sqrt(3)/2], [1,-np.sqrt(3)/2], [0, np.sqrt(3)/2]]), [0,0,0,0,0,0,0,1])
    rho_array = []

    decoherence_circuit_3.system.hyperfine_detuning = [(lambda t: 0*t) for i in range(decoherence_circuit_3.system.n_qubits)]
    decoherence_circuit_3.system.hyperfine_rabi = [(lambda t: 0*t) for i in range(decoherence_circuit_3.system.n_qubits)]
    decoherence_circuit_3.system.hyperfine_phase = [(lambda t: 0*t) for i in range(decoherence_circuit_3.system.n_qubits)]
    decoherence_circuit_3.system.rydberg_detuning = lambda t: 0*t
    decoherence_circuit_3.system.rydberg_rabi = lambda t: 0*t
    decoherence_circuit_3.system.rydberg_phase = lambda t: 0*t

    decoherence_circuit_3.system.rydberg_rabi = (lambda t: 0*t + rabi)
    decoherence_circuit_3.system.rydberg_detuning = (lambda t: 0*t + detuning)
    decoherence_circuit_3.system.rydberg_phase = (phase_function)

    rho_array = rho_array + decoherence_circuit_3.evolve_rk4(decoherence_circuit_3.density_operator(decoherence_circuit_3.states[-1]), evolve_time)

    bloch_vectors = []
    for rho in rho_array:
        u = 2*np.real(np.sqrt(3)*rho[13][14])
        v = 2*np.imag(np.sqrt(3)*rho[14][13])
        w = np.real(rho[13][13] - 3*rho[14][14])
        bloch_vectors.append([0, 0, 0, u, v, w])
    _X, _Y, _Z, U, V, W = zip(*bloch_vectors)  # noqa: N806
    figure = plt.figure()
    ax = figure.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, cmap="Greys", alpha=0.2)
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.25, rstride=8, cstride=8, linewidths=0.8)
    ax.scatter(U[0:1000], V[0:1000], W[0:1000], c = np.sqrt(np.arange(1000)), cmap="Blues", s = 1)
    ax.text(0,0,-1.1,r"$\ket{W_1}$", size="x-large")
    ax.text(0,0,1.2,r"$\ket{111}$", size="x-large")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect((2, 2, 2))
    ax.grid(visible=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    plt.show()

def cz_flat_graph(protocol:str) -> None:
    """Plot evolution of probabilities for CZ gate implementation."""
    cz_flat_system = System()
    cz_flat_circuit_1 = Circuit(cz_flat_system, [0,1,0,0,0,0,0,0])
    cz_flat_circuit_2 = Circuit(cz_flat_system, [0,0,0,1,0,0,0,0])
    if protocol == "levine":
        cz_flat_circuit_1.cz_gate([0,1])
        cz_flat_circuit_2.cz_gate([1,2])

        ts_array = np.linspace(0, (1/cz_flat_circuit_1.system.n_steps)*len(cz_flat_circuit_1.states), len(cz_flat_circuit_1.states))[0:2000]
        prob_1_1 = [cz_flat_circuit_1.probability(cz_flat_circuit_1.states[i], [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) for i in range(len(ts_array))][0:1000]  # noqa: E501
        prob_1_2 = [cz_flat_circuit_1.probability(cz_flat_circuit_1.states[i], [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) for i in range(len(ts_array))][1000:2000]  # noqa: E501
        prob_2_1 = [cz_flat_circuit_2.probability(cz_flat_circuit_2.states[i], [0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) for i in range(len(ts_array))][0:1000]  # noqa: E501
        prob_2_2 = [cz_flat_circuit_2.probability(cz_flat_circuit_2.states[i], [0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) for i in range(len(ts_array))][1000:2000]  # noqa: E501

        plt.figure(figsize=(3.7,2.4))
        plt.scatter(ts_array[0:1000], prob_1_1, c=np.sqrt(ts_array[1000:0:-1]), cmap="Greens", s=1.5)
        plt.scatter(ts_array[1000:2000:16], prob_1_2[0::16], c=np.sqrt(ts_array[1000:0:-16]), cmap="Greens", s=1.5)
        plt.scatter(ts_array[0:1000], prob_2_1, c=np.sqrt(ts_array[1000:0:-1]), cmap="Reds", s=1.5)
        plt.scatter(ts_array[1000:2000:16], prob_2_2[0::16], c=np.sqrt(ts_array[1000:0:-16]), cmap="Reds", s=1.5)
        plt.yticks([0,0.5,1.0])
        plt.xticks([0,0.5,1,1.5,2])
        plt.text(0,0.8, r"(c)", fontsize="x-large")
        plt.ylabel(r"Population", fontsize="x-large")
        plt.xlabel(r"$t/\tau$", fontsize="x-large")
        plt.tight_layout()

        plt.show()
    if protocol == "time optimal":
        cz_flat_circuit_1.cz_time_optimal([0,1])
        cz_flat_circuit_2.cz_time_optimal([1,2])

        ts_array = np.linspace(0, (1/cz_flat_circuit_1.system.n_steps)*len(cz_flat_circuit_1.states), len(cz_flat_circuit_1.states))[0:2000]
        prob_1_1 = [cz_flat_circuit_1.probability(cz_flat_circuit_1.states[i], [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) for i in range(len(ts_array))][0:1000]  # noqa: E501
        prob_2_1 = [cz_flat_circuit_2.probability(cz_flat_circuit_2.states[i], [0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) for i in range(len(ts_array))][0:1000]  # noqa: E501

        plt.figure(figsize=(4,2.8))
        plt.scatter(ts_array[0:1000], prob_1_1, c=np.sqrt(ts_array[1000:0:-1]), cmap="Greens", s=1.5)
        plt.scatter(ts_array[0:1000], prob_2_1, c=np.sqrt(ts_array[1000:0:-1]), cmap="Reds", s=1.5)
        plt.yticks([0,0.2,0.4,0.6,0.8,1.0])
        plt.xticks([0,0.5,1])
        plt.text(0,0.8, r"(c)", fontsize="x-large")
        plt.xlabel(r"$t/T$", fontsize="x-large")
        plt.ylabel(r"Population", fontsize="x-large")
        plt.tight_layout()

        plt.show()

def cz_bloch_graph(protocol:str) -> None:  # noqa: PLR0915
    """Plot evolution of CZ gate implementation on Bloch Sphere."""
    cz_system = System([[0,0], [0, 2.5]])
    cz_test_circuit_1 = Circuit(cz_system, [0,1,0,0,0,0,0,0,0])
    if protocol == "levine":
        cz_test_circuit_1.cz_gate([0,1])
        bloch_vectors = []
        for state in cz_test_circuit_1.states:
            density_operator = np.tensordot(state, np.conjugate(state), axes=0)
            u = 2*np.real(density_operator[1][2])
            v = 2*np.imag(density_operator[2][1])
            w = np.real(density_operator[1][1] - density_operator[2][2])
            bloch_vectors.append([0, 0, 0, u, v, w])
        _X, _Y, _Z, U, V, W = zip(*bloch_vectors)  # noqa: N806

        theta = np.linspace(0, 2*np.pi, 100)
        phi = np.linspace(0, np.pi, 50)
        theta, phi = np.meshgrid(theta, phi)
        r = 1
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)

        figure = plt.figure(figsize=(4.5,4.5))
        ax = figure.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, z, cmap="binary", alpha=0.2)
        ax.plot_wireframe(x, y, z, color="gray", alpha=0.25, rstride=8, cstride=8, linewidths=0.8)
        ax.scatter(U[0:1000], V[0:1000], W[0:1000], c = np.sqrt(np.arange(1000))[-1::-1], cmap="Greens", s = 1.5)
        ax.scatter(U[1000:2000:8], V[1000:2000:8], W[1000:2000:8], c = np.sqrt(np.arange(1000))[-1::-8], cmap="Greens", s = 1.5)
        ax.text(0,0,-1.1,r"$\ket{0r}$", size="x-large")
        ax.text(0,0,1.2,r"$\ket{01}$", size="x-large")
        ax.text(-1, -1, -1.5, r"(e)", size="x-large")
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_box_aspect((2, 2, 2))
        ax.grid(visible=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        plt.show()

        cz_test_circuit_1 = Circuit(cz_system, [0,0,0,0,1,0,0,0,0])
        cz_test_circuit_1.cz_gate([0,1])
        bloch_vectors = []
        for state in cz_test_circuit_1.states:
            density_operator = np.tensordot(state, np.conjugate(state), axes=0)
            u = 2*np.real(np.sqrt(2)*density_operator[4][5])
            v = 2*np.imag(np.sqrt(2)*density_operator[5][4])
            w = np.real(density_operator[4][4] - 2*density_operator[5][5])
            bloch_vectors.append([0, 0, 0, u, v, w])
        _X, _Y, _Z, U, V, W = zip(*bloch_vectors)  # noqa: N806
        figure = plt.figure(figsize=(4.5,4.5))
        ax = figure.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, z, cmap="Greys", alpha=0.2)
        ax.plot_wireframe(x, y, z, color="gray", alpha=0.25, rstride=8, cstride=8, linewidths=0.8)
        ax.scatter(U[0:1000], V[0:1000], W[0:1000], c = np.sqrt(np.arange(1000))[-1::-1], cmap="Reds", s = 1.5)
        ax.scatter(U[1000:2000:8], V[1000:2000:8], W[1000:2000:8], c = np.sqrt(np.arange(1000))[-1::-8], cmap="Reds", s = 1.5)
        ax.text(0,0,-1.1,r"$\ket{W}$", size="x-large")
        ax.text(0,0,1.2,r"$\ket{11}$", size="x-large")
        ax.text(-1, -1, -1.5, r"(f)", size="x-large")
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_box_aspect((2, 2, 2))
        ax.grid(visibile=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        plt.show()
    if protocol == "time optimal":
        cz_test_circuit_1.cz_time_optimal([0,1])
        bloch_vectors = []
        for state in cz_test_circuit_1.states:
            density_operator = np.tensordot(state, np.conjugate(state), axes=0)
            u = 2*np.real(density_operator[1][2])
            v = 2*np.imag(density_operator[2][1])
            w = np.real(density_operator[1][1] - density_operator[2][2])
            bloch_vectors.append([0, 0, 0, u, v, w])
        _X, _Y, _Z, U, V, W = zip(*bloch_vectors)  # noqa: N806

        theta = np.linspace(0, 2 * np.pi, 100)
        phi = np.linspace(0, np.pi, 50)
        theta, phi = np.meshgrid(theta, phi)
        r = 1
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)

        figure = plt.figure(figsize=(4.5,4.5))
        ax = figure.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, z, cmap="binary", alpha=0.2)
        ax.plot_wireframe(x, y, z, color="gray", alpha=0.25, rstride=8, cstride=8, linewidths=0.8)
        ax.scatter(U[0:1000], V[0:1000], W[0:1000], c = np.sqrt(np.arange(1000)[1000::-1]), cmap="Greens", s = 1.5)
        ax.text(0,0,-1.1,r"$\ket{0r}$", size="x-large")
        ax.text(0,0,1.2,r"$\ket{01}$", size="x-large")
        ax.text(-1, -1, -1.5, r"(a)", size="x-large")
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_box_aspect((2, 2, 2))
        ax.grid(visible=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        plt.show()

        cz_test_circuit_1 = Circuit(cz_system, [0,0,0,0,1,0,0,0,0])
        cz_test_circuit_1.cz_time_optimal([0,1])
        bloch_vectors = []
        for state in cz_test_circuit_1.states:
            density_operator = np.tensordot(state, np.conjugate(state), axes=0)
            u = 2*np.real(np.sqrt(2)*density_operator[4][5])
            v = 2*np.imag(np.sqrt(2)*density_operator[5][4])
            w = np.real(density_operator[4][4] - 2*density_operator[5][5])
            bloch_vectors.append([0, 0, 0, u, v, w])
        _X, _Y, _Z, U, V, W = zip(*bloch_vectors)  # noqa: N806
        figure = plt.figure(figsize=(4.5,4.5))
        ax = figure.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, z, cmap="Greys", alpha=0.2)
        ax.plot_wireframe(x, y, z, color="gray", alpha=0.25, rstride=8, cstride=8, linewidths=0.8)
        ax.scatter(U[0:1000], V[0:1000], W[0:1000], c = np.sqrt(np.arange(1000)[1000::-1]), cmap="Reds", s = 1.5)
        ax.text(0,0,-1.1,r"$\ket{W}$", size="x-large")
        ax.text(0,0,1.2,r"$\ket{11}$", size="x-large")
        ax.text(-1, -1, -1.5, r"(b)", size="x-large")
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_box_aspect((2, 2, 2))
        ax.grid(visible=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        plt.show()

def cz_phase_graph() -> None:
    """Plot dynamical phases accumulated by each basis state on implementation of Levine CZ gate."""
    cz_phase_system = System()
    cz_phase_circuit_1 = Circuit(cz_phase_system, [0,1,0,0,0,0,0,0])
    cz_phase_circuit_2 = Circuit(cz_phase_system, [0,0,0,1,0,0,0,0])
    phi_01_array = []
    phi_11_array = []
    for ratio in np.arange(-0.024, 1.5, 1/40):
        cz_phase_circuit_1 = Circuit(cz_phase_system, [0,1,0,0,0,0,0,0])
        cz_phase_circuit_2 = Circuit(cz_phase_system, [0,0,0,1,0,0,0,0])

        cz_phase_circuit_1.cz_gate([0,1], ratio)
        result_1 = cz_phase_circuit_1.states[-1]
        cz_phase_circuit_2.cz_gate([1,2], ratio)
        result_2 = cz_phase_circuit_2.states[-1]

        phi_11 = np.angle(result_2[4])
        phi_01 = np.angle(result_1[1])
        if phi_11 > 0:
            phi_11_array.append(phi_11)
        else:
            phi_11_array.append(2*np.pi + phi_11)
        if phi_01 > 0:
            phi_01_array.append(phi_01)
        else:
            phi_01_array.append(2*np.pi + phi_01)

    plt.figure(figsize=(3.5,2.5))
    plt.plot(np.arange(-0.024, 1.5, 1/40)[1:], 2*np.pi - ((2*np.array(phi_01_array[1:])) - np.pi), color="#076304")
    plt.plot(np.arange(-0.024, 1.5, 1/40)[1:], 2*np.pi - np.array(phi_11_array)[1:], color="#960B0E")
    plt.plot(0.3788478, 2*2.32357, "*", c="black", ms=12)
    plt.legend([r"$2\phi_{01}-\pi$", r"$\phi_{11}$"], frameon=False, fontsize="large")
    plt.ylim(0,6.2835)
    plt.xlim(0, 1.4)
    plt.xlabel(r"$\Delta_r/\Omega_r$", fontsize="x-large")
    plt.text(0.42, 2*2.57357, r"CZ", fontsize="x-large")
    plt.text(0.4, 1, r"(e)", fontsize="x-large")
    plt.yticks([0, np.pi, 2*np.pi, 3*np.pi], [r"$0$", r"$\pi$", r"$2\pi$", r"$3\pi$"])
    plt.tick_params(labelsize="x-large")
    plt.tight_layout()

    plt.show()

def ccz_flat_graph() -> None:
    """Plot evolution of probabilities for CCZ gate implementation."""
    ccz_flat_system = System()
    ccz_flat_circuit_1 = Circuit(ccz_flat_system, [0,1,0,0,0,0,0,0])
    ccz_flat_circuit_2 = Circuit(ccz_flat_system, [0,0,0,1,0,0,0,0])
    ccz_flat_circuit_3 = Circuit(ccz_flat_system, [0,0,0,0,0,0,0,1])
    ccz_flat_circuit_1.ccz_gate([0,1,2])
    ccz_flat_circuit_2.ccz_gate([0,1,2])
    ccz_flat_circuit_3.ccz_gate([0,1,2])

    ts_array = np.linspace(0, (1/ccz_flat_circuit_1.system.n_steps)*len(ccz_flat_circuit_1.states), len(ccz_flat_circuit_1.states))[0:1000]
    prob_1 = [ccz_flat_circuit_1.probability(ccz_flat_circuit_1.states[i], [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) for i in range(len(ts_array))]  # noqa: E501
    prob_2 = [ccz_flat_circuit_2.probability(ccz_flat_circuit_2.states[i], [0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) for i in range(len(ts_array))]  # noqa: E501
    prob_3 = [ccz_flat_circuit_3.probability(ccz_flat_circuit_3.states[i], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0]) for i in range(len(ts_array))]  # noqa: E501

    plt.figure(figsize=(4,2.8))
    plt.scatter(ts_array, prob_1, c=np.sqrt(ts_array)[-1::-1], cmap="Greens", s=1.5)
    plt.scatter(ts_array, prob_2, c=np.sqrt(ts_array)[-1::-1], cmap="Reds", s=1.5)
    plt.scatter(ts_array, prob_3, c=np.sqrt(ts_array)[-1::-1], cmap="Blues", s=1.5)
    plt.yticks([0,0.2,0.4,0.6,0.8,1.0])
    plt.xticks([0,0.5,1])
    plt.text(0,0.8, r"(e)", fontsize="x-large")
    plt.xlabel(r"$t/T$", fontsize="x-large")
    plt.ylabel(r"Population", fontsize="x-large")
    plt.tight_layout()

    plt.show()

def ccz_bloch_graph() -> None:  # noqa: PLR0915
    """Plot evolution of CCZ gate implementation on Bloch Sphere."""
    ccz_test_system = System()
    ccz_test_circuit_1 = Circuit(ccz_test_system, [0,1,0,0,0,0,0,0])
    ccz_test_circuit_1.ccz_gate([0,1,2])
    bloch_vectors = []
    for state in ccz_test_circuit_1.states:
        density_operator = np.tensordot(state, np.conjugate(state), axes=0)
        u = 2*np.real(density_operator[1][2])
        v = 2*np.imag(density_operator[2][1])
        w = np.real(density_operator[1][1] - density_operator[2][2])
        bloch_vectors.append([0, 0, 0, u, v, w])
    _X, _Y, _Z, U, V, W = zip(*bloch_vectors)  # noqa: N806

    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)
    r = 1
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    figure = plt.figure(figsize=(3,3))
    ax = figure.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, cmap="binary", alpha=0.2)
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.25, rstride=8, cstride=8, linewidths=0.8)
    ax.scatter(U[0:1000], V[0:1000], W[0:1000], c = np.sqrt(np.arange(1000)), cmap="Greens", s = 1)
    ax.text(0,0,-1.1,r"$\ket{00r}$", size="x-large")
    ax.text(0,0,1.2,r"$\ket{001}$", size="x-large")
    ax.text(-1, -1, -1.5, r"(a)", size="x-large")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect((2, 2, 2))
    ax.grid(visible=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    plt.show()

    ccz_test_circuit_2 = Circuit(ccz_test_system, [0,0,0,1,0,0,0,0])
    ccz_test_circuit_2.ccz_gate([0,1,2])
    bloch_vectors = []
    for state in ccz_test_circuit_2.states:
        density_operator = np.tensordot(state, np.conjugate(state), axes=0)
        u = 2*np.real(np.sqrt(2)*density_operator[4][5])
        v = 2*np.imag(np.sqrt(2)*density_operator[5][4])
        w = np.real(density_operator[4][4] - 2*density_operator[5][5])
        bloch_vectors.append([0, 0, 0, u, v, w])
    _X, _Y, _Z, U, V, W = zip(*bloch_vectors)  # noqa: N806
    figure = plt.figure(figsize=(3,3))
    ax = figure.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, cmap="Greys", alpha=0.2)
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.25, rstride=8, cstride=8, linewidths=0.8)
    ax.scatter(U[0:1000], V[0:1000], W[0:1000], c = np.sqrt(np.arange(1000)), cmap="Reds", s = 1)
    ax.text(0,0,-1.1,r"$\ket{0} \bigotimes \ket{W}$", size="x-large")
    ax.text(0,0,1.2,r"$\ket{011}$", size="x-large")
    ax.text(-1, -1, -1.5, r"(b)", size="x-large")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect((2, 2, 2))
    ax.grid(visible=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    plt.show()

    ccz_test_circuit_3 = Circuit(ccz_test_system, [0,0,0,0,0,0,0,1])
    ccz_test_circuit_3.ccz_gate([0,1,2])
    bloch_vectors = []
    for state in ccz_test_circuit_3.states:
        density_operator = np.tensordot(state, np.conjugate(state), axes=0)
        u = 2*np.real(np.sqrt(3)*density_operator[13][14])
        v = 2*np.imag(np.sqrt(3)*density_operator[14][13])
        w = np.real(density_operator[13][13] - 3*density_operator[14][14])
        bloch_vectors.append([0, 0, 0, u, v, w])
    _X, _Y, _Z, U, V, W = zip(*bloch_vectors)  # noqa: N806
    figure = plt.figure(figsize=(3,3))
    ax = figure.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, cmap="Greys", alpha=0.2)
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.25, rstride=8, cstride=8, linewidths=0.8)
    ax.scatter(U[0:1000], V[0:1000], W[0:1000], c = np.sqrt(np.arange(1000)), cmap="Blues", s = 1)
    ax.text(0,0,-1.1,r"$\ket{W_1}$", size="x-large")
    ax.text(0,0,1.2,r"$\ket{111}$", size="x-large")
    ax.text(-1, -1, -1.5, r"(c)", size="x-large")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect((2, 2, 2))
    ax.grid(visible=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    plt.show()

def ccz_phase_graph() -> None:
    """Plot dynamical phases accumulated by each basis state on implementation of CCZ gate."""
    phi_001_array = []
    phi_011_array = []
    phi_111_array = []
    ratio_range = np.arange(-1.5,1.5,1/100)

    for i in ratio_range:
        ccz_circuit_1 = Circuit(System(), [0,1,0,0,0,0,0,0])
        ccz_circuit_2 = Circuit(System(), [0,0,0,1,0,0,0,0])
        ccz_circuit_3 = Circuit(System(), [0,0,0,0,0,0,0,1])

        result_001 = np.angle(ccz_circuit_1.ccz_gate([0,1,2], i)[-1][1])
        result_011 = np.angle(ccz_circuit_2.ccz_gate([0,1,2], i)[-1][4])
        result_111 = np.angle(ccz_circuit_3.ccz_gate([0,1,2], i)[-1][13])

        if result_001 > 0:
            phi_001_array.append(3*result_001)
        else:
            phi_001_array.append(3*(result_001+(2*np.pi)))
        if result_011 > 0:
            phi_011_array.append((3/2)*result_011)
        else:
            phi_011_array.append((3/2)*(result_011+(2*np.pi)))
        if result_111 > 0:
            phi_111_array.append(result_111)
        else:
            phi_111_array.append(result_111+(2*np.pi))

    plt.plot(ratio_range, np.array(phi_001_array)%(2*np.pi), label=r"$\phi_{001}$")
    plt.plot(ratio_range, np.array(phi_011_array)%(2*np.pi), label=r"$\phi_{011}$")
    plt.plot(ratio_range, (np.array(phi_111_array)-np.pi)%(2*np.pi), label=r"$\phi_{111}$")

    plt.legend()
    plt.show()

def realistic_cz_decay(n:int) -> float:
    """Return values for the fidelity change upon evolving system via CZ gate n times with realistic values of decoherence."""
    decoherence_circuit = Circuit(System([[0,0], [0, 2.5]]), [1,1,0,1,1,0,0,0,0])
    nonunitary_state = decoherence_circuit.cz_decoherence([0,1], decoherence_circuit.density_operator((1/2)*np.array([1,1,0,1,1,0,0,0,0])), 1)
    unitary_state = decoherence_circuit.cz_decoherence([0,1], decoherence_circuit.density_operator((1/2)*np.array([1,1,0,1,1,0,0,0,0])), 0)

    for _i in range(n-1):
        nonunitary_state = decoherence_circuit.cz_decoherence([0,1], nonunitary_state, 1)
        unitary_state = decoherence_circuit.cz_decoherence([0,1], unitary_state, 0)

    nonunitary_fidelity = np.sum(np.array([nonunitary_state[i][i] for i in range(len(nonunitary_state))])*np.array([1,1,0,1,1,0,0,0,0]))
    unitary_fidelity = np.sum(np.array([unitary_state[i][i] for i in range(len(unitary_state))])*np.array([1,1,0,1,1,0,0,0,0]))

    return nonunitary_fidelity/unitary_fidelity, nonunitary_fidelity

def main() -> None:
    """Execute desired functionality."""
    # cz_flat_graph("time optimal")  # noqa: ERA001

    # cz_bloch_graph("levine")  # noqa: ERA001

    # cz_phase_graph()  # noqa: ERA001

    # ccz_flat_graph()  # noqa: ERA001

    # ccz_bloch_graph()  # noqa: ERA001

    # ccz_phase_graph()  # noqa: ERA001

    # decoherence_simulation_cz()  # noqa: ERA001

    # decoherence_simulation_ccz()  # noqa: ERA001

    # decoherence_simulation("resonant")  # noqa: ERA001

    # print(realistic_cz_decay(10))  # noqa: ERA001

    # grover([0,0,0,0,0,0,0,1])  # noqa: ERA001

    trial = System()

    t3 = Circuit(trial, [1,1,1,1,1,1,1,1])
    print(t3.output())  # noqa: T201
    # t3.p_gate([0], np.pi)  # noqa: ERA001
    # t3.measure_qubits([0,1,2])  # noqa: ERA001
    t3.hadamard_gate([2])
    # t3.rz_gate([1], 2*np.pi)  # noqa: ERA001
    # t3.rx_gate([0,1,2], 2*np.pi)  # noqa: ERA001
    # t3.ccz_gate([0,1,2])  # noqa: ERA001

    print(t3.output())  # noqa: T201
    # t3.measure_qubits([0,1,2])  # noqa: ERA001
    t3.graph_bloch(2)


if __name__ == "__main__":
    main()
