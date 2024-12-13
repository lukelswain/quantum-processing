INTRODUCTION
Script provides a method to compute the time evolution of a system of a two qubit system, with an excited state modelled as a Rydberg state of one of the neutral atoms. Also provided are methods to then graph this evolution over time, with adjustable physical parameters.

INPUTS
1-strength of interaction between simultaneous Rydberg states
2-strength of interaction between applied oscillating external electric field and an atom
3-detuning
4-function to vary detuning over time, with examples provided
5-inital phase of oscillating external electric field

6-time range over which the states are evolved, with variable timestep

7-initial state
8-desired state
    -basis = {|00>, |01>, |10>, |11>}

OUTPUTS
(1-5)-matrix representation of the hamiltonian of the system in the rotating frame, using the rotating wave approximation, at a time t
(1-5)&(7)-evolution of given initial state to a time t determined by the system's hamiltonian
(1-5)&(7-8)-probability of system being in a desired state at a time t
(1-8)-plot of probability for any given desired and initial states, over the specified time range
(1-8)-comparison plot of analytical solution and numerical solution for a given state over the time range
(1-8)-timer decorator that returns the value of the method and the excecution time, as well as a grapher function that plots a graph with increasing number of timesteps

N.B.
-only if hamiltonian time independent (detuning function chosen to be constant in time) can analytical solution be found, comparison plot method needs H to have this property
-to be computationally efficient, the numerical time evolution method takes the entire array containing the time range broken into timesteps, and outputs an array of equal size, of the state of the system at each of these times. The method that gives the state of the system at a time t using the numerical evolution method simply takes an index of this output array. It can therefore only take as a time input a value that is a value within the time array, so that it can be matched to an index within this array, and that index input into the evolved states array.


