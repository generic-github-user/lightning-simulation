# Lightning Simulation

A simple lightning simulator and renderer written in Python. It does not aim to be scientifically accurate, only to produce visually interesting and somewhat lifelike images of simulated lightning with a reasonable degree of customizability.

## Field-Based Method

The program uses basic fluid simulation principles; the simulation is modeled as an electrical potential field that dictates where current travels through a medium (e.g., the atmosphere). The lightning itself is represented as a series of paths taken by particles (electric charge) during a discrete-time simulation. The particles' motion is thus determined by their position in the previous timestep, factors influencing the electric field (essentially a function of the particle's position), and some random noise (hence the "stochastic"). (I have probably committed flagrant abuse of technical terminology, but hopefully this communicates the main points somewhat clearly).

The resulting motion is chaotic; a large set of particles with only slightly varying initial positions will eventually diverge, creating the characteristic branching effect. Random noise can also be applied over the course of a particle's motion to adjust the rate at which particles will fall into random "wells", thereby creating branches.