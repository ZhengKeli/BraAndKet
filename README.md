# BraAndKet

BraAndKet is a library for a convenient representation of discrete quantum systems and their evolution.



# Representation

## HSpace
The class `HSpace` represents a Hilbert Space.
For example, to represent a 

## QDimension

## QTensor

# Evolution



# Reduction

Sometimes, the space of system can be terribly big, since the space the space increases exponentially with the increase of the count of components. 

But in some cases, we just want to study the evolution of the system under certain conditions, for example from several specified start points evolves with some certain operators. Then, some states are  in fact impossible to be reached. Then those unreachable states can be dropped out of the computation. 

Class `ReducedHSpace` is designed for such cases. 

The static method `ReducedHSpace.from_seed()` can automatically detect which eigenstates can be dropped, with the given starting states and evolution operators, and return an instance of `ReducedHSpace` as a "reachable" space. This can significantly reduce the calculation and memory consumption.

The reduced and original tensors can also be easily converted to each other using method `reduce()` and `inflate()`.
