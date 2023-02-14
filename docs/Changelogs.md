# v0.8.0

This is a major release containing a lot of redesigns and breaking changes.

* Reimplemented `QTensor`, changed some APIs.
* Added "special" subclasses of `QTensor`: `NumericTensor`, `PureStateTensor`, `MixedStateTensor`, `OperatorTensor`.
* Added `Backend` to handle specific ndarray operations. Removed the "sparse" backend. Added a tensorflow backend.
* Added `QState` and `QModel`, which can represent an entity, which is a `QSpace` and has a `QState`.
* Removed packages: `braandket.evolve`, `braandket.evolution`, `braandket.kernel`, `braandket.pruning`
  (they are planned to be maintained as a separated project)

