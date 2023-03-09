# v0.8.1

Breaking changes:

* Method `QTensor.values_and_spaces()` is renamed to `QTensor.values_and_slices()`
* Function `utils.structured_iter()` is renamed to `utils.iter_structure()`, function `utils.structured_map()` is renamed to `utils.map_structure()`

Improvements:

* Added methods `StateTensor.norm_and_normalize()` and `StateTensor.component()`
* Enhanced `QTensor.values_and_slices()` (renamed from `QTensor.values_and_spaces`), allowing users to provide a pair of `Space` and index to specify the slice.
* Implemented methods `PureStateTensor.trace()`, `PureStateTensor.amplitudes()`, `PureStateTensor.probabilities()` and `MixedStateTensor.probabilities()`
* Added method `StateTensor.remain()` and property `StateTensor.ket_spaces`

Bug fixes:

* Fixed a bug in `QTensor.__getitem__()` which remains the specified spaces but not remove them
* Fixed wrong type hint of method `Backend.slice()` and `StateTensor.trace()`

# v0.8.0

This is a major release containing a lot of redesigns and breaking changes.

* Reimplemented `QTensor`, changed some APIs.
* Added "special" subclasses of `QTensor`: `NumericTensor`, `PureStateTensor`, `MixedStateTensor`, `OperatorTensor`.
* Added `Backend` to handle specific ndarray operations. Removed the "sparse" backend. Added a tensorflow backend.
* Added `QState` and `QModel`, which can represent an entity, which is a `QSpace` and has a `QState`.
* Removed packages: `braandket.evolve`, `braandket.evolution`, `braandket.kernel`, `braandket.pruning`
  (they are planned to be maintained as a separated project)

