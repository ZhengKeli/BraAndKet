# v0.8.6.post1

Bug fixes:

* Fixed the error that occurs when giving a tuple of two `KetSpace` to `StateTensor.measure()`.

Improvements:

* Optimized the signature of `StateTensor.measure()`.

# v0.8.6

Breaking changes:

* Removed `Backend.take()` and `Backend.choose()`, and also corresponding functions `bnk.choose()` and `bnk.take()`.

New features:

* Added method `measure()` in `StateTensor`, along with method `measure_pure_state()` and `measure_mixed_state()`
  in `Backend`.

Bug fixes:

* Replaced `|` with `Union` to avoid `SyntaxError` in lower python versions.

Improvements:

* Added the calling of `backend.abs()` in `PureStateTensor.probabilities()`.

# v0.8.5

Breaking changes:

* Removed subpackage `braandket.model`. It will be maintained in braandket-circuit.
* Deleted 4 methods in `QTensor`:
    * `as_numeric_tensor()`
    * `as_pure_state_tensor()`
    * `as_mixed_state_tensor()`
    * `as_operator_tensor()`

Bug fixes:

* Fixed critical bugs in `Backend.trace()` and `Backend.diag()`.
* Fixed critical bug in `MixedStateTensor.trace()`.

Improvements:

* Exposed `ArrayLike`.
* Added `Backend.compact()`.
* Added auto-converting of `PureStateTensor` in `MixedStateTensor.of()`.

# v0.8.4

Breaking changes:

* Renamed `ValuesType` to `BackendValue`.
* (For custom backend) All `Backend` operations should automatically convert arguments.

Improvements:

* Optimized dtype converting in `TensorflowBackend`.
* All `Backend` operations now automatically convert arguments.
* renamed variables in `Backend`: "values" to "value".

# v0.8.3

Improvements:

* Allowed `QTensor.inflate()` to process any shape.
* Allowed `OperatorTensor.from_matrix()` to accept a single space.

Bug fixes:

* Fixed a wrong type hint in `OperatorTensor.from_matrix()`.

Development:

* Reformat code with new rules.
* Optimized code in `braandket.tensor.operations`.

# v0.8.2

Bug fixes:

* changed python requirement to 3.8 to avoid the error: "'type' object is not subscriptable"

Development:

* Optimized GitHub workflow

# v0.8.1

Breaking changes:

* Method `QTensor.values_and_spaces()` is renamed to `QTensor.values_and_slices()`
* Function `utils.structured_iter()` is renamed to `utils.iter_structure()`, function `utils.structured_map()` is
  renamed to `utils.map_structure()`

Improvements:

* Added methods `StateTensor.norm_and_normalize()` and `StateTensor.component()`
* Enhanced `QTensor.values_and_slices()` (renamed from `QTensor.values_and_spaces`), allowing users to provide a pair
  of `Space` and index to specify the slice
* Implemented methods `PureStateTensor.trace()`, `PureStateTensor.amplitudes()`, `PureStateTensor.probabilities()`
  and `MixedStateTensor.probabilities()`
* Added method `StateTensor.remain()` and property `StateTensor.ket_spaces`

Bug fixes:

* Fixed a bug in `QTensor.__getitem__()` which remains the specified spaces but not remove them
* Fixed wrong type hint of method `Backend.slice()` and `StateTensor.trace()`

# v0.8.0

This is a major release containing a lot of redesigns and breaking changes.

* Reimplemented `QTensor`, changed some APIs
* Added "special" subclasses of `QTensor`: `NumericTensor`, `PureStateTensor`, `MixedStateTensor`, `OperatorTensor`
* Added `Backend` to handle specific ndarray operations. Removed the "sparse" backend. Added a tensorflow backend
* Added `QState` and `QModel`, which can represent an entity, which is a `QSpace` and has a `QState`
* Removed packages: `braandket.evolve`, `braandket.evolution`, `braandket.kernel`, `braandket.pruning`
  (they are planned to be maintained as a separated project)

