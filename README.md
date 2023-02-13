# BraAndKet

[![License](https://img.shields.io/github/license/ZhengKeli/BraAndKet)](https://github.com/ZhengKeli/BraAndKet/blob/master/LICENSE)

BraAndKet is a library for numeral calculations of discrete quantum systems.

# Quickstart

## Before Using

Please notice that this library is still actively developing. The stability and compatibility of APIs are **NOT**
guaranteed. Breaking changes are happening every day! Using this library right now, you may take your own risk.

## Installation

You can install the latest release from [PiPy](https://pypi.org/project/BraAndKet/).

```shell
pip install braandket
```

Then you can import this library with name `bnk`

```python
import braandket as bnk
```

## KetSpace

Any quantum states can exist in some space called _Hilbert space_. You can use `bnk.KetSpace(n)` to define such a space,
where `n` is its dimension. For example, to create a Hilbert space of a q-bit:

```python
qbit = bnk.KetSpace(2)
print(qbit)  # output: KetSpace(2)
```

You can define a name for a space using named parameter. The name is to describe this space when debugging. The name can
be a `str`, or any object to be printed out. When printed, the name of space will be shown, which is very helpful when
debugging.

```python
qbit_a = bnk.KetSpace(2, name="a")
print(qbit_a)  # output: KetSpace(2, name=a)

qbit_b = bnk.KetSpace(2, name="b")
print(qbit_b)  # output: KetSpace(2, name=b)
```

You can call these 4 methods on a `KetSpace` instance to create ket vectors and operators:

* method `.eigenstate(k)` - to get a ket vector, representing the k-th
  eigenstate ![](https://latex.codecogs.com/svg.latex?|k\\rangle)
* method `.identity()` - to get an identity operator ![](https://latex.codecogs.com/svg.latex?I) in this Hilbert space
* method `.operator(k,b)` - to get an operator ![](https://latex.codecogs.com/svg.latex?|k\\rangle\\langle%20b|)
* method `.projector(k)` - to get a projector ![](https://latex.codecogs.com/svg.latex?|k\\rangle\\langle%20k|)

```python
ket_space = bnk.KetSpace(2)

ket_vec = ket_space.eigenstate(0)
identity_op = ket_space.identity()
increase_op = ket_space.operator(1, 0)
zero_proj = ket_space.projector(0)
```

A `KetSpace` is accompanied by a `BraSpace`. You can conveniently get it with `.ct` property. To avoid confusion, is not
allowed to create any vectors or operations with a `BraSpace`. Please do so with its corresponding `KetSpace`.
Calling `.ct` property, you can get back its `KetSpace`.

```python
ket_space = bnk.KetSpace(2)
print(ket_space)  # output: KetSpace(2)

bra_space = ket_space.ct
print(bra_space)  # output: BraSpace(2)

print(bra_space.ct is ket_space)  # output: True
```

## QTensors

`QTensor` is the basic type of computing elements in this library. A `QTensor` instance holds an `np.ndarray` as its
values and a tuple of `Space` instances. Each `Space` corresponds to an axis of the `np.ndarray`.

Any vectors, operators and tensors in quantum world are represented by `QTensor`. All vectors and operators mentioned
above are all `QTensor` instances.

```python
ket_space = bnk.KetSpace(2)

ket_vec = ket_space.eigenstate(0)
print(ket_vec)
# output: QTensor(spaces=(KetSpace(2),), values=[1. 0.])

identity_op = ket_space.identity()
print(identity_op)
# output: QTensor(spaces=(KetSpace(2), BraSpace(2)), values=[[1. 0.] [0. 1.]])

increase_op = ket_space.operator(1, 0)
print(increase_op)
# output: QTensor(spaces=(KetSpace(2), BraSpace(2)), values=[[0. 0.] [1. 0.]])

zero_proj = ket_space.projector(0)
print(zero_proj)
# output: QTensor(spaces=(KetSpace(2), BraSpace(2)), values=[[1. 0.] [0. 0.]])
```

You can easily get a conjugate transposed `QTensor` calling `.ct` property. It should be noted that sometimes, such
operation does not affect the values, but spaces.

```python
ket_space = bnk.KetSpace(2)

ket_vec = ket_space.eigenstate(0)
bra_vec = ket_vec.ct
print(bra_vec)
# output: QTensor(spaces=(BraSpace(2),), values=[1. 0.])

increase_op = ket_space.operator(1, 0)
decrease_op = increase_op.ct
print(decrease_op)
# output: QTensor(spaces=(BraSpace(2), KetSpace(2)), values=[[0. 0.] [1. 0.]])
```

`QTensor` instances can take tensor product using `@` operator. They can automatically inspect which spaces to be
performed the "product-sum" (when the bra on the left meets the matching ket on the right), which to be remained.

### Example1:

![](https://latex.codecogs.com/svg.latex?\\langle0|\\cdot|1\\rangle=\\langle0|1\\rangle=0)

```python
qbit = bnk.KetSpace(2)

amp = qbit.eigenstate(0).ct @ qbit.eigenstate(1)
print(amp)
# output: QTensor(spaces=(), values=0.0)
```

### Example2:

![](https://latex.codecogs.com/svg.latex?|0\\rangle_a\\cdot|1\\rangle_b=|0\\rangle_a|1\\rangle_b)

```python
qbit_a = bnk.KetSpace(2, name="a")
qbit_b = bnk.KetSpace(2, name="b")

ket_vec_ab = qbit_a.eigenstate(0) @ qbit_b.eigenstate(1)
print(ket_vec_ab)
# output: QTensor(spaces=(KetSpace(2, name=a), KetSpace(2, name=b)), values=[[0. 1.] [0. 0.]])
```

### Example3:

![](https://latex.codecogs.com/svg.latex?\\langle0|_a\\cdot|1\\rangle_b=\\langle0|_a|1\\rangle_b)

```python
qbit_a = bnk.KetSpace(2, name="a")
qbit_b = bnk.KetSpace(2, name="b")

tensor_ab = qbit_a.eigenstate(0).ct @ qbit_b.eigenstate(1)
print(tensor_ab)
# output: QTensor(spaces=(BraSpace(2, name=a), KetSpace(2, name=b)), values=[[0. 1.] [0. 0.]])
```

### Example4:

![](https://latex.codecogs.com/svg.latex?A_%7Binc%7D%3D%5Cleft%20%7C%201%20%5Cright%20%5Crangle%20%5Cleft%20%5Clangle%200%20%5Cright%20%7C%20%3D%20%5Cbegin%7Bpmatrix%7D%200%20%26%200%5C%5C%201%20%26%200%20%5Cend%7Bpmatrix%7D)

![](https://latex.codecogs.com/svg.latex?A_{inc}|0\\rangle=|1\\rangle)

```python
qbit = bnk.KetSpace(2)

ket_vec_0 = qbit.eigenstate(0)
ket_vec_1 = qbit.eigenstate(1)
increase_op = qbit.operator(1, 0)
result = increase_op @ ket_vec_0

print(result)
# output: QTensor(spaces=(KetSpace(2),), values=[0. 1.])

print(result == ket_vec_1)
# output: True

```

# Contribution

This library is completely open source. Any contributions are welcomed. You can fork this repository, make some useful
changes and then send a pull request to me on GitHub.
