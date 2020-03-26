class KetDimension:
    def __init__(self, n, name=None, key=None):
        self.n = n
        self.name = name
        self.key = key
    
    @property
    def ct(self):
        return BraDimension(self)
    
    def __eq__(self, other):
        return isinstance(other, KetDimension) and (self is other)
    
    def __repr__(self):
        return f"<KetDimension name={self.name}, dim={self.n}, id={id(self)}>"


class BraDimension:
    def __init__(self, ket: KetDimension):
        self._ket = ket
    
    @property
    def ket(self):
        return self._ket
    
    @property
    def n(self):
        return self.ket.n
    
    @n.setter
    def n(self, value):
        self.ket.n = value
    
    @property
    def name(self):
        return self.ket.name
    
    @name.setter
    def name(self, name):
        self.ket.name = name
    
    @property
    def key(self):
        return self.ket.key
    
    @key.setter
    def key(self, key):
        self.ket.key = key
    
    @property
    def ct(self):
        return self.ket
    
    def __eq__(self, other):
        return isinstance(other, BraDimension) and (self.ket is other.ket)
    
    def __repr__(self):
        return f"<BraDimension name={self.name}, dim={self.n}, id={id(self)}>"
