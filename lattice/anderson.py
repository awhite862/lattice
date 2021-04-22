import numpy
from cqcpy import utils


class Anderson(object):
    """Single-site Anderson impurity model.

    Attribute:
        ll (int): Number of sites in the left lead
        lr (int): Number of sites in the right lead
        t (float): hopping parameter
        td (float): coupling between dot and leads
        U (float): dot repulsion
        V (float): potential
        Vg (float): gate voltage
        u (float): normalized dot repulsion (U/4t)
        v (float): normalized potential (V/t)
        vg (float): normalized gate voltage (Vg/t)
    """
    def __init__(self, ll, lr, t, td, U, V, Vg):
        """Initialize Anderson impurity model.

        Args:
            ll (int): Number of sites in the left lead
            lr (int): Number of sites in the right lead
            t (float): hopping parameter
            td (float): hopping parameter to leads
            U (float): dot repulsion
            V (float): potential
            Vg (float): gate voltage
        """
        self.ll = ll
        self.lr = lr
        self.t = t
        self.td = td
        self.tdr = td/t
        self.U = U
        self.V = V
        self.Vg = Vg
        self.v = V/t
        self.vg = Vg/t
        self.u = U/(4.0*t)

    def get_dim(self):
        return 2*(self.ll + self.lr + 1)

    def get_tmatS(self):
        N = self.ll + self.lr + 1
        t = numpy.zeros((N,N))
        idot = self.ll
        for i in range(N):
            if i < N - 1:
                dot = (i == idot or i + 1 == idot)
                t[i,i + 1] = -self.tdr if dot else -1.0
            if i > 0:
                dot = (i == idot or i - 1 == idot)
                t[i,i - 1] = -self.tdr if dot else -1.0
        return t

    def get_tmat(self):
        t = self.get_tmatS()
        return utils.block_diag(t, t)

    def get_vmatS(self):
        N = self.ll + self.lr + 1
        v = numpy.zeros((N,N))
        for i in range(self.ll):
            v[i,i] = self.v/2
        off = self.ll + 1
        v[self.ll, self.ll] = self.vg
        for i in range(self.lr):
            v[off + i,off + i] = -self.v/2
        return v

    def get_vmat(self):
        v = self.get_vmatS()
        return utils.block_diag(v, v)

    def get_umatS(self):
        N = self.ll + self.lr + 1
        idot = self.ll
        umat = numpy.zeros((N,N,N,N))
        umat[idot,idot,idot,idot] = 4.0*self.u
        return umat

    def get_umat(self):
        N = self.ll + self.lr + 1
        idot = self.ll
        umat = numpy.zeros((2*N,2*N,2*N,2*N))
        umat[idot,N + idot,idot,N + idot] = 4.0*self.u
        umat[N + idot,idot,N + idot,idot] = 4.0*self.u
        umat[idot,idot,idot,idot] = 4.0*self.u
        umat[N + idot,N + idot,N + idot,N + idot] = 4.0*self.u
        return umat
