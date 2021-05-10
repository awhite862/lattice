import numpy
from . import utils


class HubbardBase(object):
    """Generic Hubbard model."""
    def __init__(self, N, t, U, nn):
        """Initialize 2D Hubbard model.

        Args:
            N (int): Number of sites.
            t (float): Hubbard t (hopping) parameter.
            U (float): Hubbard U (on-site repulsion) parameter.
            nn (list): List of nearest neighbors.
        """
        self.N = N
        self.t = t
        self.U = U
        self.u = U/(4.0*t)
        self.nn = nn

    def get_dim(self):
        """Return spin-orbital dimension."""
        return 2*self.N

    def get_tmatS(self, phase=None):
        """ Return T-matrix in the spatial orbital basis."""
        N = self.N
        dtype = float if phase is None else complex
        t = numpy.zeros((N, N), dtype=dtype)
        for i in range(N):
            nn = self.nn[i]
            for x in nn:
                if phase is None:
                    t[i, x] -= self.t/2
                    t[x, i] -= self.t/2
                elif x > i:
                    t[i, x] -= numpy.exp(1.j*phase)*self.t/2
                    t[x, i] -= numpy.exp(-1.j*phase)*self.t/2
                else:
                    assert(x < i)
                    t[i, x] -= numpy.exp(-1.j*phase)*self.t/2
                    t[x, i] -= numpy.exp(1.j*phase)*self.t/2
        return t

    def get_tmat(self, phase=None):
        """ Return T-matrix in the spin orbital basis."""
        t = self.get_tmatS(phase=phase)
        return utils.block_diag(t, t)

    def get_umatS(self):
        """ Return U-matrix (not antisymmetrized) in the
        spatial-orbital basis."""
        N = self.N
        umat = numpy.zeros((N, N, N, N))
        for i in range(N):
            umat[i, i, i, i] = self.U
        return umat

    def get_umat(self):
        """ Return U-matrix in the spin orbital basis."""
        N = self.N
        umat = numpy.zeros((2*N, 2*N, 2*N, 2*N))
        for i in range(N):
            umat[i, N + i, i, N + i] = self.U
            umat[N + i, i, N + i, i] = self.U
            umat[i, i, i, i] = self.U
            umat[N + i, N + i, N + i, N + i] = self.U
        return umat


def _get_nn_1d(L, boundary):
    nn = []
    for i in range(L):
        if boundary == "p" or boundary == "pbc":
            l = L - 1 if i == 0 else i - 1
            r = 0 if i == (L - 1) else i + 1
            x = (l, r)
        else:
            if i == 0:
                x = (i + 1,)
            elif i == (L - 1):
                x = (i - 1,)
            else:
                l = i - 1
                r = i + 1
                x = (l, r)
        nn.append(x)
    return nn


class Hubbard1D(HubbardBase):
    """One dimensional Hubbard model.

    Attributes:
        L (int): Number of sites.
        t (float): Hubbard t (hopping) parameter.
        U (float): Hubbard U (on-site repulsion) parameter.
        u (float): reduced hubbard U-parameter (u = U/4t).
    """
    def __init__(self, L, t, U, boundary='p', lattice=None):
        """Initialize 1D Hubbard model.

        Args:
            L (int): Number of sites.
            t (float): Hubbard t (hopping) parameter.
            U (float): Hubbard U (on-site repulsion) parameter.
        """
        # lattice is specified by keyword
        if isinstance(boundary, str):
            if lattice is not None:
                raise Exception("lattice and boundary both specified")
            nn = _get_nn_1d(L, boundary)
        # lattice is specified explicitly
        else:
            nn = lattice
        HubbardBase.__init__(self, L, t, U, nn)


class Hubbard2D(HubbardBase):
    def __init__(self, N, t, U, lattice):
        """Initialize 2D Hubbard model.

        Args:
            N (int): Number of sites.
            t (float): Hubbard t (hopping) parameter.
            U (float): Hubbard U (on-site repulsion) parameter.
            lattice: Specify lattice by string or list of nearest neighbors.
        """
        # lattice is specified by keyword
        if isinstance(lattice, str):
            raise Exception("Unrecognized lattice keyword!")
        # lattice is specified explicitly
        else:
            HubbardBase.__init__(self, N, t, U, lattice)


class Hubbard3D(HubbardBase):
    def __init__(self, N, t, U, lattice):
        """Initialize 3D Hubbard model.

        Args:
            N (int): Number of sites.
            t (float): Hubbard t (hopping) parameter.
            U (float): Hubbard U (on-site repulsion) parameter.
            lattice: Specify lattice by string or list of nearest neighbors.
        """
        # lattice is specified by keyword
        if isinstance(lattice, str):
            raise Exception("Unrecognized lattice keyword!")
        # lattice is specified explicitly
        else:
            HubbardBase.__init__(self, N, t, U, lattice)
