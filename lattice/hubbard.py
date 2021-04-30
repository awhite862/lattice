import numpy
from cqcpy import utils


class Hubbard1D(object):
    """One dimensional Hubbard model.

    Attributes:
        L (int): Number of sites.
        t (float): Hubbard t (hopping) parameter.
        U (float): Hubbard U (on-site repulsion) parameter.
        u (float): reduced hubbard U-parameter (u = U/4t).
    """
    def __init__(self, L, t, U, boundary='p'):
        """Initialize 1D Hubbard model.

        Args:
            L (int): Number of sites.
            t (float): Hubbard t (hopping) parameter.
            U (float): Hubbard U (on-site repulsion) parameter.
        """
        self.bc = boundary
        self.L = L
        self.t = t
        self.U = U
        self.u = U/(4.0*t)

    def get_dim(self):
        """Return spin-orbital dimension."""
        return 2*self.L

    def _get_nn(self,i):
        L = self.L
        if self.bc == "p" or self.bc == "pbc":
            l = L - 1 if i == 0 else i - 1
            r = 0 if i == (L - 1) else i + 1
        else:
            l = i + 1 if i == 0 else i - 1
            r = i - 1 if i == (L - 1) else i + 1
        return (l,r)

    def get_tmatS(self, phase=None):
        """ Return T-matrix in the spatial orbital basis."""
        L = self.L
        dtype = float if phase is None else complex
        t = numpy.zeros((L,L), dtype=dtype)
        for i in range(L):
            nn = self._get_nn(i)
            if phase is None:
                t[i,nn[0]] = -1.0
                t[i,nn[1]] = -1.0
            else:
                if nn[0] > i:
                    t[i,nn[0]] = -1.0*numpy.exp(1.j*phase)
                else:
                    t[i,nn[0]] = -1.0*numpy.exp(-1.j*phase)
                if nn[1] > i:
                    t[i,nn[1]] = -1.0*numpy.exp(1.j*phase)
                else:
                    t[i,nn[1]] = -1.0*numpy.exp(-1.j*phase)
        return t

    def get_tmat(self, phase=None):
        """ Return T-matrix in the spin orbital basis."""
        t = self.get_tmatS(phase=phase)
        return utils.block_diag(t, t)

    def get_umatS(self):
        """ Return U-matrix (not antisymmetrized) in the
        spatial-orbital basis."""
        L = self.L
        umat = numpy.zeros((L,L,L,L))
        for i in range(L):
            umat[i,i,i,i] = 4.0*self.u
        return umat

    def get_umat(self):
        """ Return U-matrix (not antisymmetrized) in the
        spin-orbital basis."""
        L = self.L
        umat = numpy.zeros((2*L,2*L,2*L,2*L))
        for i in range(L):
            umat[i,L + i,i,L + i] = 4.0*self.u
            umat[L + i,i,L + i,i] = 4.0*self.u
            umat[i,i,i,i] = 4.0*self.u
            umat[L + i,L + i,L + i,L + i] = 4.0*self.u
        return umat


class Hubbard2D(object):
    """Two dimensional Hubbard model plaquette.

    Attributes:
        N (int): Total number of sites.
        t (float): Hubbard t (hopping) parameter.
        U (float): Hubbard U (on-site repulsion) parameter.
        u (float): reduced hubbard U-parameter (u = U/4t).
    """

    def __init__(self, N, t, U, nn):
        """Initialize 2D Hubbard model.

        Args:
            N (int): Number of sites.
            t (float): Hubbard t (hopping) parameter.
            U (float): Hubbard U (on-site repulsion) parameter.
        """
        self.N = N
        self.t = t
        self.U = U
        self.u = U/(4.0*t)
        self.nn = nn

    def get_dim(self):
        """Return spin-orbital dimension."""
        return 2*self.N

    def get_tmatS(self):
        """ Return T-matrix in the spatial orbital basis."""
        N = self.N
        t = numpy.zeros((N,N))
        for i in range(N):
            nn = self.nn[i]
            for x in nn:
                t[i,x] -= self.t/2.0
                t[x,i] -= self.t/2.0
        return t

    def get_tmat(self):
        """ Return T-matrix in the spin orbital basis."""
        t = self.get_tmatS()
        return utils.block_diag(t, t)

    def get_umatS(self):
        """ Return U-matrix (not antisymmetrized) in the
        spatial-orbital basis."""
        N = self.N
        umat = numpy.zeros((N,N,N,N))
        for i in range(N):
            umat[i,i,i,i] = 4.0*self.u
        return umat

    def get_umat(self):
        """ Return U-matrix in the spin orbital basis."""
        N = self.N
        umat = numpy.zeros((2*N,2*N,2*N,2*N))
        for i in range(N):
            umat[i, N + i,i,N + i] = 4.0*self.u
            umat[N + i,i,N + i,i] = 4.0*self.u
            umat[i,i,i,i] = 4.0*self.u
            umat[N + i,N + i,N + i,N + i] = 4.0*self.u
        return umat


class Hubbard3D(object):
    """Three dimensional Hubbard model.

    Attributes:
        N (int): Total number of sites.
        t (float): Hubbard t (hopping) parameter.
        U (float): Hubbard U (on-site repulsion) parameter.
        u (float): reduced hubbard U-parameter (u = U/4t).
    """

    def __init__(self, N, t, U, nn):
        """Initialize 2D Hubbard model.

        Args:
            N (int): Number of sites.
            t (float): Hubbard t (hopping) parameter.
            U (float): Hubbard U (on-site repulsion) parameter.
        """
        self.N = N
        self.t = t
        self.U = U
        self.u = U/(4.0*t)
        self.nn = nn

    def get_dim(self):
        """Return spin-orbital dimension."""
        return 2*self.N

    def get_tmatS(self):
        """ Return T-matrix in the spatial orbital basis."""
        N = self.N
        t = numpy.zeros((N,N))
        for i in range(N):
            nn = self.nn[i]
            for x in nn:
                t[i,x] -= self.t/2
                t[x,i] -= self.t/2
        return t

    def get_tmat(self):
        """ Return T-matrix in the spin orbital basis."""
        t = self.get_tmatS()
        return utils.block_diag(t, t)

    def get_umatS(self):
        """ Return U-matrix (not antisymmetrized) in the
        spatial-orbital basis."""
        N = self.N
        umat = numpy.zeros((N,N,N,N))
        for i in range(N):
            umat[i,i,i,i] = 4.0*self.u
        return umat

    def get_umat(self):
        """ Return U-matrix in the spin orbital basis."""
        N = self.N
        umat = numpy.zeros((2*N,2*N,2*N,2*N))
        for i in range(N):
            umat[i,N + i,i,N + i] = 4.0*self.u
            umat[N + i,i,N + i,i] = 4.0*self.u
            umat[i,i,i,i] = 4.0*self.u
            umat[N + i,N + i,N + i,N + i] = 4.0*self.u
        return umat
