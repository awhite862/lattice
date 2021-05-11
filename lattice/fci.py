import numpy
import itertools
import logging


#def binom(a, b):
#    """ Return binomial coefficient 'a choose b'"""
#    if a < 0 or b < 0:
#        return None
#    elif a < b:
#        return 0
#    elif a == b:
#        return 1
#    else:
#        return factorial(a) // factorial(b) // factorial(a - b)


class FCISimple(object):
    def __init__(self, model, nelec, m_s=None):
        self.model = model
        self.nelec = nelec
        self.norb = model.get_dim()
        self.m_s = m_s
        logging.warning("FCISimple only works in for certain cases, beware!")
        if self.norb > 16:
            raise Exception("This code cannot handle more than 8 sites")

        seed = [i for i in range(self.norb)]
        combs = list(itertools.combinations(seed, nelec))
        if self.m_s is None:
            self.basis = numpy.asarray(combs)
            self.k = self.basis.shape[0]
        else:
            new = []
            for state in combs:
                if self._get_m_s(state) == self.m_s:
                    new.append(state)
            self.basis = numpy.asarray(new)
            self.k = self.basis.shape[0]

    def _get_m_s(self, state):
        N = self.model.N
        m_s = 0
        for x in state:
            d = 1 if x < N else -1
            m_s = m_s + d
        return m_s

    def print_basis(self):
        k, nelec = self.basis.shape
        assert(k == self.k)
        assert(nelec == self.nelec)
        out = str()
        for i in range(k):
            sss = "|"
            for j in range(nelec):
                sss = sss + str(self.basis[i, j])
                if j < nelec - 1:
                    sss = sss + " "

            sss = sss + ">" + " m_s = " + str(self._get_m_s(self.basis[i])) + "\n"
            out += sss
        return out

    def _get_matrixel(self, istate, jstate, U, T):
        iset = set(istate)
        jset = set(jstate)
        ii = len(iset)
        jj = len(jset)
        assert(ii == jj)
        common = iset & jset
        di = iset - common
        dj = jset - common
        assert(len(di) == len(dj))
        ndiff = len(di)
        if ndiff == 0:
            m = 0.0
            for iel in iset:
                m += T[iel, iel]
                for jel in iset:
                    m += 0.5*(U[iel, jel, iel, jel] - U[iel, jel, jel, iel])
            return m
        elif ndiff == 1:
            i1 = list(di)[0]
            j1 = list(dj)[0]
            m = T[i1, j1]
            for x in common:
                m += (U[i1, x, j1, x] - U[i1, x, x, j1])
            ipos = numpy.argwhere(istate == i1)
            jpos = numpy.argwhere(jstate == j1)
            sign = (ipos[0, 0] - jpos[0, 0]) % 2
            return m if sign == 0 else -m
        elif ndiff == 2:
            i1 = list(di)[0]
            i2 = list(di)[1]
            j1 = list(dj)[0]
            j2 = list(dj)[1]
            ipos1 = numpy.argwhere(istate == i1)
            ipos2 = numpy.argwhere(istate == i2)
            jpos1 = numpy.argwhere(jstate == j1)
            jpos2 = numpy.argwhere(jstate == j2)
            s1 = 1 if (ipos1[0, 0] - jpos1[0, 0]) % 2 == 0 else -1
            s2 = 1 if (ipos2[0, 0] - jpos2[0, 0]) % 2 == 0 else -1
            return s1*s2*(U[i1, i2, j1, j2] - U[i1, i2, j2, j1])
        else:
            return 0.0

    def getH(self, phase=None):
        k, nelec = self.basis.shape
        assert(k == self.k)
        assert(nelec == self.nelec)
        U = self.model.get_umat()
        T = self.model.get_tmat(phase=phase)
        if phase is None:
            H = numpy.zeros((k, k))
        else:
            H = numpy.zeros((k, k), dtype=complex)

        for i in range(k):
            for j in range(k):
                istate = self.basis[i]
                jstate = self.basis[j]
                H[i, j] = self._get_matrixel(istate, jstate, U, T)
        return H

    def run(self):
        H = self.getH()
        e, v = numpy.linalg.eigh(H)

        return e, v
