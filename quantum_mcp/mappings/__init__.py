import itertools
import numpy as np

from cudaq import spin

from quantum_mcp.preprocess import MolecularHamiltonian


class JordanWignerFermion:
    def __init__(self, molecular_hamiltonian:MolecularHamiltonian):
        self.molecular_hamiltonian = molecular_hamiltonian
        self.spin_hamiltonian = None

    def get_spin_hamiltonian(self, tolerance=1e-12):
        spin_hamiltonian = self.molecular_hamiltonian.core_energy
        h_pq = self.molecular_hamiltonian.obi_coeff
        h_pqrs = self.molecular_hamiltonian.tbi_coeff

        nqubit = self.molecular_hamiltonian.obi_coeff.shape[0]

        for p in range(nqubit):

            # Diagonal one-body term (number operator): sum_p h_{pp} a_p^dagger a_p
            coef = self.molecular_hamiltonian.obi_coeff[p, p]
            if np.abs(coef) > tolerance:
                spin_hamiltonian += self._jordan_wigner_one_body(p, p, coef)

        for p, q in itertools.combinations(range(nqubit), 2):

            # Off-diagonal one-body term (excitation operator): sum_p<q (h_{`pq`} a_p^dagger a_q + h_{`qp`} a_q^dagger a_p)
            coef = 0.5 * (h_pq[p, q] + np.conj(h_pq[q, p]))
            if np.abs(coef) > tolerance:
                spin_hamiltonian += self._jordan_wigner_one_body(p, q, coef)

            # Diagonal term two-body (coulomb and exchange operators)
            # Diagonal term: p=r, q=s or p=s,q=r --> (+ h_`pqpq` = + h_`qpqp` = - h_`qppq` = - h_`pqqp`)

            # exchange operator
            coef = h_pqrs[p, q, p, q] + h_pqrs[q, p, q, p]
            if np.abs(coef) > tolerance:
                spin_hamiltonian += self._jordan_wigner_two_body(p, q, p, q, coef)

            # coulomb operator
            coef = h_pqrs[p, q, q, p] + h_pqrs[q, p, p, q]
            if np.abs(coef) > tolerance:
                spin_hamiltonian += self._jordan_wigner_two_body(p, q, q, p, coef)

        for (p, q), (r, s) in itertools.combinations(
                itertools.combinations(range(nqubit), 2), 2):

            # h_`pqrs` = - h_`qprs` = -h_`pqsr` = h_`qpsr`
            # Four point symmetry if integrals are complex: `pqrs` = `srqp` = `qpsr` = `rspq`
            # Eight point symmetry if integrals are real: `pqrs` = `rqps` = `psrq` = `srqp` = `qpsr` = `rspq` = `spqr` = `qrsp`

            coef = 0.5 * (h_pqrs[p, q, r, s] + np.conj(h_pqrs[s, r, q, p]) - h_pqrs[q, p, r, s] - np.conj(
                h_pqrs[s, r, p, q]) - h_pqrs[p, q, s, r] - np.conj(h_pqrs[r, s, q, p]) + h_pqrs[q, p, s, r] + np.conj(
                        h_pqrs[r, s, p, q]))

            # Compute number with excitation operator and double excitation operator
            if np.abs(coef) > tolerance:
                spin_hamiltonian += self._jordan_wigner_two_body(p, q, r, s, coef)

        # Remove term with zero coefficient.
        spin_hamiltonian = spin_hamiltonian.canonicalize().trim(tolerance)

        return spin_hamiltonian

    @staticmethod
    def _jordan_wigner_one_body(p, q, coef):

        # Diagonal term: 0.5 h_{pp} (I_p - Z_p)
        if p == q:
            spin_hamiltonian = 0.5 * coef * spin.i(p)
            spin_hamiltonian -= 0.5 * coef * spin.z(p)

        # h_`pq`(a_p^dagger a_q + a_q^dagger a_p) = R(h_`pq`) (a_p^dagger a_q + a_q^dagger a_p) +
        #                                          `imag` (h_`pq`) (a_p^dagger a_q - a_q^dagger a_p)
        # Off-diagonal real part: 0.5 * real(h_{`pq`}) [ X_p (Z_{p+1}^{q-1}) X_q + Y_p (Z_{p+1}^{q-1}) Y_q ]
        # Off-diagonal imaginary part: 0.5* `im`(h_`pq`) [y_p (Z_{p+1}^{q-1}) x_q - x_p (Z_{p+1}^{q-1}) y_q]

        else:
            if p > q:
                p, q = q, p
                coef = np.conj(coef)

            # Compute the parity string (Z_{p+1}^{q-1})
            z_indices = [i for i in range(p + 1, q)]
            parity_string = 1.0
            for i in z_indices:
                parity_string *= spin.z(i)

            spin_hamiltonian = 0.5 * coef.real * spin.x(p) * parity_string * spin.x(
                q)
            spin_hamiltonian += 0.5 * coef.real * spin.y(
                p) * parity_string * spin.y(q)
            spin_hamiltonian += 0.5 * coef.imag * spin.y(
                p) * parity_string * spin.x(q)
            spin_hamiltonian -= 0.5 * coef.imag * spin.x(
                p) * parity_string * spin.y(q)

        return spin_hamiltonian

    @staticmethod
    def _jordan_wigner_two_body(p, q, r, s, coef):

        # Diagonal term: p=r, q=s or p=s,q=r --> (+ h_`pqpq` = + h_`qpqp` = - h_`qppq` = - h_`pqqp`)
        #
        # exchange operator:  h_`pqpq` (a_p^dagger a_q^dagger a_p a_q) + h_`qpqp` (a_q^dagger a_p^dagger a_q a_p)
        # p<q: -1/4 (I_p I_q - I_p Z_q - Z_p I_q+Z_p Z_q)
        #
        # coulomb operator: h_`qppq` (a_q^dagger a_p^dagger a_p a_q) + h_`pqqp` (a_p^dagger a_q^dagger a_q a_p)
        # p<q: 1/4 (I_p I_q - I_p Z_q - Z_p I_q + Z_p Z_q)

        if len({p, q, r, s}) == 2:

            if p == r:
                spin_hamiltonian = -0.25 * coef * spin.i(p) * spin.i(q)
                spin_hamiltonian += 0.25 * coef * spin.i(p) * spin.z(q)
                spin_hamiltonian += 0.25 * coef * spin.z(p) * spin.i(q)
                spin_hamiltonian -= 0.25 * coef * spin.z(p) * spin.z(q)

            elif q == r:
                spin_hamiltonian = 0.25 * coef * spin.i(p) * spin.i(q)
                spin_hamiltonian -= 0.25 * coef * spin.i(p) * spin.z(q)
                spin_hamiltonian -= 0.25 * coef * spin.z(p) * spin.i(q)
                spin_hamiltonian += 0.25 * coef * spin.z(p) * spin.z(q)

        # Off-diagonal term with three different sets of non-equal indices
        # Number with excitation operator
        # + h_`pqqs` = + h_`qpsq` = - h_`qpqs` = - h_`pqsq` and their hermitian conjugate
        # Real (h_`pqqs`) (a_p^dagger a_q^dagger a_q a_s + a_s^dagger a_q^dagger a_q a_p) +
        # `imag` (h_`pqqs`) (a_p^dagger a_q^dagger a_q a_s - a_s^dagger a_q^dagger a_q a_p)
        # p <q <s: (1/4)(Z_{p+1}^{s-1}) [ I_q {real (h_`pqqs`/4) (x_p x_s + y_p y_s) + {`imag` (h_`pqqs`/4) (y_p x_s - x_p y_s)}
        #                           - Z_q {real (h_`pqqs`/4) (x_p x_s + y_p y_s) + `imag`(h_`pqqs`) (y_p x_s -x_p y_s)}]

        if len({p, q, r, s}) == 3:

            if q == r:
                if p > r:
                    a, b = s, p
                    coef = np.conj(coef)
                else:
                    a, b = p, s
                c = q

            elif q == s:
                if p > r:
                    a, b = r, p
                    coef = -1.0 * np.conj(coef)
                else:
                    a, b = p, r
                    coef *= -1.0
                c = q

            elif p == r:
                if q > s:
                    a, b = s, q
                    coef = -1.0 * np.conj(coef)
                else:
                    a, b = q, s
                    coef = -1.0 * coef
                c = p

            elif p == s:
                if q > r:
                    a, b = r, q
                    coef = np.conj(coef)
                else:
                    a, b = q, r
                c = p

            parity_string = 1.0
            z_qubit = [i for i in range(a + 1, b)]
            for i in z_qubit:
                parity_string *= spin.z(i)

            spin_hamiltonian = 0.25 * coef.real * spin.x(
                a) * parity_string * spin.x(b) * spin.i(c)
            spin_hamiltonian += 0.25 * coef.real * spin.y(
                a) * parity_string * spin.y(b) * spin.i(c)
            spin_hamiltonian += 0.25 * coef.imag * spin.y(
                a) * parity_string * spin.x(b) * spin.i(c)
            spin_hamiltonian -= 0.25 * coef.imag * spin.x(
                a) * parity_string * spin.y(b) * spin.i(c)

            spin_hamiltonian -= 0.25 * coef.real * spin.x(
                a) * parity_string * spin.x(b) * spin.z(c)
            spin_hamiltonian -= 0.25 * coef.real * spin.y(
                a) * parity_string * spin.y(b) * spin.z(c)
            spin_hamiltonian -= 0.25 * coef.imag * spin.y(
                a) * parity_string * spin.x(b) * spin.z(c)
            spin_hamiltonian += 0.25 * coef.imag * spin.x(
                a) * parity_string * spin.y(b) * spin.z(c)

        # Off-diagonal term with four different sets of non-equal indices
        # h_`pqrs` = h_`qpsr` = - h_`qprs` = - h_`pqsr`
        # real {h_`pqrs`} (a_p^dagger a_q^dagger a_r a_s + a_s^dagger a_r^dagger a_q a_p) +
        # `imag` (h_`pqrs`) (a_p^dagger a_q^dagger a_r a_s - a_s^dagger a_r^dagger a_q a_p)
        # p<q<r<s real part: -1/8 (Z_{p+1}^{q-1}) (Z_{r+1}^{s-1}) (x_p x_q x_r x_s - x_p x_q y_r y_s + x_p y_q x_r y_s + x_p y_q y_r x_s
        #                         + y_p x_q x_r y_s + y_p x_q y_r x_s - y_p y_q x_r x_s + y_p y_q y_r y_s)
        # p<q<r<s `imag` part: -1/8 (x_p x_q x_r y_s + x_p x_q y_r x_s - x_p y_q x_r x_s + x_p y_q y_r y_s
        #                         - y_p x_q x_r x_s + y_p x_q y_r y_s - y_p y_q x_r y_s - y_p y_q y_r x_s)
        # also we need to compute p<r<q<s and p<r<s<q

        elif len({p, q, r, s}) == 4:

            if (p > q) ^ (r > s):
                coef *= -1.0

            if p < q < r < s:
                a, b, c, d = p, q, r, s

                parity_string_a = 1.0
                z_qubit_a = [i for i in range(a + 1, b)]
                for i in z_qubit_a:
                    parity_string_a *= spin.z(i)

                parity_string_b = 1.0
                z_qubit_b = [i for i in range(c + 1, d)]
                for i in z_qubit_b:
                    parity_string_b *= spin.z(i)

                spin_hamiltonian = -0.125 * coef.real * spin.x(
                    a) * parity_string_a * spin.x(b) * spin.x(
                    c) * parity_string_b * spin.x(d)
                spin_hamiltonian -= -0.125 * coef.real * spin.x(
                    a) * parity_string_a * spin.x(b) * spin.y(
                    c) * parity_string_b * spin.y(d)
                spin_hamiltonian += -0.125 * coef.real * spin.x(
                    a) * parity_string_a * spin.y(b) * spin.x(
                    c) * parity_string_b * spin.y(d)
                spin_hamiltonian += -0.125 * coef.real * spin.x(
                    a) * parity_string_a * spin.y(b) * spin.y(
                    c) * parity_string_b * spin.x(d)
                spin_hamiltonian += -0.125 * coef.real * spin.y(
                    a) * parity_string_a * spin.x(b) * spin.x(
                    c) * parity_string_b * spin.y(d)
                spin_hamiltonian += -0.125 * coef.real * spin.y(
                    a) * parity_string_a * spin.x(b) * spin.y(
                    c) * parity_string_b * spin.x(d)
                spin_hamiltonian -= -0.125 * coef.real * spin.y(
                    a) * parity_string_a * spin.y(b) * spin.x(
                    c) * parity_string_b * spin.x(d)
                spin_hamiltonian += -0.125 * coef.real * spin.y(
                    a) * parity_string_a * spin.y(b) * spin.y(
                    c) * parity_string_b * spin.y(d)

                spin_hamiltonian += 0.125 * coef.imag * spin.x(
                    a) * parity_string_a * spin.x(b) * spin.x(
                    c) * parity_string_b * spin.y(d)
                spin_hamiltonian += 0.125 * coef.imag * spin.x(
                    a) * parity_string_a * spin.x(b) * spin.y(
                    c) * parity_string_b * spin.x(d)
                spin_hamiltonian -= 0.125 * coef.imag * spin.x(
                    a) * parity_string_a * spin.y(b) * spin.x(
                    c) * parity_string_b * spin.x(d)
                spin_hamiltonian += 0.125 * coef.imag * spin.x(
                    a) * parity_string_a * spin.y(b) * spin.y(
                    c) * parity_string_b * spin.y(d)
                spin_hamiltonian -= 0.125 * coef.imag * spin.y(
                    a) * parity_string_a * spin.x(b) * spin.x(
                    c) * parity_string_b * spin.x(d)
                spin_hamiltonian += 0.125 * coef.imag * spin.y(
                    a) * parity_string_a * spin.x(b) * spin.y(
                    c) * parity_string_b * spin.y(d)
                spin_hamiltonian -= 0.125 * coef.imag * spin.y(
                    a) * parity_string_a * spin.y(b) * spin.x(
                    c) * parity_string_b * spin.y(d)
                spin_hamiltonian -= 0.125 * coef.imag * spin.y(
                    a) * parity_string_a * spin.y(b) * spin.y(
                    c) * parity_string_b * spin.x(d)

            elif p < r < q < s:
                a, b, c, d = p, r, q, s

                parity_string_a = 1.0
                z_qubit_a = [i for i in range(a + 1, b)]
                for i in z_qubit_a:
                    parity_string_a *= spin.z(i)

                parity_string_b = 1.0
                z_qubit_b = [i for i in range(c + 1, d)]
                for i in z_qubit_b:
                    parity_string_b *= spin.z(i)

                spin_hamiltonian = -0.125 * coef.real * spin.x(
                    a) * parity_string_a * spin.x(b) * spin.x(
                    c) * parity_string_b * spin.x(d)
                spin_hamiltonian += -0.125 * coef.real * spin.x(
                    a) * parity_string_a * spin.x(b) * spin.y(
                    c) * parity_string_b * spin.y(d)
                spin_hamiltonian -= -0.125 * coef.real * spin.x(
                    a) * parity_string_a * spin.y(b) * spin.x(
                    c) * parity_string_b * spin.y(d)
                spin_hamiltonian += -0.125 * coef.real * spin.x(
                    a) * parity_string_a * spin.y(b) * spin.y(
                    c) * parity_string_b * spin.x(d)
                spin_hamiltonian += -0.125 * coef.real * spin.y(
                    a) * parity_string_a * spin.x(b) * spin.x(
                    c) * parity_string_b * spin.y(d)
                spin_hamiltonian -= -0.125 * coef.real * spin.y(
                    a) * parity_string_a * spin.x(b) * spin.y(
                    c) * parity_string_b * spin.x(d)
                spin_hamiltonian += -0.125 * coef.real * spin.y(
                    a) * parity_string_a * spin.y(b) * spin.x(
                    c) * parity_string_b * spin.x(d)
                spin_hamiltonian += -0.125 * coef.real * spin.y(
                    a) * parity_string_a * spin.y(b) * spin.y(
                    c) * parity_string_b * spin.y(d)

                spin_hamiltonian += 0.125 * coef.imag * spin.x(
                    a) * parity_string_a * spin.x(b) * spin.x(
                    c) * parity_string_b * spin.y(d)
                spin_hamiltonian -= 0.125 * coef.imag * spin.x(
                    a) * parity_string_a * spin.x(b) * spin.y(
                    c) * parity_string_b * spin.x(d)
                spin_hamiltonian += 0.125 * coef.imag * spin.x(
                    a) * parity_string_a * spin.y(b) * spin.x(
                    c) * parity_string_b * spin.x(d)
                spin_hamiltonian += 0.125 * coef.imag * spin.x(
                    a) * parity_string_a * spin.y(b) * spin.y(
                    c) * parity_string_b * spin.y(d)
                spin_hamiltonian -= 0.125 * coef.imag * spin.y(
                    a) * parity_string_a * spin.x(b) * spin.x(
                    c) * parity_string_b * spin.x(d)
                spin_hamiltonian -= 0.125 * coef.imag * spin.y(
                    a) * parity_string_a * spin.x(b) * spin.y(
                    c) * parity_string_b * spin.y(d)
                spin_hamiltonian += 0.125 * coef.imag * spin.y(
                    a) * parity_string_a * spin.y(b) * spin.x(
                    c) * parity_string_b * spin.y(d)
                spin_hamiltonian -= 0.125 * coef.imag * spin.y(
                    a) * parity_string_a * spin.y(b) * spin.y(
                    c) * parity_string_b * spin.x(d)

            elif p < r < s < q:
                a, b, c, d = p, r, s, q

                parity_string_a = 1.0
                z_qubit_a = [i for i in range(a + 1, b)]
                for i in z_qubit_a:
                    parity_string_a *= spin.z(i)

                parity_string_b = 1.0
                z_qubit_b = [i for i in range(c + 1, d)]
                for i in z_qubit_b:
                    parity_string_b *= spin.z(i)

                spin_hamiltonian = -0.125 * coef.real * spin.x(
                    a) * parity_string_a * spin.x(b) * spin.x(
                    c) * parity_string_b * spin.x(d)
                spin_hamiltonian += -0.125 * coef.real * spin.x(
                    a) * parity_string_a * spin.x(b) * spin.y(
                    c) * parity_string_b * spin.y(d)
                spin_hamiltonian += -0.125 * coef.real * spin.x(
                    a) * parity_string_a * spin.y(b) * spin.x(
                    c) * parity_string_b * spin.y(d)
                spin_hamiltonian -= -0.125 * coef.real * spin.x(
                    a) * parity_string_a * spin.y(b) * spin.y(
                    c) * parity_string_b * spin.x(d)
                spin_hamiltonian -= -0.125 * coef.real * spin.y(
                    a) * parity_string_a * spin.x(b) * spin.x(
                    c) * parity_string_b * spin.y(d)
                spin_hamiltonian += -0.125 * coef.real * spin.y(
                    a) * parity_string_a * spin.x(b) * spin.y(
                    c) * parity_string_b * spin.x(d)
                spin_hamiltonian += -0.125 * coef.real * spin.y(
                    a) * parity_string_a * spin.y(b) * spin.x(
                    c) * parity_string_b * spin.x(d)
                spin_hamiltonian += -0.125 * coef.real * spin.y(
                    a) * parity_string_a * spin.y(b) * spin.y(
                    c) * parity_string_b * spin.y(d)

                spin_hamiltonian -= 0.125 * coef.imag * spin.x(
                    a) * parity_string_a * spin.x(b) * spin.x(
                    c) * parity_string_b * spin.y(d)
                spin_hamiltonian += 0.125 * coef.imag * spin.x(
                    a) * parity_string_a * spin.x(b) * spin.y(
                    c) * parity_string_b * spin.x(d)
                spin_hamiltonian += 0.125 * coef.imag * spin.x(
                    a) * parity_string_a * spin.y(b) * spin.x(
                    c) * parity_string_b * spin.x(d)
                spin_hamiltonian += 0.125 * coef.imag * spin.x(
                    a) * parity_string_a * spin.y(b) * spin.y(
                    c) * parity_string_b * spin.y(d)
                spin_hamiltonian -= 0.125 * coef.imag * spin.y(
                    a) * parity_string_a * spin.x(b) * spin.x(
                    c) * parity_string_b * spin.x(d)
                spin_hamiltonian -= 0.125 * coef.imag * spin.y(
                    a) * parity_string_a * spin.x(b) * spin.y(
                    c) * parity_string_b * spin.y(d)
                spin_hamiltonian -= 0.125 * coef.imag * spin.y(
                    a) * parity_string_a * spin.y(b) * spin.x(
                    c) * parity_string_b * spin.y(d)
                spin_hamiltonian += 0.125 * coef.imag * spin.y(
                    a) * parity_string_a * spin.y(b) * spin.y(
                    c) * parity_string_b * spin.x(d)

        return spin_hamiltonian