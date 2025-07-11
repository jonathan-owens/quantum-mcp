# Heavily adapted from Marwa/NVIDIA's code: link
import json
from functools import reduce
from pathlib import Path
import sys

import numpy as np
from pyscf import ao2mo, cc, fci, gto, lib, mcscf, mp, scf
from pyscf.tools import molden


class MolecularHamiltonian:
    def __init__(self, xyz_fpath:str|Path, basis:str, symmetry:bool=False, memory:float=4000,
                 cycles:int=100, initguess:str='minao', nele_cas:int=None, norb_cas:int=None,
                 use_natural_orbitals:bool=False, integrals_natorb:bool=False, integrals_casscf:bool=False,
                 viz_orb:bool=False, log_level:int=5, logfile: str | Path=None):
        self.xyz_fpath = Path(xyz_fpath)
        self.basis = basis
        self.symmetry = symmetry
        self.memory = memory
        self.cycles = cycles
        self.initguess = initguess
        self.nele_cas = nele_cas
        self.norb_cas = norb_cas
        self.frozen = None
        self.use_natural_orbitals = use_natural_orbitals
        self.integrals_natorb = integrals_natorb
        self.integrals_casscf = integrals_casscf
        self.viz_orb = viz_orb
        self.hf_molden_file = None
        self.mp2_molden_file = None
        self.h1e = None
        self.h2e = None
        self.core_energy = None
        self.obi_coeff = None
        self.tbi_coeff = None
        self.full_hamiltonian = None
        self.init_params = None

        # We have two logging locations: our custom messages go to sys.stdout. The verbose info from PySCF goes to
        # self.logfile.
        if logfile is not None:
            self.logfile = Path(logfile)
        else:
            self.logfile = self.xyz_fpath.with_name(f'{self.xyz_fpath.stem}-pyscf.log')
        self.log = lib.logger.Logger(sys.stdout, log_level)

        # We keep spin and charge 0 for this initial implementation
        self.mol = gto.M(atom=str(self.xyz_fpath), spin=0, charge=0, basis=self.basis,
                         max_memory=self.memory, symmetry=self.symmetry, output=self.logfile, verbose=log_level)
        self.nelec = self.mol.nelec

        # Class members related to Hartree Fock
        self.hf = None
        self.norb = None
        self.run_hf()  # We run HF automatically because it is needed for all post-HF methods that follow

        # Class members related to MP2
        self.mp2 = None
        self.mp_ecorr = None
        self.mp_t2 = None
        self.natorbs = None  # These are generated with MP2

        # Class members related to CI
        self.fci = None
        self.casci = None

        # Class members related to CCSD
        self.cc = None

        # Class members related to CASSCF
        self.casscf = None

        # Class members related to UCCSD initial params (for later VQE)
        self.n_occ = None
        self.n_virt = None
        self.n_qubits = None
        self.theta_1 = None
        self.theta_2 = None
        self.amplitudes = None

    def run_hf(self):
        """
        Run the Hartree-Fock calculation using PySCF. If `self.viz_orb == True`, the HF orbitals are written out to the
        directory in which the logfile resides.

        :return: HF energy
        """

        self.log.info('[QC_MCP] == Running HF ==')
        self.hf = scf.RHF(self.mol)

        self.hf.max_cycle = self.cycles
        self.hf.chkfile = self.logfile.with_name(f'{self.xyz_fpath.stem}-pyscf.chk')
        self.hf.init_guess = self.initguess
        self.hf.kernel()

        self.norb = self.hf.mo_coeff.shape[1]

        self.log.info(f'[QC_MCP] Total number of orbitals = {self.norb}')
        self.log.info(f'[QC_MCP] Total number of electrons = {self.nelec}')
        self.log.info(f'[QC_MCP] E[HF] = {self.hf.e_tot}')

        if self.viz_orb:
            self.hf_molden_file = self.logfile.with_name(f'{self.xyz_fpath.stem}_HF_molorb.molden')
            molden.from_mo(self.mol, self.hf_molden_file, self.hf.mo_coeff)
            self.log.info(f'Saved HF molecular orbitals to {self.hf_molden_file}')

        return self.hf.e_tot

    def run_mp2(self):
        """
        Run MP2. If `self.viz_orb == True`, the MP2 orbitals are written out to the
        directory in which the logfile resides.

        :return: MP2 energy
        """
        self.log.info(f'[QC_MCP] == Running MP2 ==')
        self.mp2 = mp.MP2(self.hf)
        self.mp_ecorr, self.mp_t2 = self.mp2.kernel()

        self.log.info(f'[QC_MCP] E[MP2] = {self.mp2.e_tot}')
        self.log.info(f'[QC_MCP] E_corr[MP2] = {self.mp_ecorr}')

        if self.integrals_natorb or self.use_natural_orbitals:
            noons, natorbs = mcscf.addons.make_natural_orbitals(self.mp2)
            self.natorbs = natorbs
            self.log.info(f'[QC_MCP] Natural orbital occupation number from R-MP2] = {noons}')

            if self.viz_orb:
                self.mp2_molden_file = self.logfile.with_name(f'{self.xyz_fpath.stem}_MP2_natorb.molden')
                molden.from_mo(self.mol, self.mp2_molden_file, self.natorbs)
                self.log.info(f'Saved MP2 natural orbitals to {self.mp2_molden_file}')

        return self.mp2.e_tot

    def run_casci(self):
        self.log.info(f'[QC_MCP] == Running CASCI ==')
        if self.nele_cas is None:
            self.log.info(f'[QC_MCP] `self.nele_cas` was not passed, running FCI!')
            self.fci = fci.FCI(self.hf)
            self.fci.kernel()
            self.log.info(f'[QC_MCP] E[FCI] = {self.fci.e_tot}')
        else:
            self.casci = mcscf.CASCI(self.hf, self.norb_cas, self.nele_cas)
            if self.use_natural_orbitals:
                if self.mp2 is None:
                    self.log.info(f'[QC_MCP] Running MP2 to get natural orbitals before running CASCI!')
                    self.run_mp2()

                self.casci.kernel(self.natorbs)
            else:
                self.casci.kernel()

            self.log.info(f'[QC_MCP] E[CASCI] = {self.casci.e_tot}')

        return self.casci.e_tot

    def run_ccsd(self):
        self.log.info(f'[QC_MCP] == Running R-CCSD ==')
        if self.nele_cas is None:
            self.cc = cc.CCSD(self.hf)
        else:
            # We run CASCI before CCSD to get the frozen orbitals
            if self.casci is None:
                self.run_casci()
            if self.frozen is None:
                # We freeze the orbitals outside a range spanned by `self.norb_cas` around the Fermi level
                self.frozen = []
                self.frozen += [y for y in range(0, self.casci.ncore)]
                self.frozen += [
                    y for y in range(self.casci.ncore + self.norb_cas, len(self.casci.mo_coeff))
                ]
            # If a list of frozen orbitals was passed, then we use that instead

            if self.use_natural_orbitals:
                self.cc = cc.CCSD(self.hf, frozen=self.frozen, mo_coeff=self.natorbs)
            else:
                self.cc = cc.CCSD(self.hf, frozen=self.frozen)
        self.cc.max_cycle = self.cycles
        self.cc.kernel()

        self.log.info(f'[QC_MCP] E[R-CCSD] = {self.cc.e_tot}')

        return self.cc.e_tot

    def run_casscf(self):
        self.log.info(f'[QC_MCP] == Running CASSCF ==')
        if self.nele_cas is None:
            raise ValueError("You have to define an active space to run CASSCF. Set `nelec_cas`.")

        self.casscf = mcscf.CASSCF(self.hf, self.norb_cas, self.nele_cas)
        self.casscf.max_cycle_macro = self.cycles
        if self.use_natural_orbitals:
            if not self.mp2:
                self.log.info(f'[QC_MCP] Running MP2 to get natural orbitals before running CASSCF!')
                self.run_mp2()
            self.casscf.kernel(self.natorbs)
        else:
            self.casscf.kernel()

        self.log.info(f'[QC_MCP] E[CASSCF] = {self.casscf.e_tot}')

        return self.casscf.e_tot

    def run_fci_active_space(self):
        self.log.info(f'[QC_MCP] == Running FCI of the active space ==')
        h1e_cas, ecore = self.casscf.get_h1eff()
        h2e_cas = self.casscf.get_h2eff()

        e_fci, _ = fci.direct_spin1.kernel(h1e_cas, h2e_cas, self.norb_cas, self.nele_cas, ecore=ecore)

        self.log.info(f'[QC_MCP] Active Space E[FCI] = {e_fci}')
        return e_fci

    def compute_obi_and_tbi(self):
        """
        Which terms are used to calculate the one- and two-body integrals depends on the calculation parameters

        Scenario 1: FCI calculation. We get the atomic orbitals then convert to HF orbitals
        Scenario 2: Active space defined, natural orbitals: CASCI object gets H1/H2 and is used
        Scenario 3: Active space defined, CASSCF integrals (integrals_casscf=True): CASSCF object gets H1/H2 and is used
        Scenario 4: Active space defined, CASCI integrals, MO: CASCI object gets H1/H2 and is used

        :return:
        """
        self.log.info(f'[QC_MCP] Computing the one- and two-electron integrals')
        if self.nele_cas is None:
            self.log.info(f'[QC_MCP] Using one- and two-body integrals from HF in the MO basis')
            # Compute the 1 electron integral in the atomic orbital basis then convert to HF molecular orbitals
            h1e_ao = self.mol.intor('int1e_kin') + self.mol.intor('int1e_nuc')
            self.h1e = reduce(np.dot,(self.hf.mo_coeff.T, h1e_ao, self.hf.mo_coeff))

            # Do the same for the two electron integrals
            h2e_ao = self.mol.intor('int2e_sph', aosym='1')
            self.h2e = ao2mo.incore.full(h2e_ao, self.hf.mo_coeff)

            # Reorder from chemist's notation
            self.h2e = self.h2e.transpose(0, 2, 3, 1)

            self.core_energy = self.hf.energy_nuc()
        else:
            if self.integrals_natorb:
                if not self.casci:
                    self.run_casci()
                self.log.info(f'[QC_MCP] Using one- and two-body integrals from CASCI in the natural basis')
                self.h1e, self.core_energy = self.casci.get_h1eff(self.natorbs)
                self.h2e = self.casci.get_h2eff(self.natorbs)
                # self.h2e = ao2mo.restore('1', self.h2e, self.norb_cas)
                # self.h2e = np.asarray(self.h2e.transpose(0, 2, 3, 1))
            elif self.integrals_casscf:
                if not self.casscf:
                    self.run_casscf()
                self.log.info(f'[QC_MCP] Using one- and two-body integrals from CASSCF in the natural basis')
                self.h1e, self.core_energy = self.casscf.get_h1eff(self.natorbs)
                self.h2e = self.casscf.get_h2eff(self.natorbs)
                # self.h2e = ao2mo.restore('1', self.h2e, self.norb_cas)
                # self.h2e = np.asarray(self.h2e.transpose(0, 2, 3, 1))
            else:
                self.log.info(f'[QC_MCP] Using one- and two-body integrals from CASCI in the HF MO basis')
                self.h1e, self.core_energy = self.casci.get_h1eff(self.hf.mo_coeff)
                self.h2e = self.casci.get_h2eff(self.hf.mo_coeff)
            self.h2e = ao2mo.restore('1', self.h2e, self.norb_cas)
            self.h2e = np.asarray(self.h2e.transpose(0, 2, 3, 1))  # Do I need to do C order? order='C'
        self.log.info(f'[QC_MCP] Generated the restricted molecular spin Hamiltonian')
        return self.generate_mol_spin_ham_restricted()

    def generate_mol_spin_ham_restricted(self):
        # This function generates the molecular spin Hamiltonian
        # H = E_core+sum_{`pq`}  h_{`pq`} a_p^dagger a_q +
        #                          0.5 * h_{`pqrs`} a_p^dagger a_q^dagger a_r a_s
        # h1e: one body integrals h_{`pq`}
        # h2e: two body integrals h_{`pqrs`}
        # `ecore`: constant (nuclear repulsion or core energy in the active space Hamiltonian)

        # Total number of qubits equals the number of spin molecular orbitals
        nqubits = 2 * self.h1e.shape[0]

        # Initialization
        one_body_coeff = np.zeros((nqubits, nqubits))
        two_body_coeff = np.zeros((nqubits, nqubits, nqubits, nqubits))

        ferm_ham = []

        for p in range(nqubits // 2):
            for q in range(nqubits // 2):

                # p & q have the same spin <a|a>= <b|b>=1
                # <a|b>=<b|a>=0 (orthogonal)
                one_body_coeff[2 * p, 2 * q] = self.h1e[p, q]
                temp = str(self.h1e[p, q]) + ' a_' + str(p) + '^dagger ' + 'a_' + str(q)
                ferm_ham.append(temp)
                one_body_coeff[2 * p + 1, 2 * q + 1] = self.h1e[p, q]
                temp = str(self.h1e[p, q]) + ' b_' + str(p) + '^dagger ' + 'b_' + str(q)
                ferm_ham.append(temp)

                for r in range(nqubits // 2):
                    for s in range(nqubits // 2):
                        # Same spin (`aaaa`, `bbbbb`) <a|a><a|a>, <b|b><b|b>
                        two_body_coeff[2 * p, 2 * q, 2 * r,
                                       2 * s] = 0.5 * self.h2e[p, q, r, s]
                        temp = str(0.5 * self.h2e[p, q, r, s]) + ' a_' + str(
                            p) + '^dagger ' + 'a_' + str(
                            q) + '^dagger ' + 'a_' + str(r) + ' a_' + str(s)
                        ferm_ham.append(temp)
                        two_body_coeff[2 * p + 1, 2 * q + 1, 2 * r + 1,
                                       2 * s + 1] = 0.5 * self.h2e[p, q, r, s]
                        temp = str(0.5 * self.h2e[p, q, r, s]) + ' b_' + str(
                            p) + '^dagger ' + 'b_' + str(
                            q) + '^dagger ' + 'b_' + str(r) + ' b_' + str(s)
                        ferm_ham.append(temp)

                        # Mixed spin(`abab`, `baba`) <a|a><b|b>, <b|b><a|a>
                        # <a|b>= 0 (orthogonal)
                        two_body_coeff[2 * p, 2 * q + 1, 2 * r + 1,
                                       2 * s] = 0.5 * self.h2e[p, q, r, s]
                        temp = str(0.5 * self.h2e[p, q, r, s]) + ' a_' + str(
                            p) + '^dagger ' + 'a_' + str(
                            q) + '^dagger ' + 'b_' + str(r) + ' b_' + str(s)
                        ferm_ham.append(temp)
                        two_body_coeff[2 * p + 1, 2 * q, 2 * r,
                                       2 * s + 1] = 0.5 * self.h2e[p, q, r, s]
                        temp = str(0.5 * self.h2e[p, q, r, s]) + ' b_' + str(
                            p) + '^dagger ' + 'b_' + str(
                            q) + '^dagger ' + 'a_' + str(r) + ' a_' + str(s)
                        ferm_ham.append(temp)

        full_hamiltonian = " + ".join(ferm_ham)
        self.obi_coeff = one_body_coeff
        self.tbi_coeff = two_body_coeff
        self.full_hamiltonian = full_hamiltonian
        return one_body_coeff, two_body_coeff, self.core_energy, full_hamiltonian

    def get_thetas_unpack_restricted(self):
        self.theta_1 = np.zeros((2 * self.n_occ, 2 * self.n_virt))
        self.theta_2 = np.zeros((2 * self.n_occ, 2 * self.n_occ, 2 * self.n_virt, 2 * self.n_virt))
        for p in range(self.n_occ):
            for q in range(self.n_virt):
                self.theta_1[2 * p, 2 * q] = self.cc.t1[p, q]
                self.theta_1[2 * p + 1, 2 * q + 1] = self.cc.t1[p, q]

        for p in range(self.n_occ):
            for q in range(self.n_occ):
                for r in range(self.n_virt):
                    for s in range(self.n_virt):
                        self.theta_2[2 * p, 2 * q, 2 * s, 2 * r] = self.cc.t2[p, q, r, s]
                        self.theta_2[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1] = self.cc.t2[p, q, r, s]
                        self.theta_2[2 * p, 2 * q + 1, 2 * r + 1, 2 * s] = self.cc.t2[p, q, r, s]
                        self.theta_2[2 * p + 1, 2 * q, 2 * r, 2 * s + 1] = self.cc.t2[p, q, r, s]


    def get_uccsd_amplitudes(self):
        self.log.info('[QC_MCP] Generating initial amplitudes for UCCSD')
        if not self.cc:
            self.log.info('[QC_MCP] You forgot to run CCSD before calling get_uccsd_amplitudes! Running for you.')
            self.run_ccsd()
        self.n_qubits = 2 * self.nele_cas
        self.n_occ = self.nele_cas // 2
        self.n_virt = self.n_qubits // 2 - self.n_occ

        # Populates the thetas in the restricted problem
        self.get_thetas_unpack_restricted()

        singles_alpha = []
        singles_beta = []
        doubles_mixed = []
        doubles_alpha = []
        doubles_beta = []

        occupied_alpha_indices = [i * 2 for i in range(self.n_occ)]
        virtual_alpha_indices = [i * 2 for i in range(self.n_virt)]

        occupied_beta_indices = [i * 2 + 1 for i in range(self.n_occ)]
        virtual_beta_indices = [i * 2 + 1 for i in range(self.n_virt)]

        # Same spin single excitation
        for p in occupied_alpha_indices:
            for q in virtual_alpha_indices:
                singles_alpha.append(self.theta_1[p, q])

        for p in occupied_beta_indices:
            for q in virtual_beta_indices:
                singles_beta.append(self.theta_1[p, q])

        # Mixed spin double excitation
        for p in occupied_alpha_indices:
            for q in occupied_beta_indices:
                for r in virtual_beta_indices:
                    for s in virtual_alpha_indices:
                        doubles_mixed.append(self.theta_2[p, q, r, s])

        # Same spin double excitation
        n_occ_alpha = len(occupied_alpha_indices)
        n_occ_beta = len(occupied_beta_indices)
        n_virt_alpha = len(virtual_alpha_indices)
        n_virt_beta = len(virtual_beta_indices)

        for p in range(n_occ_alpha - 1):
            for q in range(p + 1, n_occ_alpha):
                for r in range(n_virt_alpha - 1):
                    for s in range(r + 1, n_virt_alpha):
                        # Same spin: all alpha
                        doubles_alpha.append(self.theta_2[occupied_alpha_indices[p], occupied_alpha_indices[q], \
                            virtual_alpha_indices[r], virtual_alpha_indices[s]])

        for p in range(n_occ_beta - 1):
            for q in range(p + 1, n_occ_beta):
                for r in range(n_virt_beta - 1):
                    for s in range(r + 1, n_virt_beta):
                        # Same spin: all beta
                        doubles_beta.append(self.theta_2[occupied_beta_indices[p], occupied_beta_indices[q], \
                            virtual_beta_indices[r], virtual_beta_indices[s]])

        self.amplitudes = singles_alpha + singles_beta + doubles_mixed + doubles_alpha + doubles_beta

    def write_files(self, obi_fname=None, tbi_fname=None, metadata_fname=None, params_fname=None):
        if obi_fname is None:
            obi_fname = self.logfile.with_name(f'{self.xyz_fpath.stem}_obi.dat')
        if tbi_fname is None:
            tbi_fname = self.logfile.with_name(f'{self.xyz_fpath.stem}_tbi.dat')
        if metadata_fname is None:
            metadata_fname = self.logfile.with_name(f'{self.xyz_fpath.stem}_metadata.json')
        if params_fname is None:
            params_fname = self.logfile.with_name(f'{self.xyz_fpath.stem}_params.dat')

        self.obi_coeff.astype(complex).tofile(obi_fname)
        self.tbi_coeff.astype(complex).tofile(tbi_fname)
        metadata = {'num_electrons': self.nelec, 'num_orbitals': self.norb, 'core_energy': self.core_energy,
                    'hf_energy': self.hf.e_tot}
        with open(metadata_fname, 'w') as f:
            json.dump(metadata, f)
        if not self.amplitudes:
            self.log.info('[QC_MCP] You are trying to write the CC amplitudes to file before computing them! Computing '
                          'them for you.')
            self.get_uccsd_amplitudes()
        np.array(self.amplitudes).tofile(params_fname)