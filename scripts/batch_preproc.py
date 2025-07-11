from quantum_mcp.preprocess import MolecularHamiltonian


def main():
    xyz_fpath = "/home/jowens/projects/quantum_computing/co2.xyz"
    basis = '631g'
    use_natural_orbitals = False

    mh = MolecularHamiltonian(xyz_fpath, basis, use_natural_orbitals=use_natural_orbitals, integrals_natorb=False,
                              nele_cas=4, norb_cas=4)
    mh.run_mp2()
    mh.run_casci()
    mh.run_ccsd()
    mh.run_casscf()
    mh.run_fci_active_space()
    mh.compute_obi_and_tbi()
    mh.write_files()

if __name__ == '__main__':
    main()
