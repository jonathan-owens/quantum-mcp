from pathlib import Path

from quantum_mcp.preprocess import MolecularHamiltonian


def main():
    active_space_sizes = [4, 6, 8]  #, 10, 12, 14]
    basis = '631g'
    # basis = 'sto-3g'
    use_natural_orbitals = True
    if use_natural_orbitals:
        orbs_str = 'nat'
    else:
        orbs_str = 'mol'
    save_orbitals = True
    overwrite = False
    base_dir = Path("/home/jowens/projects/quantum_computing/co2-ads-vqe/ed")
    out_csv_fpath = Path("/home/jowens/projects/quantum_computing/co2-ads-vqe/ed/energies_a.csv")

    if not overwrite and out_csv_fpath.exists():
        out_csv = open(out_csv_fpath, 'a')
    else:
        out_csv = open(out_csv_fpath, 'w')
        out_csv.write(f"system,n_e,n_o,basis,nat_orbs,e_hf,e_mp2,e_casci,e_casscf,e_ccsd\n")

    for active_space_size in active_space_sizes:
        run_dir = base_dir / f"{active_space_size}e_{active_space_size}o_{orbs_str}"
        run_dir.mkdir(exist_ok=True)
        for xyz_fpath in base_dir.glob('*.xyz'):
            logfile = run_dir / f"{xyz_fpath.stem}.log"
            mh = MolecularHamiltonian(xyz_fpath, basis, use_natural_orbitals=use_natural_orbitals,
                                      integrals_natorb=use_natural_orbitals,
                                      nele_cas=active_space_size, norb_cas=active_space_size,
                                      logfile=logfile, viz_orb=save_orbitals)
            e_mp2 = mh.run_mp2()
            e_casci = mh.run_casci()
            e_ccsd = mh.run_ccsd()
            e_casscf = mh.run_casscf()
            # mh.run_fci_active_space()
            mh.compute_obi_and_tbi()
            mh.write_files()

            out_csv.write(f"{xyz_fpath.stem},{active_space_size},{active_space_size},{basis},{use_natural_orbitals},{mh.hf.e_tot},{e_mp2},{e_casci},{e_casscf},{e_ccsd}\n")
            out_csv.flush()

if __name__ == '__main__':
    main()
