import cudaq
import numpy as np
import time


from quantum_mcp.mappings import JordanWignerFermion
from quantum_mcp.operator_pools import uccsd_pool
from quantum_mcp.preprocess import MolecularHamiltonian

start_time = time.time()

cudaq.set_target("nvidia", option="fp64")

xyz_fpath = '/home/de721625/comp_mat/qc/data/ampd/2x-ampd_1x-co2_wB97X-D.xyz'
output_csv = open('/home/de721625/comp_mat/qc/data/ampd/2x-ampd_1x-co2_wB97X-D_adapt-energes.csv', 'w')
output_csv.write('norb_cas,nelec_cas,e-adapt,n-pools,n-iter\n')
# norb_cas = 8
# nele_cas = 8
max_iter = 1000
for active_space_size in [12]:
    n_orb_cas = active_space_size
    nele_cas = active_space_size
    molecular_data = MolecularHamiltonian(xyz_fpath, '631g', norb_cas=active_space_size, nele_cas=active_space_size)

    molecular_data.run_ccsd()
    molecular_data.compute_obi_and_tbi()
    jwf = JordanWignerFermion(molecular_data)
    spin_ham = jwf.get_spin_hamiltonian()

    n_electrons = molecular_data.nele_cas
    n_qubits = 2 * molecular_data.nele_cas
    molecular_data.nqubits = n_qubits
    pools = uccsd_pool(n_electrons, 2 * molecular_data.nele_cas)
    print('Number of operator pool: ', len(pools))

    sign_pool = []
    mod_pool = []
    for i in range(len(pools)):
        op_i = pools[i]
        temp_op = []
        temp_coef = []

        for term in op_i:
            temp_coef.append(term.evaluate_coefficient())
            temp_op.append(term.get_pauli_word(n_qubits))

        mod_pool.append(temp_op)
        sign_pool.append(temp_coef)
        print(mod_pool)
        print(sign_pool)


    def commutator(pools, ham):
        com_op = []

        for i in range(len(pools)):
            # We add the imaginary number that we excluded when generating the operator pool.
            op = 1j * pools[i]

            com_op.append(ham * op - op * ham)

        return com_op


    grad_op = commutator(pools, spin_ham)
    print('Number of op for gradient: ', len(grad_op))


    @cudaq.kernel
    def initial_state(n_qubits: int, nelectrons: int):
        qubits = cudaq.qvector(n_qubits)

        for i in range(nelectrons):
            x(qubits[i])


    state = cudaq.get_state(initial_state, n_qubits, n_electrons)
    print(state)


    ###################################
    # Quantum kernels

    @cudaq.kernel
    def gradient(state: cudaq.State):
        q = cudaq.qvector(state)


    @cudaq.kernel
    def kernel(theta: list[float], qubits_num: int, nelectrons: int, pool_single: list[cudaq.pauli_word],
           coef_single: list[float], pool_double: list[cudaq.pauli_word], coef_double: list[float]):
        q = cudaq.qvector(qubits_num)

        for i in range(nelectrons):
            x(q[i])

        count = 0
        for i in range(0, len(coef_single), 2):
            exp_pauli(coef_single[i] * theta[count], q, pool_single[i])
            exp_pauli(coef_single[i + 1] * theta[count], q, pool_single[i + 1])
            count += 1

        for i in range(0, len(coef_double), 8):
            exp_pauli(coef_double[i] * theta[count], q, pool_double[i])
            exp_pauli(coef_double[i + 1] * theta[count], q, pool_double[i + 1])
            exp_pauli(coef_double[i + 2] * theta[count], q, pool_double[i + 2])
            exp_pauli(coef_double[i + 3] * theta[count], q, pool_double[i + 3])
            exp_pauli(coef_double[i + 4] * theta[count], q, pool_double[i + 4])
            exp_pauli(coef_double[i + 5] * theta[count], q, pool_double[i + 5])
            exp_pauli(coef_double[i + 6] * theta[count], q, pool_double[i + 6])
            exp_pauli(coef_double[i + 7] * theta[count], q, pool_double[i + 7])
            count += 1


    from scipy.optimize import minimize

    print('Beginning of ADAPT-VQE')

    threshold = 1e-3
    E_prev = 0.0
    e_stop = 1e-5
    init_theta = 0.0

    theta_single = []
    theta_double = []

    pool_single = []
    pool_double = []

    coef_single = []
    coef_double = []

    selected_pool = []

    for i in range(max_iter):

        print('Step: ', i)

        gradient_vec = []

        for op in grad_op:
            grad = cudaq.observe(gradient, op, state).expectation()
            gradient_vec.append(grad)

        norm = np.linalg.norm(np.array(gradient_vec))
        print('Norm of the gradient: ', norm)

        # When using mpi to parallelize gradient calculation: uncomment the following lines

        # chunks=np.array_split(np.array(grad_op), cudaq.mpi.num_ranks())
        # my_rank_op=chunks[cudaq.mpi.rank()]

        # print('We have', len(grad_op), 'pool operators which we would like to split', flush=True)
        # print('We have', len(my_rank_op), 'pool operators on this rank', cudaq.mpi.rank(), flush=True)

        # gradient_vec_async=[]

        # for op in my_rank_op:
        # gradient_vec_async.append(cudaq.observe_async(gradient, op, state))

        # gradient_vec_rank=[]
        # for i in range(len(gradient_vec_async)):
        #    get_result=gradient_vec_async[i].get()
        #    get_expectation=get_result.expectation()
        #    gradient_vec_rank.append(get_expectation)

        # print('My rank has', len(gradient_vec_rank), 'gradients', flush=True)

        # gradient_vec=cudaq.mpi.all_gather(len(gradient_vec_rank)*cudaq.mpi.num_ranks(), gradient_vec_rank)

        if norm <= threshold:
            print('\n', 'Final Result: ', '\n')
            print('Final parameters: ', theta)
            print('Selected pools: ', selected_pool)
            print('Number of pools: ', len(selected_pool))
            print('Final energy: ', result_vqe.fun)
            output_csv.write(f'{n_orb_cas},{nele_cas},{result_vqe.fun},{len(selected_pool)},{i}\n')
            output_csv.flush()

            break

        else:

            max_grad = np.max(np.abs(gradient_vec))
            print('max_grad: ', max_grad)

            temp_pool = []
            temp_sign = []
            for i in range(len(mod_pool)):
                if np.abs(gradient_vec[i]) == max_grad:
                    temp_pool.append(mod_pool[i])
                    temp_sign.append(sign_pool[i])

            print('Selected pool at current step: ', temp_pool)

            selected_pool = selected_pool + temp_pool

            tot_single = 0
            tot_double = 0
            for p in temp_pool:
                if len(p) == 2:
                    tot_single += 1
                    for word in p:
                        pool_single.append(word)
                else:
                    tot_double += 1
                    for word in p:
                        pool_double.append(word)

            for coef in temp_sign:
                if len(coef) == 2:
                    for value in coef:
                        coef_single.append(value.real)
                else:
                    for value in coef:
                        coef_double.append(value.real)

            print('pool single: ', pool_single)
            print('coef_single: ', coef_single)
            print('pool_double: ', pool_double)
            print('coef_double: ', coef_double)
            print('tot_single: ', tot_single)
            print('tot_double: ', tot_double)

            init_theta_single = [init_theta] * tot_single
            init_theta_double = [init_theta] * tot_double

            theta_single = theta_single + init_theta_single
            theta_double = theta_double + init_theta_double
            print('theta_single', theta_single)
            print('theta_double: ', theta_double)

            theta = theta_single + theta_double
            print('theta', theta)


            def cost(theta):

                theta = theta.tolist()

                energy = cudaq.observe(kernel, spin_ham, theta, n_qubits, n_electrons, pool_single,
                                       coef_single, pool_double, coef_double).expectation()

                return energy


            def parameter_shift(theta):
                parameter_count = len(theta)
                grad = np.zeros(parameter_count)
                theta2 = theta.copy()
                for i in range(parameter_count):
                    theta2[i] = theta[i] + np.pi / 4
                    exp_val_plus = cost(theta2)
                    theta2[i] = theta[i] - np.pi / 4
                    exp_val_minus = cost(theta2)
                    grad[i] = (exp_val_plus - exp_val_minus)
                    theta2[i] = theta[i]
                return grad


            result_vqe = minimize(cost, theta, method='L-BFGS-B', jac='3-point', tol=1e-7)
            # If want to use parameter shift to compute gradient, please uncomment the following line.
            # result_vqe=minimize(cost, theta, method='L-BFGS-B', jac=parameter_shift, tol=1e-7)

            theta = result_vqe.x.tolist()
            theta_single = theta[:tot_single]
            theta_double = theta[tot_single:]

            print('Optmized Energy: ', result_vqe.fun)
            print('Optimizer exited successfully: ', result_vqe.success, flush=True)
            print(result_vqe.message, flush=True)

            dE = result_vqe.fun - E_prev
            print('dE: ', dE)
            print('\n')

            if np.abs(dE) <= e_stop:
                print('\n', 'Final Result: ', '\n')
                print('Final parameters: ', theta)
                print('Selected pools: ', selected_pool)
                print('Number of pools: ', len(selected_pool))
                print('Final energy: ', result_vqe.fun)
                output_csv.write(f'{n_orb_cas},{nele_cas},{result_vqe.fun},{len(selected_pool)},{i}\n')
                output_csv.flush()
                break

            else:
                E_prev = result_vqe.fun

                # Prepare a trial state with the current ansatz.
                state = cudaq.get_state(kernel, theta, n_qubits, n_electrons, pool_single,
                                        coef_single, pool_double, coef_double)
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time: ")
# When using mpi
# cudaq.mpi.finalize()
