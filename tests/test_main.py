"""Tests for VariationalQuantum-Lab: simulator, exact solver, VQE race,
QAOA, measured plateaus, noise floors, classifier, gates."""
import math

import pytest

import main as m


# ---- statevector simulator
def test_zero_state():
    s = m.zero_state(3)
    assert s[0] == 1.0 and abs(m.state_norm(s) - 1.0) < 1e-12


def test_ry_pi_flips():
    out = m.ry(m.zero_state(1), 1, 0, math.pi)
    assert abs(abs(out[1]) - 1.0) < 1e-12


def test_hadamard_superposition():
    out = m.hadamard(m.zero_state(1), 1, 0)
    assert abs(abs(out[0]) ** 2 - 0.5) < 1e-12
    assert abs(abs(out[1]) ** 2 - 0.5) < 1e-12


def test_cnot_bell_state():
    bell = m.cnot(m.hadamard(m.zero_state(2), 2, 0), 2, 0, 1)
    assert abs(abs(bell[0]) ** 2 - 0.5) < 1e-12
    assert abs(bell[1]) < 1e-12 and abs(bell[2]) < 1e-12
    assert abs(abs(bell[3]) ** 2 - 0.5) < 1e-12


def test_pauli_expectations():
    assert abs(m.expectation(m.zero_state(1), 1, [(1.0, ("Z",))]) - 1.0) < 1e-12
    plus = m.hadamard(m.zero_state(1), 1, 0)
    assert abs(m.expectation(plus, 1, [(1.0, ("X",))]) - 1.0) < 1e-12


def test_ansatz_normalized_and_count():
    state = m.ansatz_state([0.3] * 12, 4, 2)
    assert abs(m.state_norm(state) - 1.0) < 1e-9
    assert m.ansatz_param_count(4, 2) == 12
    assert m.ansatz_param_count(8, 8) == 72


# ---- exact solver
def test_jacobi_z():
    assert m.jacobi_eigenvalues(
        m.hamiltonian_matrix(1, [(1.0, ("Z",))])) == [-1.0, 1.0]


def test_jacobi_trace_preserved():
    H = m.hamiltonian_matrix(4, m.tfim_terms(4))
    evals = m.jacobi_eigenvalues(H)
    assert abs(sum(evals) - sum(H[i][i].real for i in range(16))) < 1e-8


def test_tfim2_exact_is_minus_sqrt5():
    assert abs(m.exact_ground_energy(2, m.tfim_terms(2))
               + math.sqrt(5)) < 1e-9


def test_tfim4_exact():
    assert abs(m.exact_ground_energy(4, m.tfim_terms(4))
               - (-4.7587704831)) < 1e-9


def test_h2_exact():
    assert abs(m.exact_ground_energy(2, m.H2_TERMS)
               - (-1.8512964969)) < 1e-9


# ---- gradients
def test_param_shift_matches_finite_difference():
    probe = [0.31, -0.72, 1.05, 0.44, -0.18, 0.91]
    for i in range(len(probe)):
        ps = m.param_shift_gradient(probe, 2, 2, m.H2_TERMS, i)
        fd = m.finite_difference_gradient(probe, 2, 2, m.H2_TERMS, i)
        assert abs(ps - fd) < 1e-6


def test_gradient_of_constant_is_zero():
    probe = [0.31, -0.72, 1.05, 0.44, -0.18, 0.91]
    assert abs(m.param_shift_gradient(
        probe, 2, 2, [(2.5, ("I", "I"))], 0)) < 1e-12


# ---- VQE + race
@pytest.fixture(scope="module")
def vqe():
    return m.vqe_report()


@pytest.fixture(scope="module")
def race():
    return m.race_report()


def test_vqe_h2_variational_principle(vqe):
    row = vqe["h2_toy"]
    assert row["vqe_energy"] >= row["exact_ground_energy"] - 1e-9
    assert row["gap_to_exact"] < 1e-6


def test_vqe_tfim_variational_principle(vqe):
    row = vqe["tfim4"]
    assert row["vqe_energy"] >= row["exact_ground_energy"] - 1e-9
    assert row["gap_to_exact"] < 0.01


def test_vqe_zero_params_energies():
    assert abs(m.ansatz_energy([0.0] * 12, 4, 2, m.tfim_terms(4))
               - (-3.0)) < 1e-12
    assert abs(m.ansatz_energy([0.0] * 6, 2, 2, m.H2_TERMS)
               - 0.0001) < 1e-9


def test_race_all_above_exact(race):
    for row in race["results"]:
        assert row["final_energy"] >= race["exact_ground_energy"] - 1e-9


def test_race_gaps(race):
    by = {r["optimizer"]: r for r in race["results"]}
    assert by["gradient_descent"]["gap_to_exact"] < 0.01
    assert by["random_restart"]["gap_to_exact"] < 0.01
    assert by["nelder_mead"]["gap_to_exact"] < 0.25


def test_race_reports_evaluations(race):
    for row in race["results"]:
        assert row["evaluations"] > 0


# ---- QAOA
@pytest.fixture(scope="module")
def qaoa():
    return m.qaoa_report()


def test_maxcut_brute_force():
    assert m.maxcut_brute_force() == 4


def test_qaoa_zero_params_mean_cut():
    assert abs(m.qaoa_expectation([0.0], [0.0]) - 2.5) < 1e-9


def test_qaoa_p1_ratio(qaoa):
    assert 0.75 < qaoa["p1"]["approximation_ratio"] < 0.90


def test_qaoa_p2_improves_or_holds(qaoa):
    assert (qaoa["p2"]["approximation_ratio"]
            >= qaoa["p1"]["approximation_ratio"] - 1e-9)


def test_qaoa_p2_ratio_high(qaoa):
    assert qaoa["p2"]["approximation_ratio"] > 0.90


# ---- barren plateaus (measured)
@pytest.fixture(scope="module")
def plateau():
    return m.plateau_report()


def test_plateau_variance_endpoints(plateau):
    rows = {r["qubits"]: r
            for r in plateau["qubit_scan_depth_equals_qubits"]}
    assert 0.10 < rows[2]["global_variance"] < 0.50
    assert 0.001 < rows[8]["global_variance"] < 0.010


def test_plateau_decay_and_fit(plateau):
    assert plateau["variance_ratio_2q_to_8q_global"] > 50
    assert plateau["global_fit"]["slope_log10_per_qubit"] < -0.20


def test_plateau_local_decays_slower(plateau):
    assert (plateau["local_fit"]["slope_log10_per_qubit"]
            > plateau["global_fit"]["slope_log10_per_qubit"])
    rows = {r["qubits"]: r
            for r in plateau["qubit_scan_depth_equals_qubits"]}
    assert rows[8]["local_variance"] > rows[8]["global_variance"]


def test_plateau_depth_scan(plateau):
    rows = {r["depth"]: r for r in plateau["depth_scan_at_8_qubits"]}
    assert rows[8]["global_variance"] < rows[1]["global_variance"]


# ---- noise honesty
@pytest.fixture(scope="module")
def noise():
    return m.noise_report()


def test_noise_zero_p_recovers_clean(noise):
    row0 = noise["rows"][0]
    assert row0["p_per_layer"] == 0.0
    assert abs(row0["energy_at_clean_optimum"]
               - noise["clean_vqe_energy"]) < 1e-9


def test_noise_floor_rises_with_p(noise):
    rows = {r["p_per_layer"]: r for r in noise["rows"][:4]}
    assert (rows[0.01]["noise_floor"] < rows[0.02]["noise_floor"]
            < rows[0.05]["noise_floor"])


def test_noise_floor_gap_at_2pct(noise):
    rows = {r["p_per_layer"]: r for r in noise["rows"][:4]}
    assert 0.04 < rows[0.02]["floor_gap_to_exact"] < 0.07


def test_noise_model_is_labelled(noise):
    assert noise["model"].startswith("LABELLED toy")


# ---- classifier
def test_classifier_scenarios():
    rep = m.classifier_report()
    assert rep["2q_depth2_global_cost"] == "trainable"
    assert rep["5q_depth5_global_cost"] == "hard"
    assert rep["8q_depth8_global_cost"] == "barren"


def test_classifier_matches_model(plateau):
    for row in plateau["qubit_scan_depth_equals_qubits"]:
        knn = m.classify_trainability(
            row["qubits"], row["depth"], row["global_variance"])
        model = m.model_label(row["qubits"], row["depth"],
                              row["global_log10_variance"])
        assert knn == model


# ---- gates + seats
def test_hold_gates_14():
    assert len(m.HOLD_GATES) == 14
    assert all(v.startswith("HOLD") for v in m.HOLD_GATES.values())


def test_specific_gates_hold():
    assert m.gate("quantum_advantage_claim") == "HOLD"
    assert m.gate("real_hardware_run") == "HOLD"
    assert m.gate("drug_discovery_claim") == "HOLD"
    assert m.gate("guaranteed_convergence_claim") == "HOLD"


def test_sandbox_proceeds():
    assert m.gate("run_simulation") == "PROCEED"


def test_seats_cannot_override():
    brains = m.brains_report()
    assert brains["decider"]["can_override_hold"] is False
    assert brains["decider"]["seat"] == "Gemini Argon 4 Extended Thinking"
    assert brains["advisor"]["can_override_hold"] is False
    assert brains["flow"]["rendered"] is False
    assert brains["flow"]["live"] == 0


def test_hold_demo_action_is_gated():
    assert m.gate("certified_chemistry_result") == "HOLD"
    assert m.HOLD_GATES["certified_chemistry_result"].startswith("HOLD")


def test_checks_all_pass():
    checks = m.run_checks()
    assert all(ok for _, ok in checks), [n for n, ok in checks if not ok]
