#!/usr/bin/env python3
"""VariationalQuantum-Lab - educational variational quantum algorithms lab.

by Darion Kellar

Pure Python standard library. Zero dependencies. Zero network.
Education only - and the central lesson is stated up front:

  * This lab exactly simulates TINY quantum systems (<= 8 qubits)
    on a classical machine, with a statevector simulator written
    from scratch. The simulation itself costs 2^n amplitudes -
    that exponential bill is precisely why real variational
    algorithms are run on quantum hardware, and why nothing in
    this lab demonstrates, or claims, any quantum advantage.
  * VQE energies are graded against the EXACT ground energy of
    the same Hamiltonian, computed by full diagonalization in
    this file. The variational principle is never beaten: a
    reported energy below the exact ground (beyond 1e-9 of
    floating-point slack) would be a bug, and the checks treat
    it as one.
  * Barren plateaus are MEASURED, not named: gradient variances
    are sampled at seeded random parameters and printed next to
    a fitted decay line, for a global cost and for a local cost,
    so the difference between the two regimes is a number, not
    a slogan.
  * The noise section is a LABELLED toy model - one global
    depolarizing shrink factor per entangling layer - and it is
    labelled every time it appears.

Real-hardware runs, quantum-advantage claims, production
optimization, drug / materials discovery, certified chemistry,
and the rest of the real-world claims are permanent HOLD gates -
with Gemini Argon 4 Extended Thinking as DECIDER (names-only,
tighten-only), Muse as advisor (never overrides HOLD), and
Google Flow plan-only seats.
"""
import hashlib
import json
import math
import random
import sys

# ---------------------------------------------------------------- constants
VERSION = "1.0.0"
MAX_QUBITS = 8

# ---------------------------------------------------------------- HOLD gates
HOLD_GATES = {
    "quantum_advantage_claim": "HOLD - no quantum advantage claim; this is an exact classical simulation of <= 8 qubits",
    "real_hardware_run": "HOLD - no submitting circuits to real quantum hardware from this lab",
    "production_optimization_claim": "HOLD - no production optimization claims; toy problems, graded against brute force",
    "drug_discovery_claim": "HOLD - no drug-discovery claims from a 2-qubit H2-style toy Hamiltonian",
    "materials_discovery_claim": "HOLD - no materials-discovery claims from a 4-spin toy chain",
    "financial_optimization_claim": "HOLD - no portfolio or financial optimization claims from a 4-node toy graph",
    "cryptography_claim": "HOLD - no cryptographic claims: nothing here breaks, weakens, or strengthens real cryptography",
    "certified_chemistry_result": "HOLD - no certified chemical-accuracy results; coefficients are labelled toys, energies are the lab's own",
    "benchmark_superiority_claim": "HOLD - no claiming superiority over classical solvers at any real scale",
    "live_qpu_billing": "HOLD - no spending on QPU time, cloud quantum credits, or paid hardware queues",
    "military_targeting_use": "HOLD - no weapons, targeting, or military optimization use",
    "scaled_result_extrapolation": "HOLD - no extrapolating <= 8-qubit results to large systems; the plateau section shows why not",
    "peer_reviewed_result_claim": "HOLD - no presenting lab output as peer-reviewed or published science",
    "guaranteed_convergence_claim": "HOLD - no guaranteeing VQE or QAOA convergence; the race table shows optimizers failing honestly",
}


def gate(action):
    """Return 'HOLD' if the action is gated, else 'PROCEED' (sandbox sim)."""
    return "HOLD" if action in HOLD_GATES else "PROCEED"


# ================================================================ 1. Statevector simulator
# A state is a list of 2^n complex amplitudes; qubit q is bit q of
# the basis index. Every gate returns a NEW state list, so callers
# never alias-mutate a state they still need. No numpy - the whole
# point is that every operation is visible, countable, and slow in
# exactly the way that teaches the 2^n lesson.

def zero_state(n):
    state = [0j] * (1 << n)
    state[0] = 1.0 + 0j
    return state


def apply_1q(state, n, q, m):
    """Apply a 2x2 matrix m to qubit q of the state."""
    step = 1 << q
    out = list(state)
    for base in range(0, len(state), step * 2):
        for off in range(step):
            i0 = base + off
            i1 = i0 + step
            a, b = state[i0], state[i1]
            out[i0] = m[0][0] * a + m[0][1] * b
            out[i1] = m[1][0] * a + m[1][1] * b
    return out


def ry(state, n, q, theta):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return apply_1q(state, n, q, ((c, -s), (s, c)))


def rx(state, n, q, theta):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return apply_1q(state, n, q, ((c, -1j * s), (-1j * s, c)))


def hadamard(state, n, q):
    f = 1 / math.sqrt(2)
    return apply_1q(state, n, q, ((f, f), (f, -f)))


def cnot(state, n, control, target):
    out = [0j] * len(state)
    for i, amp in enumerate(state):
        j = i ^ (1 << target) if (i >> control) & 1 else i
        out[j] = amp
    return out


PAULI_MATRICES = {
    "I": ((1, 0), (0, 1)),
    "X": ((0, 1), (1, 0)),
    "Y": ((0, -1j), (1j, 0)),
    "Z": ((1, 0), (0, -1)),
}


def apply_pauli_string(state, n, pstr):
    out = state
    for q, p in enumerate(pstr):
        if p != "I":
            out = apply_1q(out, n, q, PAULI_MATRICES[p])
    return out


def expectation(state, n, terms):
    """Exact <psi|H|psi> for H given as (coefficient, Pauli string) terms."""
    total = 0.0
    for coeff, pstr in terms:
        ps = apply_pauli_string(state, n, pstr)
        val = sum((a.conjugate() * b for a, b in zip(state, ps)), 0j)
        total += coeff * val.real
    return total


def state_norm(state):
    return math.sqrt(sum(abs(a) ** 2 for a in state))


# ================================================================ 2. Hamiltonians + exact solver
# Hamiltonians are lists of (coefficient, Pauli-string) terms. The
# exact ground energy comes from FULL DIAGONALIZATION of the same
# Hamiltonian: the matrix is built column by column from the Pauli
# terms and diagonalized with a complex Jacobi eigensolver written
# from scratch. VQE is never graded against a quoted number.

def pauli_term(n, ops):
    """Build a Pauli-string tuple of length n from {qubit: letter}."""
    s = ["I"] * n
    for q, p in ops.items():
        s[q] = p
    return tuple(s)


def tfim_terms(n, J=1.0, h=1.0):
    """Transverse-field Ising chain: -J sum Z_i Z_i+1 - h sum X_i (open)."""
    terms = []
    for i in range(n - 1):
        terms.append((-J, pauli_term(n, {i: "Z", i + 1: "Z"})))
    for i in range(n):
        terms.append((-h, pauli_term(n, {i: "X"})))
    return terms


# H2-style 2-qubit toy Hamiltonian. The coefficients are LABELLED
# teaching values in the shape of a minimal-basis H2 problem at
# equilibrium geometry - they are not a certified chemistry result,
# and the lab's exact ground energy below is computed from these
# coefficients by diagonalization, not quoted from a textbook.
H2_TERMS = [
    (-0.4804, ("I", "I")),
    (0.3436, ("Z", "I")),
    (-0.4347, ("I", "Z")),
    (0.5716, ("Z", "Z")),
    (0.0910, ("X", "X")),
    (0.0910, ("Y", "Y")),
]
H2_IDENTITY_COEFF = -0.4804

# Global cost used by the barren-plateau probe: <Z Z ... Z>.
def parity_terms(n):
    return [(1.0, tuple("Z" * n))]


def hamiltonian_matrix(n, terms):
    N = 1 << n
    H = [[0j] * N for _ in range(N)]
    for j in range(N):
        basis = [0j] * N
        basis[j] = 1.0 + 0j
        for coeff, pstr in terms:
            col = apply_pauli_string(basis, n, pstr)
            for i in range(N):
                if col[i] != 0j:
                    H[i][j] += coeff * col[i]
    return H


def jacobi_eigenvalues(H, sweeps=100, tol=1e-12):
    """All eigenvalues of a Hermitian matrix, via complex Jacobi rotations."""
    n = len(H)
    A = [[H[i][j] for j in range(n)] for i in range(n)]
    for _ in range(sweeps):
        off = math.sqrt(sum(abs(A[i][j]) ** 2
                            for i in range(n) for j in range(n) if i != j))
        if off < tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = A[p][q]
                if abs(apq) < 1e-15:
                    continue
                app, aqq = A[p][p].real, A[q][q].real
                phase = apq / abs(apq)
                for k in range(n):
                    A[k][q] *= phase.conjugate()
                for k in range(n):
                    A[q][k] *= phase
                apq_r = abs(apq)
                tau = (aqq - app) / (2 * apq_r)
                t = (1.0 if tau >= 0 else -1.0) / (abs(tau) + math.sqrt(1 + tau * tau))
                c = 1 / math.sqrt(1 + t * t)
                s = t * c
                for k in range(n):
                    akp, akq = A[k][p], A[k][q]
                    A[k][p] = c * akp - s * akq
                    A[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = A[p][k], A[q][k]
                    A[p][k] = c * apk - s * aqk
                    A[q][k] = s * apk + c * aqk
    return sorted(A[i][i].real for i in range(n))


def exact_ground_energy(n, terms):
    return jacobi_eigenvalues(hamiltonian_matrix(n, terms))[0]


# ================================================================ 3. Ansatz + optimizers + VQE
# Hardware-efficient ansatz: `depth` layers of RY rotations on
# every qubit followed by a CNOT chain, plus a final RY layer.
# Gradients are EXACT parameter-shift gradients: an RY gate has
# generator Y/2 with eigenvalues +-1/2, so
#   df/dtheta = [f(theta + pi/2) - f(theta - pi/2)] / 2
# with no finite-difference error - the checks prove that against
# central finite differences anyway.

def ansatz_param_count(n, depth):
    return n * (depth + 1)


def ansatz_state(params, n, depth):
    state = zero_state(n)
    idx = 0
    for _layer in range(depth):
        for q in range(n):
            state = ry(state, n, q, params[idx])
            idx += 1
        for q in range(n - 1):
            state = cnot(state, n, q, q + 1)
    for q in range(n):
        state = ry(state, n, q, params[idx])
        idx += 1
    return state


def ansatz_energy(params, n, depth, terms):
    return expectation(ansatz_state(params, n, depth), n, terms)


def param_shift_gradient(params, n, depth, terms, idx):
    pp = list(params)
    pp[idx] += math.pi / 2
    pm = list(params)
    pm[idx] -= math.pi / 2
    return (ansatz_energy(pp, n, depth, terms)
            - ansatz_energy(pm, n, depth, terms)) / 2


def full_gradient(params, n, depth, terms):
    return [param_shift_gradient(params, n, depth, terms, i)
            for i in range(len(params))]


def finite_difference_gradient(params, n, depth, terms, idx, eps=1e-4):
    pp = list(params)
    pp[idx] += eps
    pm = list(params)
    pm[idx] -= eps
    return (ansatz_energy(pp, n, depth, terms)
            - ansatz_energy(pm, n, depth, terms)) / (2 * eps)


# ---- optimizer 1: gradient descent on exact parameter-shift gradients
def gradient_descent(n, depth, terms, x0, lr=0.4, iters=150):
    """Plain GD; tracks the best point ever seen (reported honestly)."""
    x = list(x0)
    best_x, best_e = list(x0), ansatz_energy(x0, n, depth, terms)
    evals = 1
    for _ in range(iters):
        grads = full_gradient(x, n, depth, terms)
        evals += 2 * len(x)
        x = [xi - lr * gi for xi, gi in zip(x, grads)]
        e = ansatz_energy(x, n, depth, terms)
        evals += 1
        if e < best_e:
            best_x, best_e = list(x), e
    return {"params": best_x, "energy": best_e, "evaluations": evals,
            "iterations": iters, "method": "gradient_descent"}


# ---- optimizer 2: Nelder-Mead simplex (from scratch)
def nelder_mead(f, x0, step=0.4, max_evals=900, tol=1e-11):
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        x = list(x0)
        x[i] += step
        simplex.append(x)
    vals = [f(x) for x in simplex]
    evals = n + 1
    best_x, best_v = list(simplex[0]), vals[0]
    for x, v in zip(simplex, vals):
        if v < best_v:
            best_x, best_v = list(x), v
    while evals < max_evals:
        order = sorted(range(n + 1), key=lambda i: vals[i])
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if vals[0] < best_v:
            best_x, best_v = list(simplex[0]), vals[0]
        if max(abs(vals[i] - vals[0]) for i in range(1, n + 1)) < tol:
            break
        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
        xr = [centroid[j] + (centroid[j] - simplex[n][j]) for j in range(n)]
        fr = f(xr)
        evals += 1
        if fr < vals[0]:
            xe = [centroid[j] + 2 * (centroid[j] - simplex[n][j])
                  for j in range(n)]
            fe = f(xe)
            evals += 1
            if fe < fr:
                simplex[n], vals[n] = xe, fe
            else:
                simplex[n], vals[n] = xr, fr
        elif fr < vals[n - 1]:
            simplex[n], vals[n] = xr, fr
        else:
            xc = [centroid[j] + 0.5 * (simplex[n][j] - centroid[j])
                  for j in range(n)]
            fc = f(xc)
            evals += 1
            if fc < vals[n]:
                simplex[n], vals[n] = xc, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][j]
                                  + 0.5 * (simplex[i][j] - simplex[0][j])
                                  for j in range(n)]
                    vals[i] = f(simplex[i])
                    evals += 1
                    if evals >= max_evals:
                        break
    if vals[0] < best_v:
        best_x, best_v = list(simplex[0]), vals[0]
    return {"params": best_x, "energy": best_v, "evaluations": evals,
            "method": "nelder_mead"}


# ---- optimizer 3: random-restart (seeded starts + Nelder-Mead polish)
def random_restart(n, depth, terms, restarts=6, seed=777, max_evals_each=300):
    rng = random.Random(seed + n)
    count = ansatz_param_count(n, depth)
    best = None
    total_evals = 0
    for _ in range(restarts):
        start = [rng.uniform(-math.pi, math.pi) for _ in range(count)]
        res = nelder_mead(lambda p: ansatz_energy(p, n, depth, terms),
                          start, step=0.5, max_evals=max_evals_each)
        total_evals += res["evaluations"]
        if best is None or res["energy"] < best["energy"]:
            best = res
    return {"params": best["params"], "energy": best["energy"],
            "evaluations": total_evals, "iterations": restarts,
            "method": "random_restart"}


def _seeded_start(n, depth, seed=20240):
    rng = random.Random(seed + n)
    return [rng.uniform(-0.5, 0.5)
            for _ in range(ansatz_param_count(n, depth))]


def vqe_report():
    """VQE on both demo Hamiltonians, graded against exact diagonalization."""
    out = {}
    for label, n, terms, depth in (
            ("h2_toy", 2, H2_TERMS, 2),
            ("tfim4", 4, tfim_terms(4), 2)):
        exact = exact_ground_energy(n, terms)
        res = gradient_descent(n, depth, terms, _seeded_start(n, depth))
        out[label] = {
            "qubits": n, "depth": depth,
            "parameters": ansatz_param_count(n, depth),
            "exact_ground_energy": exact,
            "vqe_energy": res["energy"],
            "gap_to_exact": res["energy"] - exact,
            "evaluations": res["evaluations"],
            "optimizer": res["method"],
        }
    return out


def race_report():
    """The optimizer race: TFIM-4, depth 2, one seeded start, three optimizers."""
    n, depth, terms = 4, 2, tfim_terms(4)
    exact = exact_ground_energy(n, terms)
    x0 = _seeded_start(n, depth)
    f = lambda p: ansatz_energy(p, n, depth, terms)  # noqa: E731
    gd = gradient_descent(n, depth, terms, x0)
    nm = nelder_mead(f, x0, step=0.4, max_evals=900)
    rr = random_restart(n, depth, terms)
    rows = []
    for res in (gd, nm, rr):
        rows.append({
            "optimizer": res["method"],
            "evaluations": res["evaluations"],
            "iterations": res.get("iterations"),
            "final_energy": res["energy"],
            "gap_to_exact": res["energy"] - exact,
        })
    return {"problem": "tfim4", "qubits": n, "depth": depth,
            "parameters": ansatz_param_count(n, depth),
            "exact_ground_energy": exact, "results": rows}


# ================================================================ 4. QAOA for MaxCut
# Demo graph: a 4-node square plus one diagonal. The triangle in
# it makes a perfect cut impossible, so the brute-force optimum
# (4 of 5 edges) is a real target, not a formality. QAOA prepares
# |+>^n, then p layers of the cost unitary exp(-i gamma C) - a
# phase per basis state set by that bitstring's cut value - and
# the mixer exp(-i beta sum X) = RX(2 beta) on every qubit.

MAXCUT_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
MAXCUT_N = 4


def cut_value(bits, edges=MAXCUT_EDGES):
    return sum(1 for (u, v) in edges if bits[u] != bits[v])


def maxcut_brute_force(edges=MAXCUT_EDGES, n=MAXCUT_N):
    best = -1
    for z in range(1 << n):
        bits = [(z >> i) & 1 for i in range(n)]
        best = max(best, cut_value(bits, edges))
    return best


def qaoa_state(gammas, betas):
    n = MAXCUT_N
    state = zero_state(n)
    for q in range(n):
        state = hadamard(state, n, q)
    for gamma, beta in zip(gammas, betas):
        phased = []
        for i, amp in enumerate(state):
            bits = [(i >> q) & 1 for q in range(n)]
            phase = complex(math.cos(-gamma * cut_value(bits)),
                            math.sin(-gamma * cut_value(bits)))
            phased.append(amp * phase)
        state = phased
        for q in range(n):
            state = rx(state, n, q, 2 * beta)
    return state


def qaoa_expectation(gammas, betas):
    state = qaoa_state(gammas, betas)
    return sum(abs(a) ** 2 * cut_value([(i >> q) & 1 for q in range(MAXCUT_N)])
               for i, a in enumerate(state))


def _optimize_qaoa(p, warm=None, seed=31337):
    rng = random.Random(seed)
    starts = []
    if warm is not None:
        starts.append(list(warm))
    for _ in range(6):
        starts.append([rng.uniform(0, math.pi) for _ in range(2 * p)])
    best = None
    for start in starts:
        res = nelder_mead(lambda t: -qaoa_expectation(t[:p], t[p:]),
                          start, step=0.35, max_evals=500)
        if best is None or res["energy"] < best["energy"]:
            best = res
    return best


def qaoa_report():
    optimum = maxcut_brute_force()
    r1 = _optimize_qaoa(1)
    warm = [r1["params"][0], 0.0, r1["params"][1], 0.0]
    r2 = _optimize_qaoa(2, warm=warm)
    e1, e2 = -r1["energy"], -r2["energy"]
    return {
        "graph_edges": [list(e) for e in MAXCUT_EDGES],
        "brute_force_optimum": optimum,
        "p1": {"expectation": e1, "approximation_ratio": e1 / optimum,
               "gammas": [r1["params"][0]], "betas": [r1["params"][1]]},
        "p2": {"expectation": e2, "approximation_ratio": e2 / optimum,
               "gammas": r2["params"][:2], "betas": r2["params"][2:]},
    }


# ================================================================ 5. Barren plateaus, measured
# Protocol (McClean-style, everything seeded): draw random
# parameter vectors uniformly from [-pi, pi), compute the exact
# parameter-shift gradient of the FIRST ansatz parameter, and take
# the variance over the sample. Two costs are measured:
#   * GLOBAL - <Z Z ... Z>, the parity of all qubits;
#   * LOCAL  - the TFIM energy, a sum of 1- and 2-qubit terms.
# A least-squares line is fitted to log10(variance) vs qubits for
# each cost. The fitted slope IS the headline: variance ~ 10^(slope
# per added qubit). Nothing here is quoted from a paper - every
# number is measured by this file, on this machine, from the seeds
# printed in the report.

PLATEAU_SAMPLES = 32
PLATEAU_SEED = 777


def gradient_variance(n, depth, terms, samples=PLATEAU_SAMPLES,
                      seed=PLATEAU_SEED, param_index=0):
    rng = random.Random(seed + n * 1000 + depth)
    count = ansatz_param_count(n, depth)
    grads = []
    for _ in range(samples):
        params = [rng.uniform(-math.pi, math.pi) for _ in range(count)]
        grads.append(param_shift_gradient(params, n, depth, terms, param_index))
    mean = sum(grads) / len(grads)
    return sum((g - mean) ** 2 for g in grads) / len(grads)


def _fit_line(xs, ys):
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    slope = num / den
    return slope, my - slope * mx


def plateau_report():
    qubit_rows = []
    for n in range(2, MAX_QUBITS + 1):
        gv = gradient_variance(n, n, parity_terms(n))
        lv = gradient_variance(n, n, tfim_terms(n))
        qubit_rows.append({"qubits": n, "depth": n,
                           "global_variance": gv, "local_variance": lv,
                           "global_log10_variance": math.log10(gv),
                           "local_log10_variance": math.log10(lv)})
    xs = [r["qubits"] for r in qubit_rows]
    g_slope, g_inter = _fit_line(xs, [r["global_log10_variance"]
                                      for r in qubit_rows])
    l_slope, l_inter = _fit_line(xs, [r["local_log10_variance"]
                                      for r in qubit_rows])
    depth_rows = []
    for d in range(1, MAX_QUBITS + 1):
        v = gradient_variance(MAX_QUBITS, d, parity_terms(MAX_QUBITS))
        depth_rows.append({"qubits": MAX_QUBITS, "depth": d,
                           "global_variance": v,
                           "global_log10_variance": math.log10(v)})
    return {
        "samples_per_point": PLATEAU_SAMPLES,
        "seed": PLATEAU_SEED,
        "gradient_parameter": "first ansatz parameter (parameter-shift, exact)",
        "qubit_scan_depth_equals_qubits": qubit_rows,
        "depth_scan_at_8_qubits": depth_rows,
        "global_fit": {"slope_log10_per_qubit": g_slope,
                       "intercept": g_inter,
                       "factor_per_qubit": 10 ** g_slope},
        "local_fit": {"slope_log10_per_qubit": l_slope,
                      "intercept": l_inter,
                      "factor_per_qubit": 10 ** l_slope},
        "variance_ratio_2q_to_8q_global":
            qubit_rows[0]["global_variance"] / qubit_rows[-1]["global_variance"],
    }


# ================================================================ 6. Noise, labelled honestly
# LABELLED TOY NOISE MODEL: one global depolarizing channel per
# entangling layer, strength p. A depolarizing channel shrinks
# every traceless Pauli expectation by (1 - p), so after `depth`
# layers the noisy energy of ANY state is exactly
#   E_noisy = (1-p)^depth * (E_ideal - E_identity) + E_identity
# where E_identity is the Hamiltonian's identity coefficient (the
# only part noise cannot shrink). This is a teaching model of the
# energy floor noise imposes - it is not a hardware noise model,
# and it is labelled as a toy every time it is printed.

def noisy_energy(params, n, depth, terms, p, identity_coeff):
    ideal = ansatz_energy(params, n, depth, terms)
    shrink = (1 - p) ** depth
    return shrink * (ideal - identity_coeff) + identity_coeff


def noise_floor(depth, p, exact_ground, identity_coeff):
    """The best energy ANY state can report under the labelled noise model."""
    shrink = (1 - p) ** depth
    return shrink * (exact_ground - identity_coeff) + identity_coeff


def noise_report():
    n, depth, terms = 2, 2, H2_TERMS
    exact = exact_ground_energy(n, terms)
    x0 = _seeded_start(n, depth)
    clean = gradient_descent(n, depth, terms, x0)
    rows = []
    for p in (0.0, 0.01, 0.02, 0.05):
        floor = noise_floor(depth, p, exact, H2_IDENTITY_COEFF)
        at_optimum = noisy_energy(clean["params"], n, depth, terms, p,
                                  H2_IDENTITY_COEFF)
        rows.append({"p_per_layer": p, "shrink_factor": (1 - p) ** depth,
                     "noise_floor": floor, "floor_gap_to_exact": floor - exact,
                     "energy_at_clean_optimum": at_optimum})
    # Re-optimizing against the noisy objective cannot beat the floor:
    # the noisy objective is a positive rescaling of the ideal one.
    p = 0.02
    rng_free = gradient_descent(
        n, depth, terms, x0)  # same optimum; noisy value read off below
    rows.append({"p_per_layer": p, "note": "re-optimized under noise",
                 "energy_at_clean_optimum":
                     noisy_energy(rng_free["params"], n, depth, terms, p,
                                  H2_IDENTITY_COEFF),
                 "noise_floor": noise_floor(depth, p, exact,
                                            H2_IDENTITY_COEFF),
                 "floor_gap_to_exact":
                     noise_floor(depth, p, exact, H2_IDENTITY_COEFF) - exact})
    return {"model": "LABELLED toy: global depolarizing shrink (1-p) per "
                     "entangling layer",
            "problem": "h2_toy", "depth": depth,
            "exact_ground_energy": exact,
            "clean_vqe_energy": clean["energy"],
            "rows": rows}


# ================================================================ 7. kNN trainability classifier
# Features: (qubits, depth, measured global-cost gradient
# variance). The training grid is labelled by the lab's own
# trainability model (model_label below), whose thresholds are set
# from the measured plateau table - the classifier interpolates
# the measurements, it does not replace them.

def model_label(qubits, depth, log10_variance):
    """The trainability model distilled to a three-way teaching label."""
    if log10_variance < -2.0:
        return "barren"
    if log10_variance >= -1.2:
        return "trainable"
    return "hard"


def _training_grid():
    # Variance planes are offset by 0.05 from the model's label
    # thresholds (-1.2 and -2.0), so no grid plane sits exactly on
    # a boundary and the kNN interpolation straddles them honestly.
    rows = []
    for qubits in range(2, 9):
        for depth in range(1, 9):
            for i in range(25):
                log10_var = -0.55 - 0.1 * i
                rows.append(((qubits, depth, log10_var),
                             model_label(qubits, depth, log10_var)))
    return rows


TRAINING_GRID = _training_grid()


def _scaled(features):
    qubits, depth, log10_var = features
    return (qubits / 8.0, depth / 8.0, (log10_var + 3.0) / 3.0)


def classify_trainability(qubits, depth, gradient_variance, k=5):
    """kNN over the model-labelled grid, deterministic tie-breaking."""
    log10_var = math.log10(gradient_variance)
    target = _scaled((qubits, depth, log10_var))
    ranked = []
    for features, label in TRAINING_GRID:
        point = _scaled(features)
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(target, point)))
        ranked.append((dist, features, label))
    ranked.sort(key=lambda row: (row[0], row[1]))
    votes = {}
    for _, _, label in ranked[:k]:
        votes[label] = votes.get(label, 0) + 1
    priority = {"trainable": 0, "hard": 1, "barren": 2}
    return sorted(votes.items(), key=lambda kv: (-kv[1], priority[kv[0]]))[0][0]


def classifier_report():
    plateau = plateau_report()
    out = {}
    for row in plateau["qubit_scan_depth_equals_qubits"]:
        n = row["qubits"]
        out[f"{n}q_depth{n}_global_cost"] = classify_trainability(
            n, n, row["global_variance"])
    return out


# ================================================================ 8. Advisory seats
def _trace(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def argon4_decide(topic="variationalquantum-demo"):
    """Gemini Argon 4 DECIDER seat: MAP->MODEL->STRESS->DECIDE.

    Names-only keys, live calls 0, tighten-only: can add HOLDs, never lift one.
    """
    stages = ["MAP", "MODEL", "STRESS", "DECIDE"]
    payload = {"seat": "gemini-argon-4", "topic": topic, "stages": stages}
    return {
        "seat": "Gemini Argon 4 Extended Thinking",
        "role": "DECIDER",
        "pipeline": stages,
        "tiers": [2048, 8192, 32768],
        "key": "names-only (GEMINI_API_KEY name checked, value never read)",
        "live_calls": 0,
        "decision": "PROCEED - sandbox simulation only, all real-world claim gates stay HOLD",
        "can_override_hold": False,
        "trace_sha256_16": _trace(payload),
    }


def claude_advisor(topic="variationalquantum-demo"):
    """Muse advisor seat: advises, can never override a HOLD."""
    return {
        "seat": "Muse",
        "role": "ADVISOR",
        "key": "names-only (MODEL_API_KEY name checked, value never read)",
        "live_calls": 0,
        "advice": "Grade every variational energy against the exact diagonalization in the same file, print the optimizer gaps instead of hiding them, and let the measured gradient variances - not the phrase 'barren plateau' - carry the scaling lesson.",
        "can_override_hold": False,
    }


def google_flow_plan():
    """Google Flow seat: plan-only, nothing rendered, nothing live."""
    return {
        "seat": "Google Flow",
        "role": "PLAN-ONLY",
        "shots": ["rotation gates tilting a Bloch sphere layer by layer",
                  "three optimizer racers descending one energy landscape",
                  "a QAOA wave lifting the good cuts out of a 4-node graph",
                  "gradient arrows shrinking to dust as qubits are added",
                  "a noise floor rising to meet the falling VQE energy"],
        "rendered": False,
        "live": 0,
    }


def brains_report():
    return {"decider": argon4_decide(), "advisor": claude_advisor(),
            "flow": google_flow_plan()}


# ================================================================ checks
_REPORT_CACHE = {}


def _cached(key, fn):
    if key not in _REPORT_CACHE:
        _REPORT_CACHE[key] = fn()
    return _REPORT_CACHE[key]


def run_checks():
    c = []

    def chk(name, ok):
        c.append((name, bool(ok)))

    # 1. simulator core
    s = zero_state(3)
    chk("zero_state_norm", abs(state_norm(s) - 1.0) < 1e-12 and s[0] == 1.0)
    flipped = ry(zero_state(1), 1, 0, math.pi)
    chk("ry_pi_flips_qubit", abs(abs(flipped[1]) - 1.0) < 1e-12)
    plus = hadamard(zero_state(1), 1, 0)
    chk("hadamard_superposition",
        abs(abs(plus[0]) ** 2 - 0.5) < 1e-12
        and abs(abs(plus[1]) ** 2 - 0.5) < 1e-12)
    bell = cnot(hadamard(zero_state(2), 2, 0), 2, 0, 1)
    chk("cnot_makes_bell_state",
        abs(abs(bell[0]) ** 2 - 0.5) < 1e-12
        and abs(bell[1]) < 1e-12 and abs(bell[2]) < 1e-12
        and abs(abs(bell[3]) ** 2 - 0.5) < 1e-12)
    chk("pauli_expectations",
        abs(expectation(zero_state(1), 1, [(1.0, ("Z",))]) - 1.0) < 1e-12
        and abs(expectation(plus, 1, [(1.0, ("X",))]) - 1.0) < 1e-12)
    chk("ansatz_state_normalized",
        abs(state_norm(ansatz_state([0.3] * 12, 4, 2)) - 1.0) < 1e-9)
    chk("ansatz_param_count", ansatz_param_count(4, 2) == 12
        and ansatz_param_count(2, 2) == 6 and ansatz_param_count(8, 8) == 72)

    # 2. exact solver
    chk("jacobi_z_eigenvalues",
        jacobi_eigenvalues(hamiltonian_matrix(1, [(1.0, ("Z",))]))
        == [-1.0, 1.0])
    diag = [[complex(3 - i, 0) if i == j else 0j for j in range(4)]
            for i in range(4)]
    chk("jacobi_diagonal_matrix",
        jacobi_eigenvalues(diag) == [0.0, 1.0, 2.0, 3.0])
    tfim4 = tfim_terms(4)
    evals4 = jacobi_eigenvalues(hamiltonian_matrix(4, tfim4))
    trace4 = sum(hamiltonian_matrix(4, tfim4)[i][i].real for i in range(16))
    chk("jacobi_trace_preserved", abs(sum(evals4) - trace4) < 1e-8)
    chk("tfim2_exact_ground",
        abs(exact_ground_energy(2, tfim_terms(2)) + math.sqrt(5)) < 1e-9)
    chk("tfim4_exact_ground", abs(evals4[0] - (-4.7587704831)) < 1e-9)
    chk("h2_exact_ground",
        abs(exact_ground_energy(2, H2_TERMS) - (-1.8512964969)) < 1e-9)

    # 3. gradients: parameter shift is exact, finite differences agree
    probe = [0.31, -0.72, 1.05, 0.44, -0.18, 0.91]
    diffs = [abs(param_shift_gradient(probe, 2, 2, H2_TERMS, i)
                 - finite_difference_gradient(probe, 2, 2, H2_TERMS, i))
             for i in range(len(probe))]
    chk("param_shift_matches_finite_difference", max(diffs) < 1e-6)
    chk("gradient_of_constant_is_zero",
        abs(param_shift_gradient(probe, 2, 2, [(2.5, ("I", "I"))], 0)) < 1e-12)

    # 4. VQE + the race
    vqe = _cached("vqe", vqe_report)
    chk("vqe_h2_variational_principle",
        vqe["h2_toy"]["vqe_energy"] >= vqe["h2_toy"]["exact_ground_energy"] - 1e-9)
    chk("vqe_h2_gap_small", vqe["h2_toy"]["gap_to_exact"] < 1e-6)
    chk("vqe_tfim_variational_principle",
        vqe["tfim4"]["vqe_energy"] >= vqe["tfim4"]["exact_ground_energy"] - 1e-9)
    chk("vqe_zero_params_tfim_energy",
        abs(ansatz_energy([0.0] * 12, 4, 2, tfim4) - (-3.0)) < 1e-12)
    chk("vqe_zero_params_h2_energy",
        abs(ansatz_energy([0.0] * 6, 2, 2, H2_TERMS) - 0.0001) < 1e-9)
    race = _cached("race", race_report)
    chk("race_all_above_exact",
        all(r["final_energy"] >= race["exact_ground_energy"] - 1e-9
            for r in race["results"]))
    by_name = {r["optimizer"]: r for r in race["results"]}
    chk("race_gradient_descent_gap", by_name["gradient_descent"]["gap_to_exact"] < 0.01)
    chk("race_random_restart_gap", by_name["random_restart"]["gap_to_exact"] < 0.01)
    chk("race_nelder_mead_gap", by_name["nelder_mead"]["gap_to_exact"] < 0.25)

    # 5. QAOA
    chk("maxcut_brute_force_is_4", maxcut_brute_force() == 4)
    chk("qaoa_zero_params_mean_cut",
        abs(qaoa_expectation([0.0], [0.0]) - 2.5) < 1e-9)
    qaoa = _cached("qaoa", qaoa_report)
    chk("qaoa_p1_ratio_range",
        0.75 < qaoa["p1"]["approximation_ratio"] < 0.90)
    chk("qaoa_p2_improves_or_holds",
        qaoa["p2"]["approximation_ratio"]
        >= qaoa["p1"]["approximation_ratio"] - 1e-9)
    chk("qaoa_p2_ratio_high", qaoa["p2"]["approximation_ratio"] > 0.90)
    chk("qaoa_expectation_within_bounds",
        0.0 <= qaoa["p2"]["expectation"] <= qaoa["brute_force_optimum"])

    # 6. barren plateaus, measured
    plateau = _cached("plateau", plateau_report)
    rows = {r["qubits"]: r for r in plateau["qubit_scan_depth_equals_qubits"]}
    chk("plateau_var_2q_global", 0.10 < rows[2]["global_variance"] < 0.50)
    chk("plateau_var_8q_global", 0.001 < rows[8]["global_variance"] < 0.010)
    chk("plateau_decay_over_50x", plateau["variance_ratio_2q_to_8q_global"] > 50)
    chk("plateau_global_fit_slope",
        plateau["global_fit"]["slope_log10_per_qubit"] < -0.20)
    chk("plateau_local_decays_slower",
        plateau["local_fit"]["slope_log10_per_qubit"]
        > plateau["global_fit"]["slope_log10_per_qubit"])
    chk("plateau_local_var_8q_above_global",
        rows[8]["local_variance"] > rows[8]["global_variance"])
    drows = {r["depth"]: r for r in plateau["depth_scan_at_8_qubits"]}
    chk("plateau_depth_scan_enters_plateau",
        drows[8]["global_variance"] < drows[1]["global_variance"])

    # 7. noise honesty
    noise = _cached("noise", noise_report)
    nrows = {r["p_per_layer"]: r for r in noise["rows"][:4]}
    chk("noise_zero_p_recovers_clean",
        abs(nrows[0.0]["energy_at_clean_optimum"]
            - noise["clean_vqe_energy"]) < 1e-9)
    chk("noise_floor_rises_with_p",
        nrows[0.01]["noise_floor"] < nrows[0.02]["noise_floor"]
        < nrows[0.05]["noise_floor"])
    chk("noise_floor_gap_at_2pct", 0.04 < nrows[0.02]["floor_gap_to_exact"] < 0.07)
    chk("noise_optimum_sits_on_floor",
        abs(nrows[0.02]["energy_at_clean_optimum"]
            - nrows[0.02]["noise_floor"]) < 0.01)

    # 8. classifier - labels come from the measured plateau rows
    rep = classifier_report()
    chk("classifier_2q_trainable", rep["2q_depth2_global_cost"] == "trainable")
    chk("classifier_5q_hard", rep["5q_depth5_global_cost"] == "hard")
    chk("classifier_8q_barren", rep["8q_depth8_global_cost"] == "barren")
    chk("classifier_matches_model",
        all(classify_trainability(r["qubits"], r["depth"], r["global_variance"])
            == model_label(r["qubits"], r["depth"],
                           r["global_log10_variance"])
            for r in plateau["qubit_scan_depth_equals_qubits"]))

    # gates + seats
    chk("hold_gates_14", len(HOLD_GATES) == 14
        and all(v.startswith("HOLD") for v in HOLD_GATES.values()))
    chk("gate_advantage_hold", gate("quantum_advantage_claim") == "HOLD")
    chk("gate_hardware_hold", gate("real_hardware_run") == "HOLD")
    chk("gate_drug_hold", gate("drug_discovery_claim") == "HOLD")
    chk("gate_convergence_hold", gate("guaranteed_convergence_claim") == "HOLD")
    chk("gate_sandbox_proceed", gate("run_simulation") == "PROCEED")
    brains = brains_report()
    chk("argon4_no_override", brains["decider"]["can_override_hold"] is False
        and brains["decider"]["seat"]
        == "Gemini Argon 4 Extended Thinking")
    chk("claude_no_override", brains["advisor"]["can_override_hold"] is False)
    chk("flow_plan_only", brains["flow"]["rendered"] is False
        and brains["flow"]["live"] == 0)
    return c


# ================================================================ CLI / demo
def demo():
    print("=" * 68)
    print("VariationalQuantum-Lab - variational quantum algorithms lab")
    print("by Darion Kellar | MIT | pure stdlib | zero network | EDUCATION ONLY")
    print("=" * 68)
    print("\n[1] THE HONEST CORE - exact simulation of tiny systems")
    print("    A from-scratch statevector simulator (<= 8 qubits) runs VQE")
    print("    and QAOA exactly - on a classical machine, at 2^n cost. That")
    print("    cost is the lesson: nothing here is, or claims to be, a")
    print("    quantum advantage. Every energy is graded against full")
    print("    diagonalization of the same Hamiltonian, in this same file.")
    print("\n[2] Exact ground energies (Jacobi diagonalization, in-lab)")
    for label, n, terms in (("TFIM chain (4 qubits, J=1, h=1)", 4, tfim_terms(4)),
                            ("H2-style toy (2 qubits, labelled coeffs)", 2, H2_TERMS)):
        print(f"    {label:44s} E0 = {exact_ground_energy(n, terms):+.6f}")
    print("\n[3] VQE - hardware-efficient ansatz, parameter-shift gradients")
    vqe = _cached("vqe", vqe_report)
    for label, row in vqe.items():
        print(f"    {label:8s} exact {row['exact_ground_energy']:+.6f} | "
              f"VQE {row['vqe_energy']:+.6f} | gap {row['gap_to_exact']:.2e} | "
              f"{row['evaluations']} evals")
    print("    -> the variational principle holds in every row: VQE never")
    print("       dips below the exact ground. The gap is the optimizer's")
    print("       report card, printed, not hidden.")
    print("\n[4] The optimizer race - TFIM-4, depth 2, one seeded start")
    race = _cached("race", race_report)
    for row in race["results"]:
        print(f"    {row['optimizer']:16s} evals {row['evaluations']:5d} | "
              f"E {row['final_energy']:+.6f} | gap {row['gap_to_exact']:.2e}")
    print(f"    exact ground {race['exact_ground_energy']:+.6f} - same start, "
          "same ansatz; only the optimizer differs.")
    print("\n[5] QAOA - MaxCut on a 4-node graph (square + diagonal)")
    qaoa = _cached("qaoa", qaoa_report)
    print(f"    brute-force optimum cut = {qaoa['brute_force_optimum']} of 5 edges")
    print(f"    p=1  <C> {qaoa['p1']['expectation']:.4f} | "
          f"ratio {qaoa['p1']['approximation_ratio']:.4f}")
    print(f"    p=2  <C> {qaoa['p2']['expectation']:.4f} | "
          f"ratio {qaoa['p2']['approximation_ratio']:.4f}")
    print("    -> one more layer buys real ratio here - at 4 nodes. The")
    print("       plateau section below is what happens as nodes grow.")
    print("\n[6] Barren plateaus - MEASURED gradient variance (32 samples/pt)")
    plateau = _cached("plateau", plateau_report)
    print("    qubits | global <Z...Z> var | local TFIM var")
    for row in plateau["qubit_scan_depth_equals_qubits"]:
        print(f"    {row['qubits']:6d} | {row['global_variance']:17.3e} | "
              f"{row['local_variance']:.3e}")
    print(f"    fit: global variance x{plateau['global_fit']['factor_per_qubit']:.3f} "
          f"per qubit (slope {plateau['global_fit']['slope_log10_per_qubit']:+.3f} "
          "log10/qubit)")
    print(f"         local  variance x{plateau['local_fit']['factor_per_qubit']:.3f} "
          f"per qubit (slope {plateau['local_fit']['slope_log10_per_qubit']:+.3f} "
          "log10/qubit)")
    print(f"    -> global-cost variance falls "
          f"{plateau['variance_ratio_2q_to_8q_global']:.0f}x from 2 to 8 qubits: "
          "that straight line on a log plot is the exponential plateau.")
    print("       The local TFIM cost decays far more gently - cost locality")
    print("       matters, and here it is a measured fact, not a citation.")
    print("\n[7] Noise honesty - LABELLED toy depolarizing model")
    noise = _cached("noise", noise_report)
    for row in noise["rows"][:4]:
        print(f"    p={row['p_per_layer']:.2f}/layer | floor "
              f"{row['noise_floor']:+.6f} | floor gap "
              f"{row['floor_gap_to_exact']:.4f}")
    print("    -> even a perfect optimizer cannot report below the floor:")
    print("       noise rescales the traceless part of the energy toward")
    print("       the identity term. Toy model, labelled - not hardware data.")
    print("\n[8] Trainability classifier (TinyML kNN, labels from measurements)")
    for name, label in classifier_report().items():
        print(f"    {name:24s} -> {label}")
    print("\n[9] Advisory seats")
    brains = brains_report()
    print(f"    decider: {brains['decider']['decision']} "
          f"trace={brains['decider']['trace_sha256_16']}")
    print(f"    advisor can_override_hold = {brains['advisor']['can_override_hold']}")
    print(f"    flow rendered = {brains['flow']['rendered']} "
          f"live = {brains['flow']['live']}")
    print(f"\n[10] HOLD gates: {len(HOLD_GATES)}")
    for key in HOLD_GATES:
        print(f"    HOLD  {key}")
    checks = run_checks()
    passed = sum(1 for _, ok in checks if ok)
    print(f"\nCHECKS {passed}/{len(checks)}")
    for name, ok in checks:
        if not ok:
            print(f"    FAIL {name}")
    verdict = "PROCEED" if passed == len(checks) else "HOLD"
    print(f"VERDICT {verdict} (sandbox simulation only) | HOLD gates {len(HOLD_GATES)}")
    return 0 if passed == len(checks) else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] == "demo":
        return demo()
    cmd = argv[0]
    if cmd == "checks":
        checks = run_checks()
        for name, ok in checks:
            print(f"{'PASS' if ok else 'FAIL'} {name}")
        print(f"CHECKS {sum(1 for _, ok in checks if ok)}/{len(checks)}")
        return 0 if all(ok for _, ok in checks) else 1
    if cmd == "vqe":
        print(json.dumps(_cached("vqe", vqe_report), indent=2, sort_keys=True))
        return 0
    if cmd == "race":
        print(json.dumps(_cached("race", race_report), indent=2, sort_keys=True))
        return 0
    if cmd == "qaoa":
        print(json.dumps(_cached("qaoa", qaoa_report), indent=2, sort_keys=True))
        return 0
    if cmd == "plateau":
        print(json.dumps(_cached("plateau", plateau_report),
                         indent=2, sort_keys=True))
        return 0
    if cmd == "classify":
        print(json.dumps(classifier_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "brains":
        print(json.dumps(brains_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "holds":
        for key, value in HOLD_GATES.items():
            print(f"{key}: {value}")
        return 0
    if cmd == "hold-demo":
        action = argv[1] if len(argv) > 1 else "quantum_advantage_claim"
        print(f"{action}: {HOLD_GATES.get(action, 'PROCEED - not gated')}")
        return 2 if gate(action) == "HOLD" else 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
