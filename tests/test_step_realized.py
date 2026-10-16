"""
After bolt.step, Data must hold the forward dynamics of the final state for every world: the force breakdown that
users log (limit, spring, damper, muscle, actuator, contact forces) and the derivatives the next step starts from.

Adaptive integrators keep attempting steps until the slowest world reaches the target time, and worlds that finish
early skip the remaining attempts. Their outputs must still describe their final state, so each case is compared
against a fresh full forward pass at the same state.
"""

from __future__ import annotations

import numpy as np
import pytest

import bolt
from bolt._src.pipeline import forward
from models import MODEL_NAMES, STANDING_PELVIS_HEIGHT, model_path

I = bolt.IntegratorType

N_WORLDS = 8
N_STEPS = 3
ADAPTIVE_DT = 5e-3
FIXED_DT = 2e-4  # small enough for the fixed-step integrators to stay stable
INITIAL_SPEED = 0.3  # max |initial generalized velocity|, enough for worlds to need different attempt counts
RTOL = 1e-4  # relative to the largest magnitude of each field

FIELDS = [
    # force breakdown
    "ufrc_limit", "ufrc_spring", "ufrc_damper", "ufrc_muscle", "ufrc_actuator", "qfrc_muscle",
    "body_F_gravity", "body_F_contact", "body_F_muscle", "grf", "geom_cforce",
    # totals and derivatives
    "ufrc_total", "body_F", "qacc", "m_act_dot", "m_state_dot", "a_act_dot",
]


def _snapshot(d) -> dict:
    return {f: getattr(d, f).numpy().reshape(d.nworld, -1).astype(np.float64) for f in FIELDS}


def _assert_matches(got: dict, ref: dict, what: str):
    for f in FIELDS:
        if ref[f].size == 0:
            continue
        atol = RTOL * max(np.abs(ref[f]).max(), 1.0)
        worst = np.abs(got[f] - ref[f]).max(axis=1)
        bad = np.flatnonzero(worst > atol)
        assert bad.size == 0, f"{what}: {f} differs from a fresh forward pass in worlds {bad.tolist()} " \
                              f"(max abs diff {worst.max():.3e}, tol {atol:.3e})"


def _is_adaptive(integrator: bolt.IntegratorType) -> bool:
    return integrator in (I.EULER_ADAPTIVE, I.RK_MERSON_ADAPTIVE)


def _setup(name: str, integrator: bolt.IntegratorType):
    lr = bolt.load_model(model_path=model_path(name), n_worlds=N_WORLDS, integrator=integrator,
                         requires_visuals=False)
    m, d = lr.model, lr.data
    rng = np.random.default_rng(3)
    qpos = d.qpos.numpy()
    qpos[:, lr.qpos_id_lookup["pelvis_ty"]] = STANDING_PELVIS_HEIGHT[name]
    d.qpos.assign(qpos)
    d.qvel.assign(rng.uniform(-INITIAL_SPEED, INITIAL_SPEED, size=(N_WORLDS, m.nv)).astype(np.float32))
    d.world_reset.fill_(True)
    bolt.reset(m, d)
    return m, d, rng


def _new_inputs(m, d, rng):
    if m.nmuscle:
        d.m_excitations.assign(rng.uniform(0.0, 1.0, size=(N_WORLDS, m.nmuscle)).astype(np.float32))
    if m.nactuator:
        d.a_excitations.assign(rng.uniform(0.0, 1.0, size=(N_WORLDS, m.nactuator)).astype(np.float32))


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("integrator", list(I), ids=lambda i: i.name)
def test_step_leaves_state_realized(name, integrator):
    m, d, rng = _setup(name, integrator)
    dt = ADAPTIVE_DT if _is_adaptive(integrator) else FIXED_DT
    attempts_differed = False
    for step in range(N_STEPS):
        _new_inputs(m, d, rng)
        bolt.increment_next_time(m, d, dt)
        bolt.step(m, d)
        assert np.isfinite(d.qpos.numpy()).all() and np.isfinite(d.qvel.numpy()).all(), "state is not finite"
        attempts = d.steps_attempted.numpy()
        attempts_differed |= bool(attempts.min() != attempts.max())

        after_step = _snapshot(d)
        forward.fwd(m, d)
        _assert_matches(after_step, _snapshot(d), f"after step {step}")

    if _is_adaptive(integrator):
        assert attempts_differed, "every world took the same number of attempts, so early finishers were not tested"


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("integrator", [I.EULER_ADAPTIVE, I.RK_MERSON_ADAPTIVE], ids=lambda i: i.name)
def test_realize_inputs_after_step(name, integrator):
    """ The light pass at the start of step() must agree with a full forward pass right after a step """
    m, d, rng = _setup(name, integrator)
    for step in range(N_STEPS):
        bolt.increment_next_time(m, d, ADAPTIVE_DT)
        bolt.step(m, d)
        assert np.isfinite(d.qpos.numpy()).all() and np.isfinite(d.qvel.numpy()).all(), "state is not finite"

        _new_inputs(m, d, rng)
        forward.realize_inputs(m, d)
        light = _snapshot(d)
        forward.fwd(m, d)
        _assert_matches(light, _snapshot(d), f"realize_inputs after step {step}")


def _contacts_per_world(d) -> list[list[tuple]]:
    """ Each world's contacts as sorted (geom pair, distance), from the shared contact list """
    n = min(int(d.nacon.numpy()[0]), d.naconmax)
    worldid, geom, dist = d.contact.worldid.numpy()[:n], d.contact.geom.numpy()[:n], d.contact.dist.numpy()[:n]
    out = [[] for _ in range(d.nworld)]
    for w, g, dd in zip(worldid, geom, dist):
        out[w].append((int(g[0]), int(g[1]), float(dd)))
    return [sorted(c) for c in out]


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("integrator", [I.EULER_ADAPTIVE, I.RK_MERSON_ADAPTIVE], ids=lambda i: i.name)
def test_contact_list_after_step(name, integrator):
    """ The shared contact list must hold every world's contacts, also for worlds that finished integrating early """
    m, d, rng = _setup(name, integrator)
    if m.nxn_geom_pair_filtered.shape[0] == 0:
        pytest.skip("no collision geom pairs")
    saw_contacts = attempts_differed = False
    for step in range(N_STEPS):
        _new_inputs(m, d, rng)
        bolt.increment_next_time(m, d, ADAPTIVE_DT)
        bolt.step(m, d)
        attempts = d.steps_attempted.numpy()
        attempts_differed |= bool(attempts.min() != attempts.max())

        after_step = _contacts_per_world(d)
        forward.fwd(m, d)
        ref = _contacts_per_world(d)
        saw_contacts |= any(ref)
        for w in range(d.nworld):
            assert [c[:2] for c in after_step[w]] == [c[:2] for c in ref[w]], \
                f"after step {step}: world {w} has contacts {after_step[w]}, expected {ref[w]}"
            np.testing.assert_allclose([c[2] for c in after_step[w]], [c[2] for c in ref[w]], atol=1e-6)

    assert saw_contacts, "no contacts were engaged"
    assert attempts_differed, "every world took the same number of attempts, so early finishers were not tested"


def test_reset_restarts_step_size_control():
    """ A reset world starts adaptive stepping like a freshly loaded one, not from its previous episode """
    m, d, rng = _setup(MODEL_NAMES[0], I.EULER_ADAPTIVE)
    for _ in range(N_STEPS):
        _new_inputs(m, d, rng)
        bolt.increment_next_time(m, d, ADAPTIVE_DT)
        bolt.step(m, d)
    initial = np.float32(bolt.INITIAL_STEP_SIZE)  # step sizes are float32
    step_size, prev_error = d.step_size.numpy().copy(), d.prev_error.numpy().copy()
    assert (step_size != initial).any() and (prev_error != 0.0).any(), "stepping changed nothing"

    reset = np.arange(N_WORLDS) % 2 == 0
    d.world_reset.assign(reset)
    bolt.reset(m, d)
    np.testing.assert_array_equal(d.step_size.numpy()[reset], initial)
    np.testing.assert_array_equal(d.prev_error.numpy()[reset], 0.0)
    np.testing.assert_array_equal(d.step_size.numpy()[~reset], step_size[~reset])
    np.testing.assert_array_equal(d.prev_error.numpy()[~reset], prev_error[~reset])
