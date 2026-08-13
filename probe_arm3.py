"""Internalization probe -- Arm 3: hidden-state DECODE + ABLATE.
Per PROBE_PREREGISTRATION.md (predictions fixed before run). For a norm:
  1. train the flagship run (install + ghost, learning-on) -> ghost-end policy.
  2. roll the ghost-end policy through a ghost episode, logging per-agent GRU carry
     + per-agent marked-berry opportunity behavior (the norm label).
  3. DECODE: linear-probe carry -> per-agent norm behavior. R^2 = decodability.
  4. ABLATE: re-roll projecting the decoded direction out of the carry each step;
     measure how far the norm behavior moves. Random-direction ablation = control.
Predictions: grounded -> high R^2 (decodable) + ablation moves behavior (causal);
vestige -> R^2 ~ chance + ablation inert. Coordination direction expected decodable
but adaptation-sensitive.
"""
import jax, jax.numpy as jnp, numpy as np
from jax import lax
import berryworld_jax as bwj
import train_jax as T


def _cfg_pm(env, N, marked):
    nbt = env.get("n_berry_types", 2); grid = env.get("grid", 15)
    cfg = bwj.JCfg(
        n_agents=N, episode_len=env["episode_len"], poison_delay=env["poison_delay"],
        zap_removal_steps=env["zap_removal_steps"], r_zap_bonus=env["r_zap_bonus"],
        bonus_requires_mark=env.get("bonus_requires_mark", False), c_zapped=env.get("c_zapped", 2.0),
        grid=grid, n_berry_types=nbt, ghost_keeps_bonus=env.get("ghost_keeps_bonus", True),
        convergent_berry=env.get("convergent_berry", None), coord_k=env.get("coord_k", 0.0),
        coord_a=env.get("coord_a", 1.5), conformity_berry=env.get("conformity_berry", None),
        marked_mask=tuple(t in marked for t in range(nbt)))
    pm = T.build_patch_mask(marked, N, n_berry_types=nbt, grid=grid)
    return cfg, pm


def train_ghost_policy(env, hp, n_install, N, marked, seed):
    """Full flagship run (learning-on) -> ghost-end params (the persisting policy)."""
    cfg, pm = _cfg_pm(env, N, marked)
    train = T.make_train(cfg, pm, hp, n_install=n_install)
    params, _ = jax.jit(train)(jax.random.PRNGKey(seed))
    return cfg, pm, params


def rollout(cfg, pm, params, hidden, key, E, mi, ci, ablate_w=None, ablate_scale=1.0):
    """Roll the policy through one GHOST episode (enforce off). Returns per-(agent,env)
    mean carry (E*N, H) and per-(agent,env) opp on the marked (mi) and control (ci) berry."""
    net = T.ACGRU(hidden, bwj.N_ACTIONS)
    fwd = jax.vmap(lambda p, c, o: net.apply(p, c, o), in_axes=(0, 0, 1))      # -> (N,E,*)
    vstep = jax.vmap(lambda s, a: bwj.step(cfg, s, a, enforce=False), in_axes=(0, 0))
    vreset = jax.vmap(lambda k: T.reset_env(cfg, pm, k))
    N = cfg.n_agents
    w = None if ablate_w is None else ablate_w / (jnp.linalg.norm(ablate_w) + 1e-8)
    key, kr = jax.random.split(key)
    state, obs = vreset(jax.random.split(kr, E))            # state batched (E,..), obs (E,N,D)
    carry0 = jnp.zeros((N, E, hidden))

    def step_t(cso, kt):
        carry, state, obs = cso
        new_carry, logits, _ = fwd(params, carry, obs)     # (N,E,H),(N,E,A)
        if w is not None:                                   # ablate: project direction out
            proj = jnp.einsum('neh,h->ne', new_carry, w)[..., None] * w
            new_carry = new_carry - ablate_scale * proj
        kt, ksa = jax.random.split(kt)
        acts = jax.random.categorical(ksa, logits)          # (N,E)
        nstate, nobs, _, _, info = vstep(state, acts.T)     # info fields batched (E,N,T)
        return (new_carry, nstate, nobs), (new_carry, info['eats_pa'], info['enc_pa'])

    (_, _, _), (carry_t, eats_t, enc_t) = lax.scan(
        step_t, (carry0, state, obs), jax.random.split(key, cfg.episode_len))
    # carry_t (Tstep,N,E,H); eats_t/enc_t (Tstep,E,N,Tt)
    carry_mean = carry_t.mean(0).transpose(1, 0, 2).reshape(E * N, hidden)      # (E*N, H)
    def opp(idx):
        e = eats_t[..., idx].sum(0); c = enc_t[..., idx].sum(0)                 # (E,N)
        return (e / jnp.maximum(c, 1e-9)).reshape(E * N)
    return np.asarray(carry_mean), np.asarray(opp(mi)), np.asarray(opp(ci))


def fit_probe(X, y):
    """Ridge-ish least-squares X->y. Returns weight (H,) and R^2 (in-sample)."""
    Xc = X - X.mean(0); yc = y - y.mean()
    w, *_ = np.linalg.lstsq(Xc, yc, rcond=1e-3)
    pred = Xc @ w
    ss_res = float(((yc - pred) ** 2).sum()); ss_tot = float((yc ** 2).sum()) + 1e-12
    return w, 1.0 - ss_res / ss_tot


def run(env, hp, n_install, N, marked, mi, ci, mode, seed, E=64):
    """Decode + ablate for one norm/seed. mode: 'avoid' or 'converge' (orients the gap)."""
    hidden = hp["hidden"]
    cfg, pm, params = train_ghost_policy(env, hp, n_install, N, marked, seed)
    k = jax.random.PRNGKey(1000 + seed)
    k, k1, k2, k3 = jax.random.split(k, 4)
    # normal rollout -> decode direction
    Xc, opp_m, opp_c = rollout(cfg, pm, params, hidden, k1, E, mi, ci)
    w, r2 = fit_probe(Xc, opp_m)                            # decode norm behavior
    gap_norm = float((opp_c - opp_m).mean() if mode == 'avoid' else (opp_m - opp_c).mean())
    # ablate the decoded direction
    _, opp_m_ab, opp_c_ab = rollout(cfg, pm, params, hidden, k2, E, mi, ci, ablate_w=jnp.asarray(w))
    gap_ab = float((opp_c_ab - opp_m_ab).mean() if mode == 'avoid' else (opp_m_ab - opp_c_ab).mean())
    # control: ablate a RANDOM direction of equal norm
    wr = np.asarray(jax.random.normal(k3, (hidden,))); wr = wr / (np.linalg.norm(wr) + 1e-8) * np.linalg.norm(w)
    _, opp_m_rc, opp_c_rc = rollout(cfg, pm, params, hidden, k2, E, mi, ci, ablate_w=jnp.asarray(wr))
    gap_rc = float((opp_c_rc - opp_m_rc).mean() if mode == 'avoid' else (opp_m_rc - opp_c_rc).mean())
    return dict(r2=r2, gap_norm=gap_norm, gap_ablate=gap_ab, gap_randctrl=gap_rc,
                move_decoded=gap_norm - gap_ab, move_random=gap_norm - gap_rc)


def report(name, results):
    r2 = np.mean([r['r2'] for r in results]); gn = np.mean([r['gap_norm'] for r in results])
    ga = np.mean([r['gap_ablate'] for r in results]); gr = np.mean([r['gap_randctrl'] for r in results])
    md = np.mean([r['move_decoded'] for r in results]); mr = np.mean([r['move_random'] for r in results])
    dec = 'DECODABLE' if r2 > 0.15 else 'chance (~not decodable)'
    caus = 'CAUSAL (decoded >> random)' if md > 0.02 and md > 2 * abs(mr) else 'not causal'
    print(f"[{name}] decodability R2={r2:.3f} ({dec})")
    print(f"        norm gap {gn:+.3f} -> ablate-decoded {ga:+.3f} (moved {md:+.3f}) | ablate-random {gr:+.3f} (moved {mr:+.3f})")
    print(f"        verdict: {dec} + {caus}")
    return dict(r2=r2, gap_norm=gn, gap_ablate=ga, move_decoded=md, move_random=mr)
