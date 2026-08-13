"""
train_jax.py -- recurrent IPPO in JAX for berryworld_jax, fully jitted so it
runs on device and vmaps over seeds. Per-agent INDEPENDENT parameters (stacked
leading dim = pool size), so removing an agent is dropping a slice.

Validation discipline: at N=1, marked=(), this must reproduce Gate A (a lone
agent learns to avoid the poison berry). If it can't reproduce a result we
already have on CPU/PyTorch, the port is wrong -- don't spend GPU on it.

    python train_jax.py           # N=1 Gate A smoke on CPU
"""
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import flax.linen as nn
import optax

import berryworld_jax as bwj
from berryworld import BerryWorld, Config


# ------------------------------------------------------------------- network
class ACGRU(nn.Module):
    hidden: int
    n_actions: int

    @nn.compact
    def __call__(self, carry, obs):
        x = nn.tanh(nn.Dense(self.hidden)(obs))
        carry, h = nn.GRUCell(features=self.hidden)(carry, x)
        logits = nn.Dense(self.n_actions)(h)
        val = nn.Dense(1)(h)[..., 0]
        return carry, logits, val


# ------------------------------------------------------------- jittable reset
def _place(cfg, patch_mask, key):
    """Sample N distinct free interior cells (jittable, gumbel-masked)."""
    G, N = cfg.grid, cfg.n_agents
    interior = jnp.zeros((G, G), bool).at[1:G - 1, 1:G - 1].set(True)
    free = (interior & ~jnp.asarray(patch_mask).any(0)).reshape(-1)

    def pick(carry, _):
        taken, k = carry
        k, ks = jax.random.split(k)
        g = jax.random.gumbel(ks, (G * G,)) + jnp.where(free & ~taken, 0., -1e9)
        c = jnp.argmax(g)
        taken = taken.at[c].set(True)
        return (taken, k), c
    (_, _), cells = lax.scan(pick, (jnp.zeros(G * G, bool), key), None, length=N)
    pos = jnp.stack([cells // G, cells % G], axis=1).astype(jnp.int32)
    return pos


def reset_env(cfg, patch_mask, key):
    kp, kf, ks = jax.random.split(key, 3)
    pos = _place(cfg, patch_mask, kp)
    facing = jax.random.randint(kf, (cfg.n_agents,), 0, 4)
    s, obs = bwj.reset(cfg, jnp.asarray(patch_mask), pos, facing, ks)
    return s, obs


# --------------------------------------------------------------------- train
def make_train(cfg, patch_mask, hp, n_install=None, freeze_after=None,
               isolate_after=None, n_focal=1, unmark_after=None, gate_bonus_after=None,
               gate_removal_after=None, hazard_off_after=None, flatten_returns_after=None,
               mask_self_after=None):
    # mask_self_after: perception-probe M1 (self-only mask), INDEPENDENT of unmark_after.
    # None (default) = self-mark visible every update (bit-exact). An int M = for updates
    # >= M, zero ONLY the agent's self-mark feature (its perception of its own violator
    # status), leaving the world mark planes intact -> enforcer targeting untouched, zap
    # surge preserved by construction. Pair M == n_install to drop the self cue at the
    # ghost switch. unmark_after masks BOTH channels (self+world = M1 U M2); mask_self_after
    # is the self-only decomposition. (The two compose: unmark_after already zeros self.)
    # hazard_off_after: environmental own-knockout. None (default) = physical poison
    # penalty always live (bit-exact). An int H = gate the -r_poison penalty for updates
    # >= H (paired vs the poison-ghost run: ghost-alone persists, ghost+hazard-off decays).
    # flatten_returns_after: coordination-grounding knockout. None (default) = convergent
    # berry keeps its increasing-returns bonus (bit-exact). An int F = flatten it to the
    # base return for updates >= F (does the coordination norm decay once its grounding
    # is removed). Both require the matching env feature (r_poison>0 / convergent_berry).
    # gate_removal_after: independent gate on the 25-step timeout removal. None
    # (default) = removal follows the violator-cost enforce gate (bit-exact). An int R
    # = withhold removal for updates >= R while leaving the zap penalty under n_install
    # -> isolates whether "violator cost" includes the timeout.
    # gate_bonus_after: enforcer-incentive gate, INDEPENDENT of n_install. None
    # (default) = bonus paid every update (bit-exact). An int B = for updates >= B, the
    # enforcer's mark-contingent bonus is withheld while the violator-cost enforce gate
    # is untouched -> the "enforcer-incentive-only" removal cell. Set n_install=updates
    # (enforce never gated) + gate_bonus_after=switch for that cell.
    # unmark_after: no-cue arm. None (default) = marks visible every update
    # (bit-exact). An int U = for updates >= U, zero the mark planes + self-mark
    # feature in the observation (sever the reconstruction channel). Pair U ==
    # n_install to drop the cue at the switch. Compose with freeze_after +
    # isolate_after == n_install for the cleanest stored-not-reconstructed test.
    # isolate_after: coordination knockout. None (default) = all agents active every
    # update (bit-exact). An int I = for updates >= I, deactivate all but the first
    # n_focal agents (removes the social scaffold: no other agents to enforce or
    # conform to). Pair I == n_install to isolate at the enforcement switch -> the
    # ghost(coord-on) vs isolate(coord-off) contrast that tests whether a persisting
    # norm is internalized or coordination-propped.
    # freeze_after: frozen-weights control. None (default) = weights train every
    # update (bit-exact). An int F = for updates >= F, still roll out & log behavior
    # but SKIP the optimizer update so params + Adam state stay fixed (the "stored
    # not reconstructed" baseline that subtracts continued-training drift). Pair
    # F == n_install to freeze exactly at the enforcement switch.
    # n_install: extinction schedule. None (default) = enforce ON every update
    # (bit-exact with the pre-flag trainer). An int K = enforce ON for updates < K
    # (install the norm), then enforce=False (ghost) for the rest (extinction),
    # logging the phase per update so the decay curve is recoverable.
    N, E = cfg.n_agents, hp["num_envs"]
    net = ACGRU(hp["hidden"], bwj.N_ACTIONS)
    # planes = wall + T berries + agent + T marks = 2 + 2T (was 6 hard-coded for T=2)
    obs_dim = (2 + 2 * cfg.n_berry_types) * (2 * cfg.view + 1) ** 2 + 2 + cfg.n_berry_types
    tx = optax.chain(optax.clip_by_global_norm(hp["max_grad"]),
                     optax.adam(hp["lr"]))
    n_install = hp["updates"] if n_install is None else n_install

    vstep = jax.vmap(lambda s, a, e, am, mm, eb, er, ho, fr, ms: bwj.step(cfg, s, a, e, am, mm, eb, er, ho, fr, ms),
                     in_axes=(0, 0, None, None, None, None, None, None, None, None))  # +hazard_off,flatten,mask_self broadcast
    vreset = jax.vmap(lambda k: reset_env(cfg, patch_mask, k))

    def agent_apply(params, carry, obs):                    # obs (E,D) carry (E,H)
        return net.apply(params, carry, obs)
    # over agents: params axis 0, carry axis 0, obs axis 1(agent) -> outputs (N,E,*)
    fwd = jax.vmap(agent_apply, in_axes=(0, 0, 1))

    def train(rng):
        rng, ki = jax.random.split(rng)
        c0 = jnp.zeros((E, hp["hidden"]))
        o0 = jnp.zeros((E, obs_dim))
        params = jax.vmap(lambda k: net.init(k, c0, o0))(
            jax.random.split(ki, N))
        opt_state = tx.init(params)

        def update(runner, sched_u):
            params, opt_state, rng = runner
            (enforce_u, freeze_u, active_mask_u, mask_marks_u, enf_bonus_u, enf_removal_u,
             hazard_off_u, flat_u, mask_self_u) = sched_u
            rng, kr = jax.random.split(rng)
            state, obs = vreset(jax.random.split(kr, E))     # obs (E,N,D)
            carry = jnp.zeros((N, E, hp["hidden"]))

            # --- rollout one episode across E envs via scan over time
            def step_t(carry_all, key_t):
                carry, state, obs = carry_all
                new_carry, logits, val = fwd(params, carry, obs)   # (N,E,*)
                key_t, ksa = jax.random.split(key_t)
                acts = jax.random.categorical(ksa, logits)         # (N,E)
                logp = jnp.take_along_axis(
                    jax.nn.log_softmax(logits), acts[..., None], -1)[..., 0]
                nstate, nobs, rew, done, info = vstep(state, acts.T, enforce_u, active_mask_u,
                                                      mask_marks_u, enf_bonus_u, enf_removal_u,
                                                      hazard_off_u, flat_u, mask_self_u)  # (E,N)
                # store agent-major (N,E,*); obs came from env as (E,N,D)
                trans = (obs.transpose(1, 0, 2), acts, logp, val, rew.T,
                         info["eats"], info["zaps_landed"],
                         info["zaps_on_marked"], info["marked_agents"],
                         info["active_agents"], info["berry_encounters"])
                return (new_carry, nstate, nobs), trans

            rng, kt = jax.random.split(rng)
            (_, state, _), traj = lax.scan(
                step_t, (carry, state, obs),
                jax.random.split(kt, cfg.episode_len))
            (obs_t, act_t, logp_t, val_t, rew_t, eats_t,
             zl_t, zm_t, ma_t, aa_t, enc_t) = traj                 # (T,N,E,*)

            # --- GAE per (agent, env)
            def gae_scan(carry, x):
                gae, next_v = carry
                rew, val = x
                delta = rew + hp["gamma"] * next_v - val
                gae = delta + hp["gamma"] * hp["lam"] * gae
                return (gae, val), gae
            _, adv = lax.scan(gae_scan, (jnp.zeros((N, E)), jnp.zeros((N, E))),
                              (rew_t, val_t), reverse=True)
            ret = adv + val_t
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # --- PPO update (independent per agent; single optimizer over stack)
            def loss_fn(params):
                def replay_agent(p, obs_a, act_a):            # (T,E,D),(T,E)
                    def rstep(carry, o):
                        carry, logits, val = net.apply(p, carry, o)
                        return carry, (logits, val)
                    _, (logits, val) = lax.scan(
                        rstep, jnp.zeros((E, hp["hidden"])), obs_a)
                    return logits, val
                logits, val = jax.vmap(replay_agent, in_axes=(0, 1, 1))(
                    params, obs_t, act_t)                     # (N,T,E,*)
                logits = logits.transpose(1, 0, 2, 3)         # (T,N,E,A)
                val = val.transpose(1, 0, 2)                  # (T,N,E)
                logp = jnp.take_along_axis(
                    jax.nn.log_softmax(logits), act_t[..., None], -1)[..., 0]
                ratio = jnp.exp(logp - logp_t)
                p1 = ratio * adv
                p2 = jnp.clip(ratio, 1 - hp["clip"], 1 + hp["clip"]) * adv
                pi_loss = -jnp.minimum(p1, p2).mean()
                v_loss = ((val - ret) ** 2).mean()
                probs = jax.nn.softmax(logits)
                ent = -(probs * jax.nn.log_softmax(logits)).sum(-1).mean()
                return pi_loss + hp["vf"] * v_loss - hp["ent"] * ent

            def ppo_epoch(carry, _):
                params, opt_state = carry
                g = jax.grad(loss_fn)(params)
                upd, opt_state = tx.update(g, opt_state, params)
                params = optax.apply_updates(params, upd)
                return (params, opt_state), None
            params_in, opt_in = params, opt_state
            (params, opt_state), _ = lax.scan(
                ppo_epoch, (params, opt_state), None, length=hp["epochs"])
            # frozen-weights control: in phase 2 (freeze_u=True) discard the update so
            # weights + Adam state stay fixed at the switch (behavior is still rolled
            # out & logged above -> the "stored not reconstructed" baseline). Default
            # schedule is all-False, so jnp.where picks the updated values => bit-exact.
            params = jax.tree_util.tree_map(
                lambda o, n: jnp.where(freeze_u, o, n), params_in, params)
            opt_state = jax.tree_util.tree_map(
                lambda o, n: jnp.where(freeze_u, o, n), opt_in, opt_state)

            # eats_t is (T, E, n_berry_types); per-env episode totals, env-mean
            zl, zm = zl_t.sum(), zm_t.sum()
            prev = ma_t.sum() / jnp.maximum(aa_t.sum(), 1)      # marked prevalence
            share = zm / jnp.maximum(zl, 1)                     # zaps hitting marked
            metrics = dict(ret=rew_t.sum(0).mean(),
                           selectivity=share / jnp.maximum(prev, 1e-8),
                           prevalence=prev,                     # marked frac -> ceiling=1/prev
                           zaps=zl / E,                         # landed zaps/episode
                           enforce=jnp.asarray(enforce_u, jnp.float32),  # 1=install,0=extinct
                           freeze=jnp.asarray(freeze_u, jnp.float32),    # 1=weights frozen
                           bonus_on=jnp.asarray(enf_bonus_u, jnp.float32),  # 1=enforcer bonus paid
                           removal_on=jnp.asarray(enf_removal_u, jnp.float32),  # 1=timeout removal live
                           hazard_on=1.0 - jnp.asarray(hazard_off_u, jnp.float32),  # 1=physical penalty live
                           coord_on=1.0 - jnp.asarray(flat_u, jnp.float32),  # 1=coordination returns live
                           self_cue_on=1.0 - jnp.asarray(mask_self_u, jnp.float32))  # 1=self-mark visible (M1 off)
            for _t in range(cfg.n_berry_types):        # per-type eats & encounters (eat0/eat1/...)
                metrics[f"eat{_t}"] = eats_t[..., _t].sum() / E
                metrics[f"enc{_t}"] = enc_t[..., _t].sum() / E
            return (params, opt_state, rng), metrics

        enforce_sched = jnp.arange(hp["updates"]) < n_install    # True=enforce, then ghost
        freeze_sched = ((jnp.arange(hp["updates"]) >= freeze_after)
                        if freeze_after is not None
                        else jnp.zeros(hp["updates"], bool))       # True=weights frozen
        # coordination knockout: phase-2 active-agent mask (N,) per update. Default
        # all-True (active & True == active -> bit-exact). isolate_after set ->
        # keep only the first n_focal agents once updates >= isolate_after.
        focal = jnp.arange(N) < n_focal
        active_mask_sched = (
            jnp.where((jnp.arange(hp["updates"]) >= isolate_after)[:, None],
                      focal[None, :], True)
            if isolate_after is not None
            else jnp.ones((hp["updates"], N), bool))
        # no-cue schedule: True from unmark_after on -> marks masked in obs.
        mask_marks_sched = ((jnp.arange(hp["updates"]) >= unmark_after)
                            if unmark_after is not None
                            else jnp.zeros(hp["updates"], bool))
        # perception-probe M1: self-only mask, True from mask_self_after on (default all-False -> bit-exact)
        mask_self_sched = ((jnp.arange(hp["updates"]) >= mask_self_after)
                           if mask_self_after is not None
                           else jnp.zeros(hp["updates"], bool))
        # enforcer-incentive gate: bonus ON (True) until gate_bonus_after, then withheld.
        enf_bonus_sched = ((jnp.arange(hp["updates"]) < gate_bonus_after)
                           if gate_bonus_after is not None
                           else jnp.ones(hp["updates"], bool))
        # removal gate: independent when gate_removal_after set, else follows enforce (bit-exact)
        enf_removal_sched = ((jnp.arange(hp["updates"]) < gate_removal_after)
                             if gate_removal_after is not None
                             else enforce_sched)
        # hazard-off / flatten schedules: True from the _after update on -> gate applied.
        # Default None -> all-False -> bit-exact (hazard live, coordination returns live).
        hazard_off_sched = ((jnp.arange(hp["updates"]) >= hazard_off_after)
                            if hazard_off_after is not None
                            else jnp.zeros(hp["updates"], bool))
        flat_sched = ((jnp.arange(hp["updates"]) >= flatten_returns_after)
                      if flatten_returns_after is not None
                      else jnp.zeros(hp["updates"], bool))
        (params, _, _), metrics = lax.scan(
            update, (params, opt_state, rng),
            (enforce_sched, freeze_sched, active_mask_sched, mask_marks_sched,
             enf_bonus_sched, enf_removal_sched, hazard_off_sched, flat_sched,
             mask_self_sched))
        return params, metrics

    return train


DEFAULT_HP = dict(hidden=64, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2,
                  epochs=3, ent=0.01, vf=0.5, max_grad=0.5,
                  num_envs=16, updates=400)


def build_patch_mask(marked, n_agents, seed=0, n_berry_types=2, grid=15):
    c = Config(); c.marked_berries = marked; c.n_agents = n_agents
    c.n_berry_types = n_berry_types; c.grid = grid
    return BerryWorld(c, seed=seed).patch_mask.copy()


if __name__ == "__main__":
    import time
    marked = ()
    cfg = bwj.JCfg(n_agents=1, episode_len=300, poison_delay=25,
                   zap_removal_steps=25,
                   marked_mask=tuple(t in marked for t in range(2)))
    pm = build_patch_mask(marked, 1)
    train = make_train(cfg, pm, DEFAULT_HP)
    t0 = time.time()
    params, metrics = jax.block_until_ready(jax.jit(train)(jax.random.PRNGKey(0)))
    dt = time.time() - t0
    e0 = np.array(metrics["eat0"])
    print(f"Gate A (N=1, no marks) -- {dt:.1f}s for {DEFAULT_HP['updates']} updates")
    print(f"  poison eaten/episode: first10 {e0[:10].mean():.1f} -> last10 {e0[-10:].mean():.1f}")
    print("  PASS (avoidance learned)" if e0[-10:].mean() < 0.6 * e0[:10].mean()
          else "  (no clear avoidance -- inspect)")
