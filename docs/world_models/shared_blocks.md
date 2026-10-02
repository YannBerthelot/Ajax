# Shared building blocks for DreamerV3 and TD-MPC2 in Ajax

This spec lists the modular blocks that DreamerV3 (D3) and TD-MPC2 (T2) need. Each block is designed once, as a pure-functional JAX function or a flax.linen module, with a parameterisation that reproduces **both** references exactly. For each block it gives:

- the math;
- the single signature that covers both papers;
- which paper uses which variant;
- unit tests pinned against the reference formulas.

It also flags **false friends**: blocks that look shared but differ in a way that matters.

## 0. Sources and conventions

- **D3 spec:** `specs/dreamerv3_spec.md`, cited as D3§x.y. It pins `danijar/dreamerv3` @ e3f0224 (paper-era 2411f7d) and arXiv:2301.04104v2.
- **T2 spec:** `specs/tdmpc2_spec.md`, cited as T2§x.y. It pins `nicklashansen/tdmpc2` @ e9f5932 (HEAD) and 5f6fade (PE), and arXiv:2310.16828v2.
- **Re-checked against the reference code (both official repositories at the commits pinned in the two specs):**
  - D3: `embodied/jax/{outs,heads,nets,opt,utils}.py` and `dreamerv3/agent.py:480-488` (lambda_return).
  - T2: `tdmpc2/common/{math,scale,init,layers}.py`, `tdmpc2/tdmpc2.py:208-230` (update_pi) and `common/world_model.py:186-216` (Q).
- **Pinned numbers.** Every number in the test lists below was computed by the reference translations shipped with the block tests (`tests/world_models/reference_impls.py`), which contains literal translations of the reference functions: D3 is already JAX, and T2's torch code is translated to numpy line for line. The environment was Ajax poetry: jax 0.7.2, flax 0.10.7, optax 0.2.6. Unless stated otherwise, the tolerance is rtol 1e-6 in f32.
- **Ajax conventions this spec assumes** (from CLAUDE.md and the codebase):
  - flax.linen modules plus pure functions.
  - Hyperparameters live in frozen dataclasses.
  - No per-feature kwargs on agents: optional behaviour is an Extension.
  - Tests ship with every block.
  - Existing pieces this spec reuses or warns against: `LoadedTrainState.soft_update` (optax.incremental_update), `networks.MultiCritic` (nn.vmap ensemble), `networks.memory._mask_reset`, PQN `compute_q_lambda_targets`, SAC `SquashedNormal`, `networks.utils.get_adam_tx`, `buffers.utils.get_buffer` (flashbax).

## 1. Block index

| # | Block | D3 | T2 | Shared form | False friend? |
|---|---|---|---|---|---|
| B1 | symlog / symexp | obs encoder, decoder targets, bin construction | inside two-hot only | identical | **yes**: what gets symlogged |
| B2 | Two-hot discrete regression (encode, decode, soft-CE, head) | reward, critic | reward, Q | one function and one spec (bins, transform) | **yes**: interpolation space and decode order |
| B3 | Percentile scale normaliser | retnorm on advantages | RunningScale on Q | one function and one config | **yes**: clamp before or after the EMA, init 0 or 1 |
| B4 | NormedMLP (Linear, then Dropout, then Norm, then Act) | RMSNorm + SiLU everywhere | LayerNorm + Mish, dropout in Q | one module | **yes**: norm eps, fast variance, output dtype |
| B5 | Initialisers and output scales | trunc-normal fan-in × outscale | N(0, 0.02), zeroed heads | one `scaled(init, outscale)` wrapper | **yes**: jax `truncated_normal(0.02)` is wrong for T2 |
| B6 | BlockLinear | GRU core | — | D3 only | fan-in = full width |
| B7 | Grouped softmax: SimNorm vs categorical latent | latent sample | latent itself | primitive shared, semantics not | **yes** |
| B8 | Unimix categorical, straight-through, KL, free bits | RSSM latents (actor in paper era) | — | D3 only | — |
| B9 | Q ensemble with random-pair reduction | — (single V critic) | Nq = 5 Qs | T2 only (generic wrapper) | Ajax MultiCritic member is unusable |
| B10 | EMA target / slow network | slow critic, rate 0.02 | target Q, rate 0.01 | identical update | **yes**: role (regulariser vs bootstrap) |
| B11 | λ-returns | imagination and replay | one-step TD (λ=0), planner (λ=1) | one function | **yes**: Ajax PQN version casts to bool |
| B12 | Gaussian policy heads and bounded std | bounded_normal, not squashed | tanh-squashed, log-std bounds | primitives shared, heads separate | **yes**: squash, log-prob, gradient estimator |
| B13 | MPPI planner | — | MPPI / CEM | T2 only (generic) | — |
| B14 | Latent rollout scan | observe / imagine | consistency, π-trajectories, planning | one scan helper | — |
| B15 | RSSM block-GRU core and observe/imagine steps | yes | — | D3 only | **yes**: Ajax MemoryCell 'gru' is a different GRU |
| B16 | Bernoulli (continue / termination) head | continue, soft label | termination (HEAD episodic, Extension) | one loss | soft vs hard label |
| B17 | Optimisers | LaProp + per-tensor AGC + warmup, one optimiser | Adam + global clip + per-group lr, two optimisers | two optax chain builders, shared primitives | **yes**: optax AGC is unit-wise; Ajax Adam eps |
| B18 | Sequence replay | 65-step windows across episodes, write-back, online queue | (H+1)-row slices within episodes | one per-env ring buffer with a window predicate | **yes**: action alignment and Ajax's auto-reset transitions |
| B19 | Small utilities: discount from horizon, mixed precision, categorical pick | γ = 1 − 1/333; bf16 | γ(T) heuristic; f32 | shared | MSE losses are not shared |

---

## B1. symlog / symexp

**Math.**
- symlog(x) = sign(x)·log1p(abs(x))
- symexp(x) = sign(x)·expm1(abs(x))

The two are mutual inverses, odd functions, with symlog'(0) = 1.

**Signature.** `symlog(x) -> x`, `symexp(x) -> x`. Elementwise, dtype-preserving, pure.

**Variants.** D3 uses log1p/expm1 (nets.py:59-64; D3§1.6). T2 uses log(1 + abs(x)) and exp(abs(x)) − 1 (math.py:42-55; T2§1.14). These agree to rounding, so use log1p/expm1.

**False friend: what each paper symlogs.**
- D3 symlogs **continuous vector observations** (encoder, in f32 before the bf16 cast) and **decoder targets** (symlog-MSE). It also builds the two-hot bins with symexp. D3 does **not** symlog two-hot targets.
- T2 symlogs **only inside two-hot** (reward and Q targets). It never symlogs observations or latents (T2§1.4, T2§1.14).

**Tests.**
1. symlog([−1000, −1, −0.5, 0, 0.5, 1, 1000]) = [−6.908755, −0.6931472, −0.4054651, 0, 0.4054651, 0.6931472, 6.908755].
2. symexp(symlog(x)) == x at rtol 1e-6 on that vector.
3. Oddness: symlog(−x) == −symlog(x) exactly.
4. grad(symlog)(0.0) == 1.0, and grad(symexp)(0.0) == 1.0.
5. bf16 in gives bf16 out.

---

## B2. Two-hot discrete regression (encode, decode, soft-CE, head)

### Math (generic, any sorted bins b_0 < … < b_{K−1})

**Encode** a target y (after `transform`):
- below = clip(#{b ≤ y} − 1, 0, K−1)
- above = clip(K − #{b > y}, 0, K−1)
- If below == above, put weight 1 on that bin. This covers out-of-range targets, which go to the edge bin.
- Otherwise: w_below = abs(b_above − y)/(b_above − b_below) and w_above = 1 − w_below.

This is Dreamer's `TwoHot.loss` algorithm (outs.py:311-330). For **uniformly spaced** bins it equals T2's floor formula exactly: k = floor((y − vmin)/δ), o = frac, t[k] = 1 − o, t[k+1] = o (math.py:58-71). An exact bin hit puts weight 1 on that bin in both.

**Soft-CE:** −Σ_k encode(sg(y))_k · log_softmax(logits)_k. D3 writes this as logits − logsumexp; T2 as `F.log_softmax`. They are the same.

**Decode:**
- p = softmax(logits).
- m = (K−1)/2 (K odd).
- E = p_m·b_m + Σ_{i<m} (p_i·b_i reversed + p_{m+1+i}·b_{m+1+i}). This is the **symmetric sum**, which gives exactly 0 for uniform p.
- Return inverse_transform(E).

### Signature

```python
@dataclass(frozen=True)
class TwoHotSpec:
    bins: tuple[float, ...]           # sorted, odd K, antisymmetric, bins[K//2] == 0, in INTERPOLATION space
    transform: Literal["identity", "symlog"]   # applied to targets before encoding; inverse applied after expectation

def twohot_encode(y, spec) -> f32[..., K]
def twohot_decode(logits, spec) -> f32[...]
def twohot_ce(logits, y, spec) -> f32[...]          # target stop-gradiented inside
class TwoHotHead(nn.Module): spec; out_kernel_init (zeros by default)   # Linear(K), logits in f32

def dreamer_twohot_spec(K=255, limit=20.0):          # half = symexp(linspace(-limit, 0, (K+1)//2)); bins = concat(half, -half[:-1][::-1])
    return TwoHotSpec(bins, "identity")
def tdmpc2_twohot_spec(K=101, vmin=-10.0, vmax=10.0): # mirrored linspace(vmin, 0, (K+1)//2), transform symlog
    return TwoHotSpec(bins, "symlog")
```

Build both bin sets **by mirroring a half**, so that bins[K//2] == 0 and antisymmetry hold exactly.

### Variants

| | D3 (D3§1.9, Algorithm D; heads.py:132-144; outs.py:273-330) | T2 (T2§1.15-1.17; math.py:5-9, 58-83) |
|---|---|---|
| Bins | 255, symexp(linspace(−20, 20)) in **raw** space; range ±4.85e8 | 101, linspace(−10, 10) in **symlog** space; step 0.2; raw range ±22025 |
| Target transform | none (raw reward or return) | symlog, then clamp to [−10, 10] |
| Interpolation | linear in **raw** space | linear in **symlog** space |
| Decode | E_p[raw bins], symmetric sum | symexp(E_p[symlog bins]), naive sum |
| Head init | zero kernel and bias (reward and value) | zero final weight (reward and every Q member), bias 0 |
| Users | reward head, critic (imagination and replay losses) | reward head, Q ensemble, TD targets, planner |

**Deviation:** use the symmetric sum for T2 as well. In symlog space the difference from T2's naive sum is ≤ 1e-6. Mirrored bins differ from `torch.linspace` by at most 1 ulp.

### False friends

1. **Interpolation space.** For y = 1.0, D3's bin pair (131, 132) = (0.8775, 1.1977) gets upper weight **0.3827** with raw-space interpolation. Symlog-space interpolation on the same pair would give **0.4015**. The targets differ even on identical bins.
2. **Decode order.** symexp(E[b]) ≠ E[symexp(b)]. Do not "simplify" one into the other.
3. **Range.** T2 saturates at abs(y) ≈ 22025. That is harmless for DMC (abs(Q) ≤ 200 at γ = 0.99), but not for large-return tasks.
4. **K must be ≥ 2.** T2's num_bins ∈ {0, 1} modes are broken (T2§1.18), so do not port them.

### Tests (pinned)

1. **D3 bins:**
   - len 255; b[127] == 0.0 exactly; b[128] = 0.17055759; b[0] = −b[254] = −4.8516518e8.
   - b == −b[::-1] exactly.
   - abs(b − symexp(linspace(−20, 20, 255))) / max(1, abs(b)) ≤ 2e-6.
2. **D3 encode:**

   | y | indices | weights |
   |---|---|---|
   | 1.0 | (131, 132) | (0.61732966, 0.38267030) |
   | −3.7 | (117, 118) | (0.81556690, 0.18443313) |
   | 0.0 | 127 | 1 |
   | 1e10 | 254 | 1 |
   | −1e10 | 0 | 1 |
   | b[130] = 0.60390407 | 130 | 1 |

3. **T2 encode:**

   | y | indices | weights |
   |---|---|---|
   | 1.0 | (53, 54) | (0.5342636, 0.4657364) |
   | −3.7 | (42, 43) | (0.73781204, 0.26218796) |
   | 0 | 50 | 1 |
   | ±1e10 | 100 / 0 | 1 |
   | 22025 | (99, 100) | (1.068e-4, 0.99989) |

   The generic encoder must match T2's floor formula to atol 1e-6 on 10⁴ random y in [−3e4, 3e4].
4. **Soft-CE at uniform logits** = log K for any y: 5.5412636 (K = 255) and 4.6151205 (K = 101).
5. **Soft-CE with logits = linspace(−1, 1, K), y = 1.0:** D3 5.669425; T2 4.710373.
6. **Decode of uniform logits:**
   - D3 gives 0.0 **exactly**. Regression guard: the naive left-to-right sum gives **2.0** in f32 on D3 bins.
   - T2 gives abs ≤ 1e-6.
7. **Decode of linspace(−1, 1, K):** D3 2.4558884e7 (rtol 1e-5); T2 23.267637.
8. **Round trip:** decode(log(encode(y))) == y for y ∈ {1, −3.7, 123.4} in both specs (rtol 1e-6).
9. **Gradients:** grad of CE w.r.t. logits = softmax − target (sums to 0). No gradient reaches y.
10. **Head at init:** zero output layer gives decode == 0 for any input in both specs.

---

## B3. Percentile scale normaliser (D3 retnorm vs T2 RunningScale)

### Math (unified)

r ← (1 − τ)·r + τ·g(P_hi(x) − P_lo(x)), then S = max(floor, r). Here:

- P is `jnp.percentile(method="linear")` over **all** elements of sg(f32(x)).
- The update happens **before** the read, so the current batch already counts.
- There is no debias.

### Signature

```python
@dataclass(frozen=True)
class PercentileScaleConfig:
    rate: float = 0.01; lo: float = 5.0; hi: float = 95.0; floor: float = 1.0
    clamp_before_ema: bool          # g = max(floor, .) if True else identity
    init: float                      # initial r

@struct.dataclass
class PercentileScaleState: r: f32[]   # (+ optional lo/hi EMAs for logging only)

def percentile_scale_update(state, x, cfg) -> state
def percentile_scale(state, cfg) -> f32[]       # max(floor, r)
```

### Variants

| | D3 retnorm (D3§3.8; utils.py:16-91, impl 'perc', debias False, limit 1.0) | T2 RunningScale (T2§2.13; scale.py:5-47) |
|---|---|---|
| Config | `clamp_before_ema=False, init=0.0` | `clamp_before_ema=True, init=1.0` |
| Data x | all imagined λ-returns R, flattened (B·K·H = 15360) | Qp[0] of shape (B, 1): t = 0 policy-loss Q values (decoded, 2-head avg) |
| Applied to | advantage (R − v)/S; offset lo **not** subtracted | Q/S for all H+1 steps in the policy loss |
| When updated | training pass only (not report/eval) | every policy update |
| State | lo and hi EMAs kept separately. Because the EMA is linear and both start at 0, this is exactly the EMA of hi − lo. | scalar `value` |
| Checkpointed | yes (ninjax variable) | **no** (resets to 1 on reload) |

The torch `_positions` formula equals `jnp.percentile` 'linear'. For B = 256, the positions are 12.75 and 242.25.

### False friend

The papers describe the same thing ("EMA of the 5 to 95 percentile range, floored at 1"). The configurations still differ:

- T2 clamps **before** the EMA; D3 clamps **after**.
- D3 starts at 0; T2 starts at 1.
- They normalise different quantities over different data.

**Tests (pinned).** Feed batches x_i = arange(256)·s, with s ∈ {0.001, 0.01, 1, 10}. The 5-95 ranges are 0.2295, 2.295, 229.5 and 2295.

| | after batch 1 | after batch 2 | after batch 3 | after batch 4 |
|---|---|---|---|---|
| D3 S | 1.0 | 1.0 | 2.3199697 | 25.246771 |
| T2 S | 1.0 | 1.01295 | 3.2978203 | 26.214842 |

Also, after batch 4 the D3 lo/hi EMAs are 1.4025983 and 26.649368.

| Further case | D3 S | T2 S |
|---|---|---|
| Constant range 50, after 100 updates | 31.698427 | 32.064415 |
| Ranges alternating 0 and 3, after 2000 updates | 1.50753 | 2.00502 |

The alternating case pins down clamp-before vs clamp-after.

Two more checks:
- No gradient flows through S or the state.
- The single-scalar r form equals D3's two-EMA form to rtol 1e-6.

---

## B4. NormedMLP (Linear, then Dropout, then Norm, then Act, plus an optional output layer)

### Math (per hidden layer i)

h ← act(Norm(Dropout_{p·[i==0]}(h W_i + b_i))). The bias is **always** present. Dropout is the inverted kind and is active only when not deterministic.

**Output layer**, one of:
- none (the trunk output is the result);
- plain Linear;
- Linear, then Norm, then `out_act` (T2's encoder and dynamics end with LayerNorm then SimNorm).

### Signature

```python
class NormedMLP(nn.Module):
    hidden: Sequence[int]
    out: int | tuple | None = None       # None -> no output layer (D3 encoder tokens)
    out_norm_act: Callable | None = None # None -> plain Linear output; e.g. simnorm -> Linear->Norm->SimNorm
    norm: Literal["rms", "layer", "none"] = "rms"
    norm_eps: float = 1e-4
    act: Callable = jax.nn.silu
    dropout: float = 0.0                 # first hidden layer only
    kernel_init: Initializer = dreamer_trunc_normal()
    out_kernel_init: Initializer = dreamer_trunc_normal(outscale=1.0)
    dtype: Any = jnp.float32             # compute dtype (bf16 for D3 on GPU)
    @nn.compact
    def __call__(self, x, deterministic: bool = True): ...
```

### Variants

- **D3** (D3§1.1-1.4; nets.py:230-251, 361-409, 565-587):
  - `norm="rms", norm_eps=1e-4, act=silu, dropout=0`.
  - RMSNorm: f32 statistics, learnable scale only (no shift), output cast back to the compute dtype.
  - Flax mapping: `nn.RMSNorm(epsilon=1e-4, dtype=compute_dtype)`. Verified **0.0** max abs difference from the D3 formula in both f32 and bf16.
  - The input is cast to the compute dtype, and leading dims are flattened, then restored.
  - Heads apply `outscale` to the output layer only.
- **T2** (T2§1.1-1.2; layers.py:94-133):
  - `norm="layer", norm_eps=1e-5, act=jax.nn.mish`.
  - `dropout=0.01` only for Q members, on the first layer only.
  - Flax mapping: `nn.LayerNorm(epsilon=1e-5, use_fast_variance=False)`, with learnable γ and β.
  - The output layer is plain Linear for reward, Q and π, and Linear → LN → SimNorm for the encoder and dynamics.

**Configurations used** (d = 256 at D3 12M; mlp = 512, enc = 256 and L = 512 at T2 5M):

| Network | hidden | out | other |
|---|---|---|---|
| D3 encoder | [d]×3 | None | — |
| D3 reward / continue head | [d]×1 | head | — |
| D3 policy / value | [d]×3 | head | — |
| D3 decoder | [d]×3 | per-key heads | — |
| D3 RSSM prior | [d]×2 | S·C | — |
| D3 RSSM posterior | [d]×1 | S·C | — |
| T2 encoder | [enc]×max(n−1, 1) | L | out_norm_act = simnorm |
| T2 dynamics | [mlp]×2 | L | out_norm_act = simnorm |
| T2 reward / Q | [mlp]×2 | 101 | Q: dropout 0.01 |
| T2 π | [mlp]×2 | 2A | — |

### False friends

1. **Flax defaults are wrong for both papers.**
   - `nn.LayerNorm` defaults to eps 1e-6 with fast variance E[x²] − E[x]². With mean-1000 inputs that gives a max error of **2.2e-2**, against 3.9e-5 with `use_fast_variance=False`.
   - `nn.RMSNorm` defaults to eps 1e-6, and with `dtype=None` a bf16 input gives an **f32** output.
2. **D3 has no shift on RMSNorm.** Do not use LayerNorm with eps 1e-4 for D3. The D3 'layer' implementation exists in the code but is unused.
3. **Dropout sits before the norm in T2.** It is not after the activation.
4. **Ajax's existing `Encoder` and `parse_architecture`** add an output LayerNorm or L2 norm and use an orthogonal init. Neither paper does this, so do not reuse them.

### Tests (pinned)

1. Values of the norms and activations:

   | Expression | Expected |
   |---|---|
   | RMSNorm([1, 2, 3, 4]), eps 1e-4 | [0.36514595, 0.7302919, 1.0954379, 1.4605838] |
   | LayerNorm([1, 2, 3, 4]), eps 1e-5 | [−1.3416355, −0.44721183, 0.44721183, 1.3416355] |
   | silu(1), silu(−1) | 0.7310586, −0.2689414 |
   | mish(1), mish(−1) | 0.8650984, −0.3034015 |

   For comparison, flax's default eps 1e-6 gives LayerNorm([1, 2, 3, 4]) = −1.3416403 in the first entry (test the configured eps). `jax.nn.mish` exists.
2. **Parameter counts:**
   - T2: NormedLinear(i, o) has i·o + 3o parameters.
   - T2 ST walker (S = 24, A = 6): encoder 139,520; dynamics 794,112; reward 582,245; π 533,516; Q (5 members) 2,911,225; total **4,960,618**.
   - T2 MT reference: total **5,389,930** (T2§1.24).
   - D3: an RMS layer has i·o + 2o parameters.
3. **Order test.** With `dropout > 0` and `deterministic=False`, each LN output row still has mean ≈ 0 and variance ≈ 1 (dropout comes before the norm). With `deterministic=True`, the output equals the dropout-0 module.
4. **Layout.** A D3 MLP on input (B, T, F) equals the same MLP on the flattened (B·T, F) input, reshaped back.

---

## B5. Initialisers and output scales

### Math and signature

```python
def dreamer_trunc_normal(outscale=1.0):   # W = outscale * 1.1368 * sqrt(1/fan_in) * TruncNormal(-2, 2) (unit sigma), f32
def normal_init(std=0.02, outscale=1.0)   # T2: W = outscale * N(0, std^2)
# fan_in: rank2 (in,out) -> in;  rank>=3 -> shape[-2]*prod(shape[:-2])  (D3 compute_fans == flax variance_scaling in_axis=-2)
# biases: zeros (both); norm scale: ones (both); LN shift: zeros (T2); T2 task embedding: U(-0.02, 0.02) (Ajax uniform_init)
```

### Variants

- **D3** (D3§1.5; nets.py:144-197): trunc-normal fan-in for every Linear, BlockLinear and Conv. Output outscales:

  | Output layer | outscale |
  |---|---|
  | reward | **0** |
  | value | **0** |
  | policy (mean, std or logits) | 0.01 |
  | continue | 1 |
  | RSSM logits | 1 |
  | decoder | 1 at HEAD; 0.1 at 2411f7d |

  The slow critic starts as an exact copy of the critic.
- **T2** (T2§1.13; init.py):
  - N(0, 0.02) for all Linears. Torch `trunc_normal_(std=.02, a=-2, b=2)` has **absolute** bounds of ±100σ, so it is effectively untruncated.
  - Zero **final weight** for the reward head and for **every** Q member.
  - π, encoder, dynamics and termination heads are not zeroed.
  - Each Q member gets its own key.
  - The target Q is cloned **after** the zeroing.

### False friends

1. **`jax.nn.initializers.truncated_normal(0.02)` is wrong for T2.** It truncates at ±2σ = ±0.04 with no std correction (std **0.0176** in jax 0.7.2). Use `normal(0.02)`, which gives std 0.02.
2. **`flax lecun_normal` is D3's distribution up to a constant.** Flax's 1/0.87962566 = 1.1368472 differs from D3's literal 1.1368 by 4.2e-5 relative (max abs difference 5.9e-6 at the same key). Use a 3-line custom initialiser with the literal constant so parity tests are exact.
3. **Ajax `parse_layer` defaults to orthogonal(1.0).** Neither paper uses it.

### Tests

1. `dreamer_trunc_normal()(k, (256, 512))` == `jax.random.truncated_normal(k, -2, 2, (256, 512)) * 1.1368 * sqrt(1/256)` exactly. Its std ≈ 0.0625.
2. BlockLinear kernel (8, 1024, 256): fan_in = 8192 and std ≈ 0.011048.
3. `normal_init(0.02)` has std ≈ 0.0200 and samples beyond ±0.04 exist.
4. The zeroed heads have all-zero kernel and bias. Every two-hot head then decodes to 0.
5. Ensemble members differ pairwise. Target and slow trees equal the online trees bitwise at init.

---

## B6. BlockLinear (D3 only)

**Math.** Kernel W has shape (g, I/g, U/g) and the bias has shape (U,). The output is y = einsum('...ki,kio->...ko', x.reshape(..., g, I/g), W).reshape(..., U) + b. Init fan_in = I, the full width (B5). Used in the GRU core: `dynhid0` with U = D and `dyngru` with U = 3D (D§2.4).

**Signature.** `BlockLinear(features: int, blocks: int, kernel_init, dtype)`.

**Tests.**
1. The Jacobian block (output k, input j) is zero for j ≠ k.
2. The result equals a dense layer with a block-diagonal kernel.
3. 12M shapes: dynhid (8, 1024, 256); dyngru (8, 256, 768).
4. Init std as in B5.

---

## B7. Grouped softmax: SimNorm (T2) vs categorical latents (D3), a false friend

**Shared primitive.** `grouped_softmax(x, group) = softmax(x.reshape(..., D/group, group), -1).reshape(..., D)`.

- **T2 SimNorm** (T2§1.3; layers.py:74-91): group V = 8, τ = 1, deterministic. Placed after the LayerNorm at the end of the encoder and the dynamics. The **probabilities are the latent**: there is no sampling and no unimix. Paper Eq. 5 divides by τ, while the App. H prose describes the opposite. If τ is exposed, use Eq. 5.
- **D3 latent** (D3§1.8): logits are reshaped to (S = 32, C = d/16). The probabilities get **1% unimix**, and the latent is a **sampled one-hot** with straight-through gradients (B8).

They share the reshape and softmax and nothing else. Do not build D3 latents on SimNorm.

**Tests.**
1. simnorm(arange(16)/4, V = 8) gives the same 8-vector in each group: [0.04445499, 0.05708134, 0.07329389, 0.09411121, 0.12084118, 0.15516315, 0.19923344, 0.2558208].
2. Each group sums to 1.
3. The output is invariant to adding a constant per group.

---

## B8. Unimix categorical, straight-through sample, KL and free bits (D3 only)

### Math (D3§1.8, §2.14; outs.py:208-270)

- **Unimix:** p = (1 − u)·softmax(l) + u/C, and l' = log p, with u = 0.01.
- **Sample:** idx ~ Cat(l'). The value is sg(onehot(idx)) + p − sg(p), so the gradient flows through the **mixed** p.
- **KL(P‖Q)** = Σ_C softmax(l'_P)·(log_softmax(l'_P) − log_softmax(l'_Q)), summed over the S latents.
- **Unimix is re-applied after any sg.** Store raw logits.
- **Free bits:** max(KL_summed_over_S, 1.0) per (b, t).
- dyn = KL(sg(post) ‖ prior) and rep = KL(post ‖ sg(prior)).

### Signature

```python
def unimix_logits(logits, unimix=0.01) -> logits
def onehot_st_sample(key, logits, unimix=0.01) -> f32 one-hot w/ ST grads
def categorical_kl(logits_p, logits_q, unimix=0.01) -> per-latent KL
def categorical_entropy(logits, unimix=0.01)
def free_bits(x, nats=1.0) = jnp.maximum(x, nats)
```

### Variants

- **Discrete actor.** The paper and 2411f7d use 1% unimix. HEAD's 'categorical' head uses **none** (D3§3.14, OQ 5). Expose `unimix` as a head option.
- **T2:** not used.

### Tests (pinned)

1. unimix probs of [10, 0, 0, 0] = [0.9923651, 0.00254494, 0.00254494, 0.00254494]. Uniform logits stay uniform.
2. **KL with S = 2, C = 4:**
   - post = [[2, 0, 0, 0], [0, 1, 0, 0]] and prior = zeros give Σ KL = **0.5745868**. Free bits give 1.0, with **zero gradient**.
   - post = [[8, 0, 0, 0], [0, 8, 0, 0]] gives 2.6559892, and the gradient passes.
3. The straight-through value is exactly one-hot.
4. d sample[0] / d logits at logits [10, 0, 0, 0] (unimix) = [1.3480149e-4, −4.4933695e-5, −4.4933695e-5, −4.4933695e-5]. This is the Jacobian of the **mixed** softmax.
5. **Stop-gradient placement:** grad of dyn w.r.t. post logits is 0, and grad of rep w.r.t. prior logits is 0.

---

## B9. Q ensemble with random-pair reduction (T2 only; D3 has a single V critic plus a slow copy)

### Math (T2§1.10-1.12, §3.14; world_model.py:186-216)

- Nq members (5 by default; 2 at 1M; 8 at 317M). Each member is NormedMLP([512, 512], 101, dropout = 0.01).
- The members are vmapped over stacked params with the input broadcast, and each member gets its **own dropout mask**.
- **'all'** returns logits of shape (Nq, ..., K).
- **'min' / 'avg':** one `permutation(Nq)[:2]` per **call**, shared across the whole batch and horizon. Decode each head (B2), then take the min or the mean of the **decoded scalars**.

### Signature

```python
class Ensemble(nn.Module):
    member: Callable[[], nn.Module]; num: int
    # nn.vmap(member_cls, in_axes=None, out_axes=0, variable_axes={"params": 0},
    #         split_rngs={"params": True, "dropout": True}, axis_size=num)
def random_pair(key, num) -> idx[2]                  # jax.random.permutation(key, num)[:2]
def reduce_pair(q_scalars[2, ...], how: Literal["min", "avg"])
```

**Usage map:**

| Use | Network | Reduction | Dropout | Gradient |
|---|---|---|---|---|
| TD target | target params | 'min' | off | — |
| Value loss | online params | 'all' | on | — |
| Policy loss | sg(online params, post-step) | 'avg' | on | flows w.r.t. the action |
| Planner terminal value | online params | 'avg' | off | — |

### False friends

- **Ajax `networks.MultiCritic`.** Its nn.vmap pattern is reusable, but `split_rngs` lacks `"dropout"`, so the members would share one mask. Its member is Ajax's `Critic`, which has its own Encoder and LayerNorm output. Write a fresh member.
- **Reduction order.** Reduce **decoded scalars**, not logits or probabilities.

### Tests

1. Members have distinct params, and each member is initialised from its own key.
2. With identical inputs and `deterministic=False`, the dropout masks differ across members.
3. With a zero final layer, all decoded Qs are 0 at init.
4. `random_pair` always returns 2 distinct indices. With Nq = 2 it returns {0, 1}.
5. The same pair is used for every batch element.
6. 'min' of the decoded pair differs from decode(mean of logits) on a crafted example.
7. Parameter count: 2,911,225 (5 members, walker).

---

## B10. EMA target / slow network

**Math.** θ_tgt ← rate·θ + (1 − rate)·θ_tgt over **all** leaves, including norm scales and biases. Initialise with an exact copy.

**Signature.** `ema_update(target, online, rate) = optax.incremental_update(online, target, rate)`. This is Ajax's existing `LoadedTrainState.soft_update`; reuse it.

### Variants

| | D3 slow critic (D3§3.18) | T2 target Q (T2§2.15) |
|---|---|---|
| Rate | 0.02 (every 1 step) | 0.01 |
| When | after the single optimiser step | after **both** the world-model and π steps |
| Role | only a regularisation target: two-hot CE toward its **scalar mean**, weight 1.0. **Never** used for bootstrap or baseline (slowtar = False). | TD bootstrap (min of 2). No target for the encoder, dynamics or π. |

**False friend: the role.** A "target critic" in D3 must **not** feed the λ-return.

**Tests.**
1. Target 0 and online 1 give 0.02 (D3) and 0.01 (T2) after one step.
2. At init the trees are identical.
3. Leaf coverage includes LayerNorm γ and β.
4. The target receives no gradient.

---

## B11. λ-returns

### Math (time-major, rlax convention)

Index t is the transition t → t+1:
- G_{T−1} = r_{T−1} + d_{T−1}·v_{T−1}
- G_t = r_t + d_t·[(1 − λ_t)·v_t + λ_t·G_{t+1}]

Here v_t = V(s_{t+1}) is the bootstrap at the **arrival** state, d_t is the per-step discount times continuation (soft allowed), and λ_t is the per-step λ times "not cut".

**Signature.** `lambda_returns(rew, disc, boot, lam) -> G` with all inputs [T, ...] f32, as a reverse `lax.scan` with carry init boot[−1]. lam may be a scalar or [T, ...].

### Mapping from D3's lambda_return (agent.py:480-488)

D3's arrays have length L and the outputs are R_0..R_{L−2}. Set:
- rew_t := rew[t+1]
- disc_t := disc·(1 − term[t+1])
- lam_t := λ·(1 − last[t+1])
- boot_t := boot[t+1]

**Uses:**
- **D3 imagination:** disc = 1 (contdisc), term = 1 − ĉ (soft), last = 0, boot = online v, λ = 0.95.
- **D3 replay:** disc = 1 − 1/333, term/last = the replay flags, boot = imag R_0 reshaped to (B, K).
- **T2 one-step TD target:** y = r + γ(1 − term)·Q̄min(h(s')). This is λ = 0 with T = 1. The bootstrap state is the **encoded** next obs (T2§2.3).
- **T2 planner value:** Σ γ^t r̂_t + γ^H·Q_avg. This is λ = 1 with disc = γ.

Both T2 uses are degenerate cases. Use the block or inline them; neither needs it.

### False friends

1. **D3 indexing.** Step t uses r̂, ĉ and v of the **next** state, and the continuation weight w_t = Π_{i≤t} ĉ_i includes ĉ_0. The weight is computed outside this block.
2. **Ajax PQN `compute_q_lambda_targets`** has the same recursion, but it casts done flags to **bool**, which breaks the soft ĉ. It also takes terminated and truncated separately. Do not reuse it for D3 as is.

### Tests (pinned, D3 indexing)

Inputs: L = 5, rew = [0, 1, 2, 3, 4], boot = [10, 20, 30, 40, 50].

| Case | Expected R_0..R_3 |
|---|---|
| λ = 0.95, disc = 0.99 | [54.179844, 55.491592, 55.29675, 53.5] |
| term at index 2 | [3.8710003, 2.0, 55.29675, 53.5] |
| last (truncation) at index 2 | [31.803852, 31.7, 55.29675, 53.5] |
| soft ĉ = [.997, .997, .5, .997, .997], disc = 1 | [29.794964, 29.349062, 55.998024, 53.85] |
| λ = 0 | [20.8, 31.7, 42.6, 53.5] |
| λ = 1 | [57.831295, 57.40535, 55.965, 53.5] |

The time-major block must reproduce these through the mapping above.

---

## B12. Gaussian policy heads (shared primitives, separate heads; a false friend)

### Shared primitives

```python
def diag_gaussian_logpdf(x, mean, std) -> per-dim
def diag_gaussian_entropy(std) = 0.5*log(2*pi*e*std^2) per-dim
def reparam_sample(key, mean, std) = mean + std * eps
def tanh_squash_logdet(u, eps=1e-6) = sum(log(relu(1 - tanh(u)^2) + eps))   # T2 form (NOT distrax's exact log-det)
```

### D3 'bounded_normal' (D3§3.13; heads.py:146-155)

- μ = tanh(m) and σ = 0.9·sigmoid(s + 2) + 0.1, so σ ∈ (0.1, 1).
- Normal(μ, σ) is **not squashed**: samples are unbounded, the log-prob is the plain Normal logpdf of the unclipped sample, and there is no Jacobian term.
- The entropy is that of the unsquashed Normal, summed over dims.
- The gradient estimator is **REINFORCE** on sg(action).
- The env clips the action, the replay stores it unclipped, and the RSSM divides the action by max(1, abs(a)).
- Both output Linears use outscale 0.01.

### T2 policy prior (T2§1.8-1.9, §2.12)

- log σ = −10 + 6·(tanh(raw) + 1), so log σ ∈ (−10, 2).
- a = tanh(μ + σ·ε), with **reparameterised** gradients.
- lp_g = Σ(−½ε² − log σ − ½ln2π). The log-prob subtracts tanh_squash_logdet(u, 1e-6).
- The entropy bonus is scaled by n, the number of valid action dims. PE and HEAD differ here (T2 open question).
- Planning and TD targets always use the **sample**. Acting with mpc = false in eval uses tanh(μ).
- The last layer is N(0, 0.02) and is not zeroed.

### False friends

1. **D3 is not a SAC-style tanh-Gaussian.** Do not reuse Ajax's `SquashedNormal` for D3.
2. **T2's Jacobian uses +1e-6 inside the log.** Ajax's `SquashedNormal` uses the exact stable log-det. At saturation (abs(u) > 7) T2's version is flat (log 1e-6), while the exact one has slope ±2. This changes PE gradients.
3. **The bounds are on different quantities.** D3 bounds σ with a sigmoid and a +2 offset. T2 bounds **log** σ with a tanh.

### Tests (pinned)

1. **D3:** σ at raw 0 = **0.8927173**; entropy per dim at init = 1.3054532; logpdf of an out-of-[−1, 1] sample is finite and has no Jacobian term.
2. **T2:** log_std(0) = −4, log_std(+∞) = 2, log_std(−∞) = −10.
3. **T2 worked example.** Inputs: ε = [0.5, −1], μ = [0.1, −0.2], raw = [0, 1].

   | Quantity | Value |
   |---|---|
   | log σ | [−4, 0.5695648] |
   | u | [0.10915782, −1.9674977] |
   | a | [0.10872632, −0.9616578] |
   | lp_g | 0.967558 |
   | Σ squash log-det | −2.5992928 |
   | HEAD lp | 3.5668507 |
   | HEAD scaled_entropy ≈ −n·lp_g | −1.935116 |
   | PE log_pi = n·(Σ(−½ε² − log σ) − ½ln2π) − squash | 6.372286 |

---

## B13. MPPI planner (T2 only; generic)

### Math (T2§3.1-3.24, pseudocode 3.C)

- **Warm start.** μ ← shift(prev_mean) (last row 0; zeros if t0). σ ← max_std. **σ is never warm-started.**
- The P = 24 π-trajectories are fixed for all iterations but re-scored every iteration.
- **Each iteration** (6, or 8 if A ≥ 20):
  - Sample the N − P Gaussian candidates and clip them to [−1, 1] **before** evaluation.
  - V = nan_to_num(value_fn(·)).
  - Take the top-E elites from the current iteration only.
  - score = softmax(0.5·(eV − max eV)). The temperature **multiplies**.
  - μ = Σ s·eA / (Σs + 1e-9).
  - σ = sqrt(Σ s·(eA − μ_new)² / (Σs + 1e-9)), the biased std around the new mean, clipped to [0.05, 2].
  - Mask μ and σ for MT after the clip.
- **Final action.** Pick k ~ Cat(score) and take a = eA[0, k]. Add σ[0]·N(0, I) if not eval. Clip to [−1, 1]. Store prev_mean = μ (not the executed action).

### Signature

```python
@dataclass(frozen=True)
class MPPIConfig: horizon=3; iterations=6; num_samples=512; num_pi_trajs=24; num_elites=64
                  min_std=0.05; max_std=2.0; temperature=0.5
def mppi_plan(key, value_fn, pi_actions | None, prev_mean, t0, eval_mode, cfg, action_mask=None)
    -> (action[A], new_prev_mean[H,A], aux{score, mean, std})
# value_fn(key, actions[H,N,A]) -> V[N]  (caller closes over z, networks; T2: Σγ^t r̂(z_t,a_t) + γ^H Q_avg(z_H, π(z_H)))
```

All shapes are static. Use `lax.fori_loop`, `lax.top_k` and `jax.random.categorical(log score)`. Per-env t0 and prev_mean come from the actor state (vmapped over envs; T2§3.A).

### Tests (pinned)

1. **Update step.** Inputs: H = 2, E = 3, A = 1, eV = [1, 2, 4], eA[0] = [0, .5, 1], eA[1] = [−1, 0, .2].
   - score = [0.14024438, 0.2312239, 0.6285317]
   - μ = [0.74414366, −0.01453803]
   - σ = [0.3641262, 0.4064164]
2. Warm start shifts only μ, and the last row is 0. σ resets to 2. t0 gives zeros.
3. The π columns are unchanged across iterations.
4. Candidates lie in [−1, 1].
5. eval_mode removes **only** the final noise.
6. Sanity: with V = −‖a − a*‖², μ converges toward a*.
7. Under jit and vmap over envs, the result equals the per-env loop.

---

## B14. Latent rollout scan (both)

**Signature.** `rollout(step_fn, carry0, length, xs=None, policy=None, key=None) -> (carry_T, outs[T])`, a time-major `lax.scan`. It takes either given inputs (xs) or a policy callable sampled per step with split keys.

**Uses:**
- **D3:** observe (posterior step with is_first masking, Algorithm B); imagine (the policy sees sg(carry); prior straight-through sample, Algorithm C), with H = 15.
- **T2:** consistency rollout with **buffer** actions over H = 3; π-trajectory rollout (P = 24); planner value rollout. The T2 dynamics are deterministic.

**Tests.**
1. The scan equals a Python loop.
2. With a linear step function, the result matches the closed form.
3. Keys are split per step: no repeated noise.
4. Gradient flows through the carry unless the caller applies sg.

---

## B15. RSSM block-GRU core and observe/imagine steps (D3 only)

### Math

The core follows D3 Algorithm A:
- x0, x1, x2 = SiLU(RMS(Linear_d(·))) of deter, the flattened stoch, and a / sg(max(1, abs(a))).
- Concatenate per block with each block's deter slice, then BlockLinear(D) → RMS over the **full** D → SiLU.
- BlockLinear(3D) with no norm and no activation. The per-block layout is [r | c | u].
- r = σ(r), c = tanh(r·c) (the reset gate scales the **whole** candidate), u = σ(u − 1).
- h' = u·c + (1 − u)·h.

Observe and imagine steps follow Algorithms B and C, including the double masking of is_first (zero deter, stoch and prevact **before** the core, and zero the encoded action again after one-hot). The initial state is zeros, not learned.

### Shared sub-blocks

NormedMLP (B4), BlockLinear (B6), unimix straight-through (B8), rollout (B14), and Ajax `memory._mask_reset` for is_first zeroing (zero fresh carry).

### False friend: Ajax `MemoryCell("gru")`

Ajax's `MemoryCell("gru")` (flax GRUCell) is a **different cell**:
- the reset gate applies only to the recurrent term;
- there is no −1 update bias;
- it is dense, not block-diagonal;
- it has no norm or hidden layer.

Do not reuse it. T2 has no recurrence: the latent is re-encoded every env step.

### Tests

1. With the dyngru kernel and bias zeroed: u = σ(−1) = 0.2689414 exactly, and h' = 0.7310586·h.
2. is_first gives h = core(0, 0, 0) regardless of the carry and the previous action. A one-hot of action index 0 is masked.
3. Block structure: output block k depends on other blocks only through x0.
4. **Shapes and parameter counts at 12M** (d = 256, S = 32, C = 16, A = 6). These are derived from the reference layer shapes, not from a reference run.
   - dynin0 524,800; dynin1 131,584; dynin2 2,048; dynhid0 2,101,248; dyngru 1,579,008; core total **4,338,688**.
   - Prior MLP 722,432; posterior 721,920.

---

## B16. Bernoulli head (D3 continue; T2 termination as an Extension)

**Math.** loss(l, c) = −[c·log σ(l) + (1 − c)·log σ(−l)]. Soft labels are allowed.

**Variants.**
- **D3** (D3§2.13): target c = 0.997·(1 − is_terminal), a **soft** label; never is_last. Imagination uses σ(l) as ĉ (γ folded in).
- **T2 HEAD episodic** (T2§1.7, §2.7, §3.11): off by default, so it belongs in an **Extension**. It is a hard 0/1 label, BCE-mean on the **predicted** z_1..z_H. Planning thresholds it at 0.5 and treats termination as absorbing.

**Tests.**
1. l = 0 gives log 2 = 0.6931472 for any c.
2. At c = 0.997 the minimum is at l = logit(0.997) = 5.8061385, where loss = 0.0204229 and the gradient ≈ 0.

---

## B17. Optimisers

### Builders (shared optax primitives)

```python
def clip_by_agc_per_tensor(clip=0.3, pmin=1e-3)   # custom: g * 1/max(1, ||g|| / (clip*max(pmin, ||p||))) per LEAF
def laprop_agc(lr=4e-5, warmup=1000, agc=0.3, pmin=1e-3, beta1=0.9, beta2=0.999, eps=1e-20):
    sched = optax.join_schedules([optax.linear_schedule(0., lr, warmup), optax.constant_schedule(lr)], [warmup])
    return optax.chain(clip_by_agc_per_tensor(agc, pmin),
                       optax.scale_by_rms(decay=beta2, eps=eps, eps_in_sqrt=False, bias_correction=True),
                       optax.ema(decay=beta1, debias=True),          # == D3 scale_by_momentum
                       optax.scale_by_learning_rate(sched))
def adam_grouped(lr=3e-4, lr_scales={"encoder": 0.3}, label_fn=..., clip_norm=20.0, eps=1e-8):
    return optax.chain(optax.clip_by_global_norm(clip_norm),        # ONE global norm over all groups
                       optax.multi_transform({g: optax.adam(lr * s, eps=eps) for g, s in ...}, label_fn))
```

### Variants

- **D3** (D3§4.3-4.7; opt.py:109-164): **one** optimiser over all modules. The order is AGC, then RMS, then momentum, then the lr. The first update has lr = 0 (warmup).
- **T2** (T2§2.9-2.14):
  - **World model:** `adam_grouped(3e-4, {encoder: 0.3})`, eps 1e-8, global clip 20.
  - **π:** a separate `chain(clip_by_global_norm(20), adam(3e-4, eps=1e-5))`.
  - Torch's clip coefficient adds +1e-6 to the norm; this is negligible.

### False friends

1. **`optax.adaptive_grad_clip` is NFNet unit-wise AGC.** Example with p = I₂ and g = [[3, 0], [0, 0.1]]:
   - D3 per-tensor AGC gives [[0.42402855, 0], [0, 0.01413429]].
   - optax gives [[0.3, 0], [0, 0.1]].
   - Its `axis` argument cannot express "whole tensor" for a tree with mixed ranks, so a custom transform is needed.
2. **Ajax `get_adam_tx` defaults to eps 1e-5 and has no param groups.** T2's world-model Adam needs eps **1e-8**. Its π Adam needs 1e-5.
3. **Order of clip and groups.** In T2 the global clip must come **before** `multi_transform`. Clipping per group is wrong.
4. **β₂.** The D3 paper says 0.99 and the code says 0.999 (D3 OQ 2). Keep it as a config value.

### Tests (pinned)

1. The `laprop_agc` chain equals the D3 reference transforms (copied from opt.py) to max abs difference **2.1e-14** over 5 steps on a {(5, 3), (3,), (2, 4, 3)} tree.
2. The first update is all zeros (lr_0 = 0).
3. **AGC:** p = ones(2, 2), g = 3·ones gives 0.3·ones (cap 0.6). p = 0, g = [3, 4] gives [1.8e-4, 2.4e-4]. Include the unit-wise counterexample above.
4. **T2 grouped Adam:** g_enc = 30 and g_rest = 40 are globally clipped to 12 and 16. The first-step updates are −8.99994e-5 (encoder) and −2.99998e-4 (rest).
5. **Scale invariance (D3):** multiplying the loss by 1000 leaves the LaProp updates unchanged (eps 1e-20).

---

## B18. Sequence replay (shared ring buffer, per-algorithm window predicate)

### Design (pure-JAX, on device)

- **Storage.** Per-env ring buffers [N_env, cap] of step rows {obs…, action, reward, is_first, is_last, is_terminal, (+ latent fields)}, a per-env write pointer and a per-env "committed" pointer.
- **Sampling** returns `(batch, (env_idx, start_idx))`. The indices are required for D3's latent write-back. Validity is a predicate over the window flags.

### Variants

| | D3 (D3§5; replay.py) | T2 (T2§2.1, §4.6-4.9) |
|---|---|---|
| Window | 65 steps (64 trained + 1 context), stride-1 starts | H+1 = 4 rows |
| Episode boundaries | windows **cross** them (mid-window is_first is normal) | windows **never** cross them (valid iff no is_first in rows 1..H) |
| Sampling | uniform over items, with replacement; **online queue** of non-overlapping fresh windows popped first (train only) | uniform over episodes, then start (= uniform over valid windows for fixed-length episodes) |
| Annotation | is_first[:, 0] = 1; is_last ∨= roll(is_first, −1), last column excluded | none |
| Freshness | everything inserted is sampleable | **only completed episodes** are sampleable |
| Extra fields | stored deter/stoch per step, written back after each update (indices 1..64) | none |
| Capacity | 5e6 items | 1e6 transitions (b67b21c: 2000 episodes × 501 rows) |

### False friends

1. **Action alignment.**
   - D3 row t = (x_t, r_t received **on entering** x_t, a_t chosen **after** x_t). prevact for row t is the action stored at row t − 1.
   - T2 row k = (o_k, a_{k−1}, r_{k−1}), the action that **led into** o_k, with a placeholder at row 0.
   - Rewards align the same way; actions are shifted by one. Pick one storage convention and derive the other in the sampler.
2. **Ajax's collector writes SARS′ `Transition`s under gymnax auto-reset.** Each step is (obs, action, reward, terminated, truncated, next_obs), with next_obs = the final obs on done. So an episode has T rows, and the terminal obs exists only as next_obs.
   - T2 maps cleanly onto this: a window is H consecutive transitions with no done among the first H−1, with obs[0..H] = obs_t..obs_{t+H−1} plus next_obs_{t+H−1}.
   - **D3 does not map cleanly.** It needs the terminal obs as its **own row** (is_last, plus is_terminal if terminal) so that the RSSM filters it and the reward and continue heads train on it. That is T+1 rows per episode, which breaks per-env lockstep. See Q1.
3. **Ajax `get_buffer(sequence_length=…)` (flashbax trajectory buffer)** gives stride-1, episode-crossing windows like D3's. However, it returns **no indices** (so no write-back), has no online queue, and has no episode-validity filter.

### Tests

1. **D3 annotation.** Flags is_first = [0, 0, 1, 0], is_last = [0, 0, 0, 0] become is_first = [1, 0, 1, 0] and is_last = [0, 1, 0, 0].
2. **T2 validity.** No sampled window contains is_first after row 0. Over 10⁵ draws, a chi-square test is uniform over valid starts. In a buffer with one completed episode and one running episode, no sample comes from the running one.
3. **Alignment round trip.** Ajax Transitions → T2 slice gives obs[H] == next_obs of the last transition.
4. **Write-back.** Values written at the returned (env, idx) are read back on the next sample of the same window.
5. **Ring wrap-around** keeps the per-env order.

---

## B19. Small shared utilities and non-shared look-alikes

- **Discount from horizon.**
  - D3: γ = 1 − 1/333 = 0.996997.
  - T2: γ(T) = clip(1 − 5/T, 0.95, 0.995), giving {T = 100: 0.95, 500: 0.99, 1000: 0.995, 50: 0.95}.
  - Signature: `discount_from_horizon(h, lo=None, hi=None)`, with h = 333 (D3) or h = T/5 (T2).
- **Mixed precision.**
  - D3: bf16 compute, with f32 params, norm statistics, distributions, losses, returns and normalisers.
  - T2: f32 (HEAD uses TF32 matmuls).
  - Every module takes `dtype`. Distribution, loss and normaliser code upcasts to f32. Parity tests set `jax_default_matmul_precision="highest"`.
- **Categorical pick.** `jax.random.categorical` replaces T2's Gumbel-max and `np.random.choice`, and D3's sampling.
- **Not shared (look-alikes):**
  - **MSE losses:**
    - D3 symlog-MSE decoder: (pred − symlog(y))², **summed** over feature dims, no ½, mean over (B, T).
    - T2 consistency loss: plain MSE, **mean over batch and latent dim**, no symlog, weighted by ρ^t and divided by H.
  - **Temporal weights:** D3 uses cumprod(ĉ) including ĉ_0; T2 uses ρ^t = 0.5^t.
  - **Loss reductions:** D3 takes means over (B, T) and H. T2 uses Σ_t ρ^t·mean_B, divided by H (or H·Nq for values).

---

## Open questions for the user (shared-block scope)

1. **D3 replay rows under Ajax's auto-reset environments.** This blocks B18 for D3. The options:
   - **(a)** An explicit-reset env wrapper, as in the embodied driver. After an is_last step, the next `step` call ignores the action and emits the reset obs with is_first = 1 and reward 0. This gives one row per vector step, keeps lockstep and reproduces the reference stream exactly, including the train-ratio clock counting reset steps.
   - **(b)** Keep auto-reset, and write 2 rows on done steps with per-env write pointers.
   - **(c)** Store transitions, and synthesise the terminal row at sample time.

   **Recommended: (a).** It is the reference's own data stream and the simplest correct option.
2. **T2 "only completed episodes are sampleable".** Should we implement a per-env committed pointer (recommended; cheap, exact) or accept per-step visibility as a documented deviation?
3. **Package location and retrofitting.** The proposal is a new `src/ajax/blocks/` package with `transforms`, `twohot`, `normalizers`, `layers`, `distributions`, `ensemble`, `returns`, `planning`, `rssm`, `optim` and `replay`. Should existing Ajax code be retrofitted onto these blocks now, or left alone? The candidates are PQN q-lambda → B11, `soft_update` → B10 (already identical) and `get_adam_tx` → B17.
4. **bf16 support** (D3 OQ 10). Should the blocks support a bf16 compute dtype from day one (recommended), or start f32-only?

Recommendations that do not need a user decision:
- Use the symmetric two-hot sum for both algorithms (B2).
- Use the literal-constant D3 initialiser (B5).
- Use `normal(0.02)`, not `truncated_normal(0.02)`, for T2 (B5).
- Use a custom per-tensor AGC (B17).
