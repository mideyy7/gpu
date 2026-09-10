# Part 04 — The compute shaders

Files: `shaders/force_calc.comp` (43 lines), `shaders/integrate.comp` (35
lines). (`stub.comp` is a 3-line no-op left over from early scaffolding —
ignore it.)

These are the programs that actually run on the GPU, once per particle, in
parallel. They're written in **GLSL**, compiled to **SPIR-V** by `glslc`,
and loaded at runtime.

---

## 1. GLSL in 3 minutes

GLSL looks like C with built-in vector types.

- **Types:** `float`, `uint`, `int`, `bool`, and vectors `vec2`, `vec3`,
  `vec4` (2/3/4 floats). Also `uvec3` etc.
- **Swizzling:** if `v` is a `vec4`, `v.xyz` is a `vec3` of its first three
  components, `v.w` is the fourth, `v.x` the first. You can read and assign
  swizzles: `p.xyz = newVec3;`
- **Vector math is element-wise and built in:** `a + b`, `a * scalar`,
  `dot(a, b)` (dot product), `inversesqrt(x)` (= 1/√x, one fast
  instruction).
- **`void main()`** is the entry point, same as C.
- **No pointers, no dynamic allocation, no recursion.** Shaders are meant to
  be simple and fast.

### The layout qualifiers

`layout(...)` annotations connect the shader to the outside world:

- `layout(local_size_x = 64) in;` — this shader's **workgroup size** is 64.
  One `vkCmdDispatch(n,1,1)` runs `n` workgroups × 64 invocations.
- `layout(std430, binding = 0) buffer Name { ... };` — declares a **storage
  buffer** at descriptor slot 0. `std430` is the memory layout rule (see
  §4). `readonly` / `writeonly` are optimisation hints.
- `layout(push_constant) uniform Name { ... } pc;` — declares the
  push-constant block; `pc` is the name you read fields through.

### Built-in inputs

- `gl_GlobalInvocationID` — a `uvec3`. `.x` is this invocation's global
  index across the whole dispatch (0 … total-1). Each of the 64 threads in
  each of the N workgroups gets a distinct `.x`.

---

## 2. The physics being simulated

### Newtonian gravity between N bodies

Every particle `i` is pulled toward every other particle `j`. The
acceleration on `i` from `j` is:

```
        G * m_j
a_ij = --------- * (p_j - p_i) / |p_j - p_i|
       |p_j - p_i|²
```

- `G` — the gravitational constant (here just a tunable `gravity` scalar).
- `m_j` — mass of `j`.
- `(p_j - p_i)` — the direction from `i` toward `j` (a vector).
- `|p_j - p_i|²` — distance squared; gravity falls off with distance
  squared.
- The extra `/ |p_j - p_i|` normalises the direction vector to length 1.

Combined, that's `G * m_j * (p_j - p_i) / |p_j - p_i|³`.

The total acceleration on `i` is the **sum over all j ≠ i**. That sum is
**O(N²)**: N particles, each looping over N others. Double N → 4× the work.
This is the expensive stage and the one whose scaling the sweep studies.

### Softening

If two particles get very close, `|p_j - p_i|³` → 0 and the acceleration
blows up to infinity (numerically: NaN). To prevent that, add a small
constant `softening²` to the distance-squared before taking the root:

```
distSqr = |p_j - p_i|² + softening²
```

Now even at zero separation the denominator is `softening³`, finite. This is
the standard "Plummer softening" trick in real N-body codes. `softening =
0.5` here.

### Integration: semi-implicit (symplectic) Euler

Once you have each particle's acceleration `a`, you advance time by a small
step `dt`:

```
v_new = v_old + a * dt          (update velocity using the acceleration)
p_new = p_old + v_new * dt      (update position using the NEW velocity)
```

The key detail is the order: position is updated with `v_new`, not `v_old`.
This variant ("semi-implicit" or "symplectic" Euler) conserves energy far
better over long runs than plain Euler — orbits stay stable instead of
spiralling out. It's one line different from the naive version and it's the
right default for physics.

### Momentum conservation (used by the tests)

Because the force `i` feels from `j` is exactly opposite the force `j` feels
from `i` (Newton's third law), when you sum `m_i * a_i` over *all* particles,
every pair cancels and the total is **zero**. So the total linear momentum
`Σ mᵢvᵢ` should not change from step to step. The Phase 3 self-test
(`momentumCase`) checks exactly this: it's a cheap invariant that catches
broken synchronisation or broken math without needing a reference
simulation.

---

## 3. `force_calc.comp` line by line

```glsl
#version 450
```

Which GLSL version. 450 = OpenGL 4.5 GLSL, the version Vulkan compute uses.

```glsl
layout(local_size_x = 64) in;
```

Workgroup = 64 invocations along x. (64 is a common sweet spot: a multiple
of the GPU's native execution width, small enough for good occupancy.)

```glsl
struct Particle {
    vec4 pos_mass;  // xyz = position, w = mass
    vec4 vel_pad;   // xyz = velocity (unused this phase), w = unused
};
```

The layout of one particle **as the GPU sees it**: two `vec4`s = 32 bytes.
Position and mass are packed into one `vec4` (`pos_mass.xyz` + `pos_mass.w`);
velocity into another. This must match the C++ `Particle` struct byte for
byte (§4).

```glsl
layout(std430, binding = 0) readonly buffer Particles {
    Particle particles[];
};
```

Descriptor slot 0 is a storage buffer holding an array of `Particle` of
run-time length (`[]` with no size). `readonly` — this shader only reads it.
`std430` — the packing rule.

```glsl
layout(std430, binding = 1) writeonly buffer Accel {
    vec4 accel[];  // xyz = acceleration, w = unused
};
```

Slot 1: an array of `vec4`, one acceleration per particle. `writeonly` —
this shader only writes it.

```glsl
layout(push_constant) uniform PushConstants {
    uint particleCount;
    float gravity;
    float softening;
} pc;
```

The 12-byte push-constant block. (`integrate.comp` adds a 4th field `dt`;
the C++ side uses one combined 16-byte struct for both — a shader is allowed
to declare fewer fields than the range provides.)

```glsl
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.particleCount) return;
```

`i` = which particle this invocation handles. If the dispatch rounded up and
`i` is past the end, do nothing.

```glsl
    vec3 pi = particles[i].pos_mass.xyz;
    vec3 acc = vec3(0.0);
```

`pi` = my position. `acc` = running sum of acceleration, starts at
`(0,0,0)`.

```glsl
    for (uint j = 0; j < pc.particleCount; ++j) {
        if (j == i) continue;
```

Loop over every other particle. Skip myself (a particle exerts no force on
itself; also avoids dividing by zero even with softening).

```glsl
        vec3 pj = particles[j].pos_mass.xyz;
        float mj = particles[j].pos_mass.w;
        vec3 d = pj - pi;
```

`pj` = j's position, `mj` = j's mass, `d` = vector from me toward j.

```glsl
        float distSqr = dot(d, d) + pc.softening * pc.softening;
```

`dot(d, d)` = `d.x² + d.y² + d.z²` = distance squared. Add `softening²`
(the anti-singularity trick).

```glsl
        float invDist = inversesqrt(distSqr);      // 1 / sqrt(distSqr)
        float invDist3 = invDist * invDist * invDist;  // 1 / distSqr^1.5
```

`inversesqrt` is a single fast GPU instruction. `invDist3` is `1 /
|d|³` (softened).

```glsl
        acc += pc.gravity * mj * invDist3 * d;
    }
```

Add this pair's contribution: `G * m_j * d / |d|³`. Accumulate.

```glsl
    accel[i] = vec4(acc, 0.0);
}
```

Write my total acceleration into slot 1, index `i`. `w` component unused, set
to 0.

**What this shader is:** "for my particle, sum the softened gravitational
pull from every other particle, and store the result."

---

## 4. `std430` layout and matching C++

The GPU and CPU must agree on the exact byte layout of a `Particle`, or the
GPU reads a position where the CPU wrote a mass.

`std430` (the rule for storage buffers) packs like this:

- a `float` occupies 4 bytes, aligned to 4,
- a `vec4` occupies 16 bytes, aligned to 16,
- a `struct` of two `vec4`s = 32 bytes, no padding.

The C++ side (`simulation.h:16-19`):

```cpp
struct Particle {
    float px = 0, py = 0, pz = 0, mass = 0;   // bytes  0..15  -> shader's pos_mass
    float vx = 0, vy = 0, vz = 0, _pad = 0;   // bytes 16..31  -> shader's vel_pad
};
```

Eight consecutive `float`s = 32 bytes. `px,py,pz` line up with
`pos_mass.xyz`; `mass` with `pos_mass.w`; `vx,vy,vz` with `vel_pad.xyz`;
`_pad` is a deliberate unused float so the struct is exactly 32 bytes and
the second `vec4` starts at byte 16 as `std430` requires.

The `accel` buffer is an array of `vec4` on the GPU; on the CPU
(`readAccelerations`, `simulation.cpp:426-433`) it's read as groups of 4
`float`s, taking indices `0,1,2` (the xyz) and skipping index `3` (the w).
That's why `accelBytes_ = sizeof(float) * 4 * particleCount`
(`simulation.cpp:104`) — 16 bytes per particle even though only 12 are used.

**Interview soundbite:** "The C++ `Particle` is eight floats laid out to
match the shader's two `vec4`s under `std430`, including a padding float, so
the same 32-byte record is valid on both sides."

---

## 5. `integrate.comp` line by line

```glsl
#version 450
layout(local_size_x = 64) in;

struct Particle {
    vec4 pos_mass;
    vec4 vel_pad;
};

layout(std430, binding = 0) buffer Particles {   // NOTE: no readonly/writeonly
    Particle particles[];
};

layout(std430, binding = 1) readonly buffer Accel {
    vec4 accel[];
};

layout(push_constant) uniform PushConstants {
    uint particleCount;
    float gravity;
    float softening;
    float dt;
} pc;
```

Same buffers, same slots. Differences:

- `Particles` here is **read-write** (no qualifier) — this shader both reads
  the old position/velocity and writes the new ones.
- `Accel` is `readonly` — it consumes what `force_calc` produced.
- The push-constant block has the extra `dt` (time step). `gravity` and
  `softening` are present but unused here — harmless, they're just part of
  the shared 16-byte block.

```glsl
// Semi-implicit (symplectic) Euler: advance velocity first, then position.
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.particleCount) return;

    vec3 v = particles[i].vel_pad.xyz + accel[i].xyz * pc.dt;
    vec3 p = particles[i].pos_mass.xyz + v * pc.dt;

    particles[i].vel_pad.xyz = v;
    particles[i].pos_mass.xyz = p;
}
```

- `v` = old velocity + acceleration × dt.
- `p` = old position + **new** velocity × dt (the symplectic order).
- Write both back, in place, touching only `.xyz` so `mass` (`pos_mass.w`)
  and the pad are preserved.

**What this shader is:** "move my particle one time-step using the
acceleration the force pass computed."

---

## 6. Why two shaders and not one?

You could compute force and integrate in a single shader. Two reasons this
project splits them:

1. **Correctness.** In the force loop, particle `i` reads *every* particle's
   position. If a combined shader also *wrote* positions in the same pass,
   particle `i` might read particle `j`'s position after `j` already moved
   this step — inconsistent state. Splitting guarantees all forces are
   computed from the same snapshot, then all particles move.
2. **The profiler needs stages.** Two dispatches with a barrier between them
   is exactly the "dispatch → barrier → dispatch" structure the profiler
   measures. One fused shader would be one opaque number.

---

## Explain it in one sentence

> `force_calc.comp` runs once per particle and sums the softened O(N²)
> gravitational acceleration from every other particle into an output
> buffer; `integrate.comp` runs once per particle and advances its velocity
> then position by `dt` using semi-implicit Euler; they're split so all
> forces come from one consistent snapshot and so the profiler has two
> distinct dispatch stages to measure.

## Questions you should be able to answer

1. What is `gl_GlobalInvocationID.x` and why does the shader start with
   `if (i >= particleCount) return;`?
2. What does "softening" do and what would happen without it?
3. What is semi-implicit Euler and how is its one line different from naive
   Euler? Why prefer it?
4. Why is the force calculation O(N²)?
5. How does the C++ `Particle` struct match the shader's `Particle`, and
   what is the `_pad` float for?
6. Why does `force_calc.comp` mark `Particles` `readonly` but
   `integrate.comp` doesn't?
7. Why are force and integration two separate shaders instead of one?
8. What invariant does momentum conservation give the tests, and why is it
   true here?
