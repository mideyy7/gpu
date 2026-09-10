# Part 08 — `report.{h,cpp}`: from raw numbers to a table

Files: `src/report.h` (36 lines), `src/report.cpp` (117 lines).

**Job:** collect one `StepTimings` per simulation step, then produce (a) a
human-readable table with mean / min / p50 / p99 / max for each stage, and
(b) a CSV with one raw row per step, and (c) a "means only" struct used by
the sweep.

This file is plain C++ — no Vulkan. If you found the Vulkan parts hard, this
one is a breather.

---

## 1. The header, `report.h`

```cpp
class Report {
   public:
    void addSample(const StepTimings& t);
    void printTable(std::FILE* out, const std::string& header) const;
    void writeCsv(std::FILE* out) const;

    struct Means {
        double force = 0, barrier1 = 0, integrate = 0, barrier2 = 0, copy = 0;
        double gpuTotal = 0, submitCall = 0, cpuWait = 0, frameTotal = 0;
    };
    Means means() const;

    size_t sampleCount() const { return frameTotal_.size(); }

   private:
    struct Stat {
        double mean = 0, min = 0, p50 = 0, p99 = 0, max = 0;
    };
    static Stat summarize(std::vector<double> v);

    std::vector<double> force_, barrier1_, integrate_, barrier2_, copy_;
    std::vector<double> gpuTotal_, submitCall_, cpuWait_, frameTotal_;
    bool anyGpuInvalid_ = false;
};
```

### The data model

A `Report` keeps **nine parallel `std::vector<double>`** — one per metric.
After `addSample` has been called `M` times, each vector has `M` elements,
and `force_[k]` is the force-dispatch time of step `k`, `barrier1_[k]` its
first barrier, etc.

"Parallel arrays" (index `k` means the same step across all nine vectors)
rather than "one vector of a 9-field struct." Both work; this makes
`summarize` trivial — it just takes one vector.

### `Stat` and `Means` — two little output bundles

- `Stat` (private) — the five summary numbers for one metric: mean, min,
  p50 (median), p99, max.
- `Means` (public) — just the means of all nine metrics, in one struct. Used
  by `--sweep`, which only wants one number per metric per particle count.

### `summarize` is `static`

`static Stat summarize(std::vector<double> v);` — `static` on a member
function means "doesn't need a `Report` object, it's just a function that
lives in the class's namespace." It's a pure function: vector in, `Stat`
out.

Note it takes `std::vector<double> v` **by value** (a copy), not by
reference — because it *sorts* the vector, and you don't want to disturb the
caller's data. The copy is the price of not mutating the original.

---

## 2. `addSample` (report.cpp:6-17)

```cpp
void Report::addSample(const StepTimings& t) {
    if (!t.gpuValid) anyGpuInvalid_ = true;
    force_.push_back(t.forceDispatchNs);
    barrier1_.push_back(t.barrier1Ns);
    integrate_.push_back(t.integrateDispatchNs);
    barrier2_.push_back(t.barrier2Ns);
    copy_.push_back(t.copyNs);
    gpuTotal_.push_back(t.gpuTotalNs);
    submitCall_.push_back(t.submitCallNs);
    cpuWait_.push_back(t.cpuWaitNs);
    frameTotal_.push_back(t.frameTotalNs);
}
```

Take one `StepTimings` and append each of its nine numbers to the matching
vector. If any sample had no GPU timestamps, remember that (`anyGpuInvalid_`)
so `printTable` can print a warning.

Called from `collectSteps` in `main.cpp`, once per step.

---

## 3. `summarize` (report.cpp:19-35)

```cpp
Report::Stat Report::summarize(std::vector<double> v) {
    Stat s;
    if (v.empty()) return s;                 // all-zero Stat for no data
    std::sort(v.begin(), v.end());           // ascending
    double sum = 0;
    for (double x : v) sum += x;
    s.mean = sum / static_cast<double>(v.size());
    s.min = v.front();                       // smallest (after sort)
    s.max = v.back();                        // largest
    auto pct = [&](double p) {
        double idx = p * static_cast<double>(v.size() - 1);
        return v[static_cast<size_t>(std::lround(idx))];
    };
    s.p50 = pct(0.50);
    s.p99 = pct(0.99);
    return s;
}
```

- `std::sort(v.begin(), v.end())` — sort the copy ascending. `.begin()` /
  `.end()` are iterators marking the range.
- `mean` — plain average.
- `min` / `max` — first / last element after sorting.
- **Percentile** via the `pct` lambda: the p-th percentile is the element at
  fractional index `p * (n-1)`, rounded to the nearest integer index
  (`std::lround`). So:
  - `pct(0.50)` on 200 samples → index `0.5 * 199 = 99.5` → round to 100 →
    the 101st-smallest value = the **median**.
  - `pct(0.99)` → index `0.99 * 199 = 197.01` → round to 197 → the
    198th-smallest = the value only ~1% of steps exceed.

### Why p50 and p99, not just mean?

This is the "latency mindset." A mean hides tail behaviour: if 99 steps take
1 ms and one takes 100 ms, the mean is ~2 ms and looks fine, but that one
100 ms hitch is what a user feels. **p99** ("99% of steps were at least this
fast") surfaces the tail. **p50** (median) shows the typical case
undistorted by outliers. Seeing mean, p50 and p99 side by side tells you
whether the workload is steady (all three close) or spiky (p99 far above
p50).

In the sample output, `barrier 1` has mean 44 µs but max 2630 µs — one
enormous outlier (a scheduling hiccup), which the mean partly absorbs but
p50 (31.6 µs) reveals as unrepresentative.

---

## 4. `printTable` (report.cpp:37-92)

```cpp
void Report::printTable(std::FILE* out, const std::string& header) const {
    if (frameTotal_.empty()) { std::fprintf(out, "no samples\n"); return; }

    const Stat frame = summarize(frameTotal_);
    const double meanFrame = frame.mean > 0 ? frame.mean : 1.0;   // avoid /0
```

Summarise the frame-total vector once; `meanFrame` is the denominator for
the "% of frame" column (guarded against zero).

```cpp
    struct Row { const char* name; Stat s; };
    const Row stageRows[] = {
        {"force dispatch",            summarize(force_)},
        {"barrier 1 (accel w->r)",    summarize(barrier1_)},
        {"integrate dispatch",        summarize(integrate_)},
        {"barrier 2 (part. w->xfer)", summarize(barrier2_)},
        {"readback copy",             summarize(copy_)},
    };
    const Row aggRows[] = {
        {"GPU total",         summarize(gpuTotal_)},
        {"submit call (CPU)", summarize(submitCall_)},
        {"fence wait (CPU)",  summarize(cpuWait_)},
        {"frame total",       frame},
    };
```

Two small arrays of `{label, Stat}`. `stageRows` are the five GPU pipeline
stages; `aggRows` are the roll-ups. Each `summarize(...)` call sorts a copy
and returns the five numbers.

```cpp
    std::fprintf(out, "\n%s\n", header.c_str());
    if (anyGpuInvalid_)
        std::fprintf(out, "(warning: some frames had no GPU timestamps)\n");
    std::fprintf(out, "samples: %zu frames\n\n", frameTotal_.size());

    const char* fmtHead = "%-26s %10s %10s %10s %10s %10s %8s\n";
    const char* fmtRow  = "%-26s %9.1fus %9.1fus %9.1fus %9.1fus %9.1fus %7.1f%%\n";
    std::fprintf(out, fmtHead, "stage", "mean", "min", "p50", "p99", "max", "%frame");
    std::fprintf(out, "------------------ ... ----\n");     // separator line
    for (const auto& r : stageRows)
        std::fprintf(out, fmtRow, r.name, r.s.mean/1e3, r.s.min/1e3, r.s.p50/1e3,
                     r.s.p99/1e3, r.s.max/1e3, 100.0 * r.s.mean / meanFrame);
    std::fprintf(out, "------------------ ... ----\n");
    for (const auto& r : aggRows)
        std::fprintf(out, fmtRow, r.name, r.s.mean/1e3, ..., 100.0 * r.s.mean / meanFrame);
```

- The `fmt` strings do all the alignment: `%-26s` = left-justified label in
  a 26-char column; `%9.1f` = a float in a 9-char column, one decimal;
  `%7.1f%%` = a float then a literal `%`.
- `/1e3` converts nanoseconds → microseconds for display.
- `100.0 * r.s.mean / meanFrame` = this stage's mean as a percentage of the
  mean frame time.
- Loop over the two row arrays, printing one formatted line each, with a
  separator between.

```cpp
    const Stat g = summarize(gpuTotal_);
    const double accountedGpu = summarize(force_).mean + summarize(barrier1_).mean +
                                summarize(integrate_).mean + summarize(barrier2_).mean +
                                summarize(copy_).mean;
    std::fprintf(out, "\nstage sum vs GPU total (mean): %.1fus vs %.1fus (%.2f%% accounted)\n",
                 accountedGpu / 1e3, g.mean / 1e3,
                 g.mean > 0 ? 100.0 * accountedGpu / g.mean : 0.0);
}
```

The **reconciliation check**. Sum the five stage means; compare to the GPU
total mean. They should match (each internal timestamp is the end of one
span and the start of the next, so the spans telescope). Printing
"100.00% accounted" is a self-check that the timestamp bookkeeping is
internally consistent — if it were, say, 94%, something would be recording
timestamps at the wrong points.

---

## 5. `means` (report.cpp:94-106)

```cpp
Report::Means Report::means() const {
    Means m;
    m.force      = summarize(force_).mean;
    m.barrier1   = summarize(barrier1_).mean;
    m.integrate  = summarize(integrate_).mean;
    m.barrier2   = summarize(barrier2_).mean;
    m.copy       = summarize(copy_).mean;
    m.gpuTotal   = summarize(gpuTotal_).mean;
    m.submitCall = summarize(submitCall_).mean;
    m.cpuWait    = summarize(cpuWait_).mean;
    m.frameTotal = summarize(frameTotal_).mean;
    return m;
}
```

Just the nine means, bundled. `--sweep` calls this once per particle count
and writes the numbers as one CSV row (`main.cpp`'s `runSweep`).

---

## 6. `writeCsv` (report.cpp:108-117)

```cpp
void Report::writeCsv(std::FILE* out) const {
    std::fprintf(out,
                 "step,force_ns,barrier1_ns,integrate_ns,barrier2_ns,copy_ns,"
                 "gpu_total_ns,submit_ns,cpu_wait_ns,frame_total_ns\n");
    for (size_t i = 0; i < frameTotal_.size(); ++i) {
        std::fprintf(out, "%zu,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f\n", i, force_[i],
                     barrier1_[i], integrate_[i], barrier2_[i], copy_[i], gpuTotal_[i],
                     submitCall_[i], cpuWait_[i], frameTotal_[i]);
    }
}
```

Header row, then one line per step: the step index and the nine raw
nanosecond values (`%.0f` = no decimals). This is the file the Python
plotting scripts read, and it's what lets you inspect the *distribution* of
each stage, not just its summary.

`out` is a `std::FILE*` the caller opened with `std::fopen` and will
`std::fclose` (`main.cpp`'s `runReport`).

---

## Explain it in one sentence

> `Report` stores nine parallel vectors of per-step nanosecond values,
> `summarize` sorts a copy of one vector and returns mean/min/p50/p99/max
> (percentile = element at `p*(n-1)` rounded), `printTable` lays that out as
> an aligned table with a "% of frame" column and a stage-sum-vs-GPU-total
> reconciliation line, and `writeCsv` dumps the raw rows for the plotting
> scripts.

## Questions you should be able to answer

1. Why nine parallel vectors instead of one vector of a struct?
2. Why does `summarize` take its argument by value?
3. How is the p99 computed, and why report p50 and p99 alongside the mean?
4. What is the "stage sum vs GPU total ... % accounted" line checking, and
   why should it be ~100%?
5. What does the "% frame" column mean and how is it computed?
6. Who calls `means()` and why does that caller not want the full table?
