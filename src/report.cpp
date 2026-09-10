#include "report.h"

#include <algorithm>
#include <cmath>

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

Report::Stat Report::summarize(std::vector<double> v) {
    Stat s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (double x : v) sum += x;
    s.mean = sum / static_cast<double>(v.size());
    s.min = v.front();
    s.max = v.back();
    auto pct = [&](double p) {
        double idx = p * static_cast<double>(v.size() - 1);
        return v[static_cast<size_t>(std::lround(idx))];
    };
    s.p50 = pct(0.50);
    s.p99 = pct(0.99);
    return s;
}

void Report::printTable(std::FILE* out, const std::string& header) const {
    if (frameTotal_.empty()) {
        std::fprintf(out, "no samples\n");
        return;
    }

    const Stat frame = summarize(frameTotal_);
    const double meanFrame = frame.mean > 0 ? frame.mean : 1.0;

    struct Row {
        const char* name;
        Stat s;
    };
    const Row stageRows[] = {
        {"force dispatch", summarize(force_)},
        {"barrier 1 (accel w->r)", summarize(barrier1_)},
        {"integrate dispatch", summarize(integrate_)},
        {"barrier 2 (part. w->xfer)", summarize(barrier2_)},
        {"readback copy", summarize(copy_)},
    };
    const Row aggRows[] = {
        {"GPU total", summarize(gpuTotal_)},
        {"submit call (CPU)", summarize(submitCall_)},
        {"fence wait (CPU)", summarize(cpuWait_)},
        {"frame total", frame},
    };

    std::fprintf(out, "\n%s\n", header.c_str());
    if (anyGpuInvalid_)
        std::fprintf(out, "(warning: some frames had no GPU timestamps)\n");
    std::fprintf(out, "samples: %zu frames\n\n", frameTotal_.size());

    const char* fmtHead = "%-26s %10s %10s %10s %10s %10s %8s\n";
    const char* fmtRow = "%-26s %9.1fus %9.1fus %9.1fus %9.1fus %9.1fus %7.1f%%\n";
    std::fprintf(out, fmtHead, "stage", "mean", "min", "p50", "p99", "max", "%frame");
    std::fprintf(out,
                 "-------------------------- ---------- ---------- ---------- ---------- "
                 "---------- --------\n");
    for (const auto& r : stageRows)
        std::fprintf(out, fmtRow, r.name, r.s.mean / 1e3, r.s.min / 1e3, r.s.p50 / 1e3,
                     r.s.p99 / 1e3, r.s.max / 1e3, 100.0 * r.s.mean / meanFrame);
    std::fprintf(out,
                 "-------------------------- ---------- ---------- ---------- ---------- "
                 "---------- --------\n");
    for (const auto& r : aggRows)
        std::fprintf(out, fmtRow, r.name, r.s.mean / 1e3, r.s.min / 1e3, r.s.p50 / 1e3,
                     r.s.p99 / 1e3, r.s.max / 1e3, 100.0 * r.s.mean / meanFrame);

    const Stat g = summarize(gpuTotal_);
    const double accountedGpu = summarize(force_).mean + summarize(barrier1_).mean +
                                summarize(integrate_).mean + summarize(barrier2_).mean +
                                summarize(copy_).mean;
    std::fprintf(out, "\nstage sum vs GPU total (mean): %.1fus vs %.1fus (%.2f%% accounted)\n",
                 accountedGpu / 1e3, g.mean / 1e3,
                 g.mean > 0 ? 100.0 * accountedGpu / g.mean : 0.0);
}

Report::Means Report::means() const {
    Means m;
    m.force = summarize(force_).mean;
    m.barrier1 = summarize(barrier1_).mean;
    m.integrate = summarize(integrate_).mean;
    m.barrier2 = summarize(barrier2_).mean;
    m.copy = summarize(copy_).mean;
    m.gpuTotal = summarize(gpuTotal_).mean;
    m.submitCall = summarize(submitCall_).mean;
    m.cpuWait = summarize(cpuWait_).mean;
    m.frameTotal = summarize(frameTotal_).mean;
    return m;
}

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
