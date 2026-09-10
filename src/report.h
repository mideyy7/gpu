#pragma once

#include <cstdio>
#include <string>
#include <vector>

#include "simulation.h"

// Accumulates StepTimings across many frames and renders a per-stage
// breakdown: mean / min / p50 / p99 / max per stage, and each stage's
// share of mean frame time. Also dumps raw per-step rows as CSV.
class Report {
   public:
    void addSample(const StepTimings& t);

    void printTable(std::FILE* out, const std::string& header) const;
    void writeCsv(std::FILE* out) const;

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
