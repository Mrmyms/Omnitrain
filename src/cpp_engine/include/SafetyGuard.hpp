#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

namespace omni {

/**
 * Safety Guard (Native C++ Port)
 * 
 * Implements Tier 1 Hardware Failsafe logic.
 * Checks raw sensor readings against hard limits with microsecond latency.
 */
class SafetyGuard {
public:
    struct Constraint {
        std::string sensor_id;
        float min_val;
        float max_val;
    };

    SafetyGuard() = default;

    void add_constraint(const std::string& id, float min, float max) {
        constraints_.push_back({id, min, max});
    }

    /**
     * Returns true if ALL constraints are satisfied.
     * If any violation is found, it prints a warning and returns false.
     */
    bool check(const std::string& sensor_id, float value) {
        for (const auto& c : constraints_) {
            if (c.sensor_id == sensor_id) {
                if (value < c.min_val || value > c.max_val) {
                    std::cerr << "⚠️  SAFETY VIOLATION [" << sensor_id << "]: " 
                              << value << " is outside [" << c.min_val << ", " << c.max_val << "]" << std::endl;
                    return false;
                }
            }
        }
        return true;
    }

    void clear() {
        constraints_.clear();
    }

private:
    std::vector<Constraint> constraints_;
};

} // namespace omni
