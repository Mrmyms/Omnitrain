#ifndef OMNI_SHIELD_HPP
#define OMNI_SHIELD_HPP

#include <vector>
#include <cstdint>

// OmniShield: Deterministic Tier 1 Safety System (Failsafe)
// Intercepts the outputs of the neural engine and enforces strict hardware rules.
class OmniShieldGuard {
public:
    OmniShieldGuard(uint32_t num_sensors, uint32_t action_dim) 
        : num_sensors_(num_sensors), action_dim_(action_dim) {
        
        // Initialize limits to "infinity" by default
        hw_min_.assign(num_sensors, -1e9f);
        hw_max_.assign(num_sensors, 1e9f);
        emergency_action_.assign(action_dim, 0.0f);
    }

    // Configure hard limits from the configuration array
    void SetHardwareLimits(const std::vector<float>& mins, const std::vector<float>& maxs) {
        if (mins.size() == num_sensors_ && maxs.size() == num_sensors_) {
            hw_min_ = mins;
            hw_max_ = maxs;
        }
    }

    // Configure the safe action (e.g., Inject 0.0 U of insulin or halt motors)
    void SetEmergencyAction(const std::vector<float>& safe_action) {
        if (safe_action.size() == action_dim_) {
            emergency_action_ = safe_action;
        }
    }

    // Audits the proposed action from the neural network
    // Returns the modified (or intact) action and a boolean indicating if intervention occurred
    std::vector<float> Enforce(const float* current_sensors, const std::vector<float>& proposed_action, bool& intervened) {
        intervened = false;
        
        // Verify hardware limits (Tier 1)
        for (uint32_t i = 0; i < num_sensors_; ++i) {
            if (current_sensors[i] < hw_min_[i] || current_sensors[i] > hw_max_[i]) {
                intervened = true;
                break;
            }
        }

        if (intervened) {
            // Safety breach detected: Override network and return safe fallback
            return emergency_action_;
        }

        // Apply action clipping based on output range (Tier 2 Convex Barrier projection proxy)
        // Here we could implement the specific math for solving ICNN projection if required
        // For standard Edge deployment, we clip between -1.0 and 1.0 (motor RPM / dose)
        std::vector<float> safe_out = proposed_action;
        for (uint32_t i = 0; i < action_dim_; ++i) {
            if (safe_out[i] > 1.0f) safe_out[i] = 1.0f;
            if (safe_out[i] < -1.0f) safe_out[i] = -1.0f;
        }

        return safe_out;
    }

private:
    uint32_t num_sensors_;
    uint32_t action_dim_;
    std::vector<float> hw_min_;
    std::vector<float> hw_max_;
    std::vector<float> emergency_action_;
};

#endif // OMNI_SHIELD_HPP
