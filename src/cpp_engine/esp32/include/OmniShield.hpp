#ifndef OMNI_SHIELD_HPP
#define OMNI_SHIELD_HPP

#include <vector>
#include <cstdint>

// OmniShield: Sistema Determinista de Seguridad Tier 1 (Failsafe)
// Intercepta las salidas del motor neuronal y aplica reglas estrictas de hardware.
class OmniShieldGuard {
public:
    OmniShieldGuard(uint32_t num_sensors, uint32_t action_dim) 
        : num_sensors_(num_sensors), action_dim_(action_dim) {
        
        // Inicializar límites en "infinito" por defecto
        hw_min_.assign(num_sensors, -1e9f);
        hw_max_.assign(num_sensors, 1e9f);
        emergency_action_.assign(action_dim, 0.0f);
    }

    // Configurar límites duros desde el array de configuración
    void SetHardwareLimits(const std::vector<float>& mins, const std::vector<float>& maxs) {
        if (mins.size() == num_sensors_ && maxs.size() == num_sensors_) {
            hw_min_ = mins;
            hw_max_ = maxs;
        }
    }

    // Configurar la acción segura (Ej. Inyectar 0.0 U de insulina o detener motores)
    void SetEmergencyAction(const std::vector<float>& safe_action) {
        if (safe_action.size() == action_dim_) {
            emergency_action_ = safe_action;
        }
    }

    // Audita la acción propuesta por la red neuronal
    // Devuelve la acción modificada (o intacta) y un booleano indicando si hubo intervención
    std::vector<float> Enforce(const float* current_sensors, const std::vector<float>& proposed_action, bool& intervened) {
        intervened = false;
        
        // Verificar límites de hardware (Tier 1)
        for (uint32_t i = 0; i < num_sensors_; ++i) {
            if (current_sensors[i] < hw_min_[i] || current_sensors[i] > hw_max_[i]) {
                intervened = true;
                break;
            }
        }

        if (intervened) {
            // Si el paciente o el hardware están en riesgo, ignorar a la IA
            return emergency_action_;
        }

        // Si es seguro, permitir el paso
        return proposed_action;
    }

private:
    uint32_t num_sensors_;
    uint32_t action_dim_;
    
    std::vector<float> hw_min_;
    std::vector<float> hw_max_;
    std::vector<float> emergency_action_;
};

#endif // OMNI_SHIELD_HPP
