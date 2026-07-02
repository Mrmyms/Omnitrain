#include "OmniShield.hpp"

// En sistemas embebidos críticos, la implementación de OmniShield debe mantenerse
// lo más reducida posible en el archivo .cpp para asegurar un inlining eficiente 
// si el compilador así lo decide.

// Por ahora, toda la lógica de Tier 1 está encapsulada en OmniShield.hpp
// para maximizar rendimiento. Aquí podríamos implementar Tier 2 (Proyección CBF)
// si el microcontrolador tiene ciclos libres para resolver el QP (Quadratic Program).
