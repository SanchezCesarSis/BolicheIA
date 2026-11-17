#pragma once

#include <cstdint>
#include <string>

// ------------------------------------------------------------------
// Seguimiento de los pinos: deduce cuantos pinos cayeron en cada tiro.
// El unico cierre de tiro es la barra barredora.
// ------------------------------------------------------------------
class SeguimientoTiros {
public:
    // La bola tiene que haber salido de escena este tiempo para cerrar un
    // tiro por canaleta cuando la barra ya estaba a la vista (aparcada).
    static constexpr int64_t MS_POST_BOLA = 1200;
    static constexpr int64_t MS_ENFRIAMIENTO = 900;    // respiro tras un barrido sin tiro
    static constexpr int64_t MS_MAX_PENDIENTE = 2500;  // maxima espera para cerrar
    static constexpr int64_t MS_REARME_RACK = 800;
    static constexpr int64_t MS_MAX_BARRIDO = 8000;

    // Actualiza el estado con el cuadro actual.
    // Devuelve los pinos caidos si se acaba de completar un tiro, o -1 si no hay tiro nuevo.
    //
    // pinosEnPie: conteo normal del validador (mantiene 1.3 s a los pinos caidos).
    // pinosRack: pinos vistos en los ultimos MS_RACK; establece el rack en pie.
    // pinosAhora: pinos vistos en los ultimos MS_CAIDA; es lo que sigue en pie.
    // msUltimoPino: ms desde la ultima vez que se vio un pino (-1 = sin dato).
    // maxEnPie: pinos que deberian seguir en pie segun el marcador. Es un tope
    // para el rack: las detecciones fantasma de pinos ya caidos no deben
    // inflar el conteo de la siguiente tirada.
    int actualizar(int pinosEnPie, bool hayBola, bool hayBarra, int64_t ahoraMs,
                   int pinosRack = -1, int pinosAhora = -1, int64_t msUltimoPino = -1,
                   int maxEnPie = 10);

    bool tiroEnCurso() const { return tiroEnCurso_; }
    bool enBarrido() const { return enBarrido_; }
    bool pendiente() const { return pendiente_; }
    int rack() const { return pinosAntes_; }
    int ultimoSinBarra() const { return ultimoSinBarra_; }
    int antesPendiente() const { return antesBarrido_; }
    int despuesPendiente() const { return despuesBarrido_; }

    // Diagnostico: entrega el detalle del ultimo barrido confirmado (una vez).
    bool tomarUltimoCierre(std::string& texto);

private:
    void aviso(const std::string& texto, int64_t ahoraMs);

    bool tiroEnCurso_ = false;
    bool enBarrido_ = false;          // la barra esta bajada, tiro ya cerrado
    bool bolaVista_ = false;          // se vio la bola desde el ultimo barrido
    int pinosAntes_ = 0;              // rack en pie antes del barrido
    int ultimoSinBarra_ = -1;         // ultimo conteo medido sin la barra
    int cuadrosBarra_ = 0;            // detecciones de barra en la ventana
    int64_t ultimoCambioSinBarraMs_ = 0; // ultimo cambio de conteo sin barra
    int64_t ultimoAvisoMs_ = 0;       // ultimo aviso de barrido sin tiro
    int64_t primeraBarraMs_ = 0;      // inicio de la ventana de detecciones
    int64_t ultimaBarraMs_ = 0;       // ultimo cuadro con la barra en escena
    int64_t ultimaBolaMs_ = 0;        // ultimo cuadro con la bola en escena
    bool barraPresente_ = false;      // la barra ya estaba antes de este cuadro
    int64_t barraApareceMs_ = 0;      // cuando aparecio la barra
    bool pendiente_ = false;          // barrido confirmado, esperando cerrar
    int64_t pendienteDesdeMs_ = 0;
    int antesBarrido_ = 0;
    int despuesBarrido_ = 0;
    bool permiteSoloBola_ = false;    // la bola sola puede cerrar (canaleta)
    int64_t enfriamientoMs_ = 0;      // no confirmar otro barrido hasta aqui
    int64_t registroMs_ = 0;          // cuando se registro el ultimo tiro
    int countAlRegistrar_ = 0;        // pinos que quedaron tras ese tiro
    int64_t ignorarRackHastaMs_ = 0;  // no recontar pinos viejos como rack
    bool cierreNuevo_ = false;
    std::string cierreTexto_;
};
