#include "vision/seguimiento_tiros.h"

#include "configuracion.h"

#include <algorithm>

int SeguimientoTiros::actualizar(int pinosEnPie, bool hayBola, bool hayBarra, int64_t ahoraMs,
                                 int pinosRack, int pinosAhora, int64_t msUltimoPino,
                                 int maxEnPie) {
    if (pinosEnPie < 0) return -1;      // sin informacion de pinos
    if (maxEnPie < 0 || maxEnPie > 10) maxEnPie = 10;

    // ------------------------------------------------------------------
    // Conteo del rack y pinos que siguen en pie
    // ------------------------------------------------------------------
    int rack = (pinosRack >= 0) ? pinosRack : pinosEnPie;
    rack = std::min(rack, maxEnPie);
    int ahoraCuenta = (pinosAhora >= 0) ? pinosAhora : rack;
    ahoraCuenta = std::min(ahoraCuenta, maxEnPie);

    // Ultimo conteo medido SIN barra: es la referencia fiable de lo que
    // queda en pie justo antes de que la barredora tape los pinos.
    // Mientras la barra esta bajada no se actualiza: cuando parpadea, la
    // barra tapa los pinos, el detector deja de verlos y el conteo caeria
    // a 0 aunque sigan de pie. Si los pinos se ven, el conteo si es fiable
    // (una deteccion suelta de la barra no debe congelarlo).
    const bool barraReciente = (ultimaBarraMs_ > 0) &&
                               (ahoraMs - ultimaBarraMs_ <= MS_CONGELAR_CONTEO);
    const bool pinosVisiblesAhora = (msUltimoPino >= 0) && (msUltimoPino <= MS_CAIDA);
    if (!hayBarra && (!barraReciente || pinosVisiblesAhora)) {
        if (ultimoSinBarra_ != ahoraCuenta) {
            ultimoSinBarra_ = ahoraCuenta;
            ultimoCambioSinBarraMs_ = ahoraMs;
        }
    }

    // Rack en pie: el maximo visto desde el ultimo barrido. Tras registrar
    // un tiro se ignora un momento el rack para no recontar los pinos que
    // acaban de caer (siguen en la ventana de MS_RACK).
    if (ahoraMs >= ignorarRackHastaMs_ && rack > pinosAntes_) pinosAntes_ = rack;

    // La bola se procesa tambien con la barra a la vista: hay
    // instalaciones donde la barra queda aparcada delante de los pinos.
    if (hayBola) {
        if (pinosAntes_ > 0) {
            bolaVista_ = true;
            tiroEnCurso_ = true;
        }
        ultimaBolaMs_ = ahoraMs;
    }
    const bool bolaPasada = bolaVista_ && (ahoraMs - ultimaBolaMs_ >= MS_POST_BOLA);

    if (hayBarra && !barraPresente_) barraApareceMs_ = ahoraMs;
    barraPresente_ = hayBarra;

    // Rearme del cierre: la barra se fue, pusieron un rack nuevo o paso
    // demasiado tiempo. Asi una barra detectada de forma permanente no
    // bloquea los tiros siguientes.
    if (enBarrido_) {
        const bool barraFuera = !hayBarra && (ahoraMs - ultimaBarraMs_ > MS_FIN_BARRIDO);
        const bool rackNuevo = (rack >= countAlRegistrar_ + 2) &&
                               (ahoraMs - registroMs_ >= MS_REARME_RACK);
        const bool timeout = (ahoraMs - registroMs_) >= MS_MAX_BARRIDO;
        if (barraFuera || rackNuevo || timeout) {
            enBarrido_ = false;
            cuadrosBarra_ = 0;
        }
    }

    // ------------------------------------------------------------------
    // La barra barredora confirma que el tiro termino. Se toma una foto
    // del rack y de lo que queda; el cierre se resuelve aunque la barra
    // desaparezca antes (antes solo se cerraba con la barra a la vista).
    // ------------------------------------------------------------------
    if (hayBarra) {
        ultimaBarraMs_ = ahoraMs;
        if (!enBarrido_ && !pendiente_) {
            if (cuadrosBarra_ == 0 || (ahoraMs - primeraBarraMs_) > MS_VENTANA_BARRA) {
                primeraBarraMs_ = ahoraMs;
                cuadrosBarra_ = 0;
            }
            // En la pista real la barra a veces solo se detecta en cuadros
            // sueltos al bajar. Si los pinos ya no se ven y la cuenta lleva
            // quieta, un solo cuadro es evidencia suficiente de barrido.
            // msUltimoPino ya es una duracion (ms desde la ultima vista).
            const bool sinPinosVisibles = (msUltimoPino >= 0) &&
                                          (msUltimoPino > MS_CAIDA);
            const bool cuentaQuieta =
                (ahoraMs - ultimoCambioSinBarraMs_) >= MS_ESTABLE_CAIDA;
            const bool cierreRapido =
                tiroEnCurso_ && sinPinosVisibles && cuentaQuieta;
            const int necesarios = cierreRapido ? 1 : CUADROS_BARRA;
            if (++cuadrosBarra_ >= necesarios) {
                cuadrosBarra_ = 0;
                if (pinosAntes_ <= 0) {
                    aviso("Barrido sin tiro: sin rack validado", ahoraMs);
                } else if (ahoraMs >= enfriamientoMs_) {
                    pendiente_ = true;
                    pendienteDesdeMs_ = ahoraMs;
                    antesBarrido_ = pinosAntes_;
                    // Foto de lo que queda: el ultimo conteo medido sin
                    // barra, antes de que la barredora tape los pinos.
                    despuesBarrido_ = (ultimoSinBarra_ >= 0)
                                        ? ultimoSinBarra_
                                        : ahoraCuenta;
                    despuesBarrido_ = std::min(despuesBarrido_, maxEnPie);
                    permiteSoloBola_ =
                        (ahoraMs - barraApareceMs_ <= MS_VENTANA_BARRA) || bolaPasada;
                }
            }
        }
    }

    // Cierre pendiente: se espera a que el conteo quede quieto (la barra
    // ya tapo los pinos y la foto esta tomada).
    if (pendiente_) {
        const int64_t quieto = ahoraMs - ultimoCambioSinBarraMs_;
        const int64_t espera = ahoraMs - pendienteDesdeMs_;
        if (quieto >= MS_ESTABLE_CAIDA || espera >= MS_MAX_PENDIENTE) {
            pendiente_ = false;

            // Foto tomada al confirmar el barrido: ultimo conteo medido
            // sin barra. No se refresca despues porque el re-rack de la
            // maquina volveria a poner 10 pinos y el tiro saldria 0.
            int despues = despuesBarrido_;
            despues = std::min(despues, maxEnPie);
            despues = std::max(0, std::min(despues, antesBarrido_));
            const int caidos = std::min(std::max(antesBarrido_ - despues, 0), 10);

            // Se registra si la bola inicio el tiro (tiroEnCurso_) y cayeron
            // pinos o la bola salio de escena (canaleta = 0). Sin bola no se
            // registra: cuando la maquina recoloca pinos la barra los tapa y
            // el conteo cae a 0, lo que se confundiria con un tiro (falso).
            const bool registra = (antesBarrido_ > 0) && tiroEnCurso_ &&
                                  (caidos > 0 || (bolaVista_ && permiteSoloBola_));

            if (registra) {
                enBarrido_ = true;
                registroMs_ = ahoraMs;
                countAlRegistrar_ = despues;
                ignorarRackHastaMs_ = ahoraMs + MS_RACK;

                cierreTexto_ = "Barrido: rack=" + std::to_string(antesBarrido_) +
                               " enPie=" + std::to_string(despues) +
                               " caidos=" + std::to_string(caidos) +
                               " bola=" + (bolaVista_ ? "si" : "no") +
                               " -> registra";
                cierreNuevo_ = true;

                tiroEnCurso_ = false;
                bolaVista_ = false;
                pinosAntes_ = despues;      // el rack nuevo se mide desde aqui
                ultimoSinBarra_ = despues;
                ultimoCambioSinBarraMs_ = ahoraMs;
                return caidos;
            }

            // Barra a la vista sin tiro (p. ej. esperando al jugador): no
            // cierra nada. Tras un respiro puede volver a confirmar.
            enfriamientoMs_ = ahoraMs + MS_ENFRIAMIENTO;
            aviso("Barrido sin tiro: rack=" + std::to_string(antesBarrido_) +
                  " enPie=" + std::to_string(despues) +
                  " bola=" + (bolaVista_ ? "si" : "no"), ahoraMs);
        }
    }

    // El cambio de tiro/frame SOLO lo confirma la barra barredora.
    return -1;
}

bool SeguimientoTiros::tomarUltimoCierre(std::string& texto) {
    if (!cierreNuevo_) return false;
    texto = cierreTexto_;
    cierreNuevo_ = false;
    return true;
}

void SeguimientoTiros::aviso(const std::string& texto, int64_t ahoraMs) {
    if (ahoraMs - ultimoAvisoMs_ < 3000) return;
    ultimoAvisoMs_ = ahoraMs;
    cierreTexto_ = texto;
    cierreNuevo_ = true;
}
