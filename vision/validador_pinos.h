#pragma once

#include "configuracion.h"

#include <opencv2/core.hpp>
#include <cstdint>
#include <vector>

// ------------------------------------------------------------------
// Validacion temporal de pinos
// Un pino solo se cuenta si la IA lo detecta de forma continua durante
// 0.5 segundos. Asi se evita contar pinos de mas por falsos positivos.
// El conteo se limita a 10 pinos.
// ------------------------------------------------------------------
class ValidadorPinos {
public:
    static constexpr int MAX_PINOS = 10;
    static constexpr int64_t MS_VALIDACION = 500;    // 0.5 s para dar el pino por bueno
    static constexpr int64_t MS_PERDIDA = 1300;      // sin verse 1.3 s -> se descarta (cayo)
    static constexpr int64_t MS_PERDIDA_CANDIDATO = 1200; // candidato sin confirmar
    static constexpr int64_t MS_HUECO = 700;         // hueco maximo para seguir validando

    struct Pino {
        cv::Rect box;
        int64_t primeraVezMs = 0;
        int64_t ultimaVezMs = 0;
        int64_t ultimaVistaIAms = 0; // ultima vez visto de verdad (IA o franjas)
        int64_t vistoMs = 0;        // tiempo real que lleva detectado
        bool confirmado = false;    // ya valido los 0.5 s: cuenta como pino
        float progreso = 0.0f;      // 0..1 de la validacion
    };

    // detecciones: cajas de la clase "Bolo" del cuadro actual.
    // franjas: cajas de las franjas rojas del cuello (bajo luz azul).
    // pausar = true durante el barrido: la barra tapa los pinos y no se
    // deben descartar por no verlos.
    int actualizar(const std::vector<cv::Rect>& detecciones, int64_t ahoraMs, bool pausar = false,
                   const std::vector<cv::Rect>& franjas = {});

    // Pinos validados (0.5 s) que siguen en pie
    int enPie() const;

    // Pinos que se estan validando (aun no llegan a 0.5 s)
    int validando() const;

    // Pinos confirmados vistos hace muy poco (sin la tolerancia de 1.3 s).
    // Es la medida fiable del rack en el momento en que entra la barra: los
    // pinos caidos dejan de verse y no se cuentan, aunque sigan en la lista.
    int enPieRecientes(int64_t ahoraMs, int64_t ventanaMs = MS_CAIDA) const;

    const std::vector<Pino>& pines() const { return pines_; }

    // Instante de la ultima vez que se vio cualquier pino (IA o franjas).
    // 0 = todavia no se ha visto ninguno.
    // Se guarda aparte de la lista porque, si la barra tapa los pinos mas de
    // la tolerancia, la lista queda vacia y aun asi hay que saber cuando se
    // vieron por ultima vez (lo usa el cierre rapido del barrido).
    int64_t ultimaVistaMs() const;

private:
    static cv::Point2f centro(const cv::Rect& r);
    static double iou(const cv::Rect& a, const cv::Rect& b);
    static bool mismaCaja(const cv::Rect& a, const cv::Rect& b);

    // Suma el tiempo real transcurrido desde la ultima vista. Si el hueco es
    // grande (el pino desaparecio) se reinicia: no se valida un pino sumando
    // apariciones sueltas separadas por segundos.
    static void sumarPresencia(Pino& p, int64_t ahoraMs);

    std::vector<Pino> pines_;
    int64_t ultimaVistaGlobalMs_ = 0;   // ultima vez que se vio cualquier pino
};
