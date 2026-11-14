//
// Created by sait1 on 19/10/2026.
//

#include "deteccion_IA.h"
#include "marcador.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#define NOMINMAX   // evita que windows.h defina las macros min/max
#include <Windows.h>
#pragma comment(lib, "user32.lib")
#endif

const float CONFIDENCE_THRESHOLD = 0.5f;
const float NMS_THRESHOLD = 0.4f;

// Con la GPU dedicada la inferencia tarda ~10 ms, asi que se puede detectar en cada cuadro
const int PROCESAR_CADA_N_CUADROS = 1;
const int CLASES = 3; // 0 = Bola, 1 = Bolo (pino), 2 = Barra barredora

// Umbral de confianza por clase
const float UMBRAL_BOLA = 0.45f;
const float UMBRAL_BOLO = 0.45f;
const float UMBRAL_BARRA = 0.45f;

// Filtro permanente: hace que la camara y todo el procesamiento operen siempre en blanco y negro
const bool CAMARA_SIEMPRE_BLANCO_Y_NEGRO = true;

// Seguimiento del tiro (para saber cuando termina y cuantos pinos cayeron)
const int CUADROS_BARRA = 3;       // detecciones de barra para confirmar el barrido
const int64_t MS_VENTANA_BARRA = 1200; // margen para juntar esas detecciones (tolera parpadeos)
const int64_t MS_FIN_BARRIDO = 600;    // sin ver la barra este tiempo: el barrido termino
// Mientras la barra esta bajada (aunque el detector solo la vea a ratos) no se
// actualiza el conteo de pinos: la barredora los tapa y el conteo caeria a 0.
const int64_t MS_CONGELAR_CONTEO = 2000;
const int64_t MS_ESTABLE_CAIDA = 600;  // pinos quietos este tiempo antes de cerrar el tiro

// Tiempos del registro (independientes de la tolerancia visual del validador).
// Un pino se da por caido para el marcador cuando deja de verse este tiempo:
// es corto a proposito, porque la barra barredora puede bajar enseguida.
// Medir el rack en pie con la tolerancia de 3 s hacia que el tiro saliera 0.
const int64_t MS_CAIDA = 450;
// Ventana para medir el rack en pie (los pinos estan quietos, se toleran
// parpadeos de la deteccion). El rack solo crece, nunca se resta.
const int64_t MS_RACK = 1200;

// La ruta la define CMakeLists.txt para no depender del directorio de trabajo
#ifndef BOLICHE_MODEL_PATH
#define BOLICHE_MODEL_PATH "bolos.onnx"
#endif

// Carpeta con las DLL de CUDA (cuBLAS / cuDNN) que necesita el Execution Provider
#ifndef BOLICHE_CUDA_BIN_PATH
#define BOLICHE_CUDA_BIN_PATH "C:/ort/cuda/bin"
#endif

// Convierte una ruta ASCII a UTF-16 para la API de Windows
static std::wstring aWide(const std::string& ruta) {
    return std::wstring(ruta.begin(), ruta.end());
}

// Una deteccion ya filtrada por NMS (se hace NMS por clase para que un pin
// no sea eliminado por la caja de la bola)
struct Deteccion {
    cv::Rect box;
    float confianza;
    int clase;
};

// ------------------------------------------------------------------
// Franjas del cuello del pino.
// Detecta las franjas tanto en color (con luz azul o normal por R - G)
// como en blanco y negro (mediante filtro morfologico Black-Hat:
// las franjas son bandas oscuras sobre el fondo blanco del pino).
// Se devuelven las cajas de las franjas para usarlas como evidencia
// adicional en la validacion del pino.
// ------------------------------------------------------------------
static std::vector<cv::Rect> detectarFranjasRojas(const cv::Mat& bgr) {
    std::vector<cv::Rect> franjas;
    if (bgr.empty()) return franjas;

    std::vector<cv::Mat> canales;
    cv::split(bgr, canales);

    // Comprobar si la imagen es monocromatica / blanco y negro
    cv::Mat difRG;
    cv::absdiff(canales[2], canales[1], difRG);
    double maxDif = 0.0;
    cv::minMaxLoc(difRG, nullptr, &maxDif);
    bool esBlancoYNegro = (maxDif < 5.0);

    cv::Mat mascara;

    if (!esBlancoYNegro) {
        // --- Metodo por Color (R - G) ---
        cv::Mat rojo, diferencia, diferenciaUmbral;
        cv::threshold(canales[2], rojo, 130, 255, cv::THRESH_BINARY);
        cv::subtract(canales[2], canales[1], diferencia);        // R - G (saturado a 0)
        cv::threshold(diferencia, diferenciaUmbral, 18, 255, cv::THRESH_BINARY);
        cv::bitwise_and(rojo, diferenciaUmbral, mascara);
    } else {
        // --- Metodo para Blanco y Negro (Caso 2: Filtro Morfologico Black-Hat) ---
        // En escala de grises, el pino es blanco brillante y la franja roja es una banda
        // horizontal oscura en el cuello. Black-Hat (Close - Original) realza esas bandas oscuras.
        cv::Mat gris = canales[0];
        cv::Mat blackhat;
        cv::Mat elemFranja = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 3));
        cv::morphologyEx(gris, blackhat, cv::MORPH_BLACKHAT, elemFranja);

        // Mascara de brillo: asegurarse de que la franja este sobre el pino iluminado (brillo > 80)
        cv::Mat mascaraBrillo;
        cv::threshold(gris, mascaraBrillo, 80, 255, cv::THRESH_BINARY);

        cv::threshold(blackhat, mascara, 16, 255, cv::THRESH_BINARY);
        cv::bitwise_and(mascara, mascaraBrillo, mascara);
    }

    // Unir la franja horizontal y eliminar pixeles aislados
    cv::morphologyEx(mascara, mascara, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 1)));
    cv::morphologyEx(mascara, mascara, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 1)));

    std::vector<std::vector<cv::Point>> contornos;
    cv::findContours(mascara, contornos, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto& c : contornos) {
        cv::Rect r = cv::boundingRect(c);
        if (r.width < 5 || r.height < 1 || r.height > 24) continue;
        if (r.width < 1.3 * r.height) continue;      // una franja es alargada horizontal
        if (cv::contourArea(c) < 5.0) continue;
        franjas.push_back(r);
    }
    return franjas;
}

// ------------------------------------------------------------------
// Deteccion de la barra de barrido (por color o en blanco y negro).
// En color:
//   1) una linea magenta muy ancha (>= 20% del ancho de la imagen) y plana
//   2) el cuerpo azul saturado de la barra (ancho, plano y que no toca el foso)
// En blanco y negro:
//   3) banda oscura horizontal que cruza la pista
// ------------------------------------------------------------------
static bool detectarBarraPorColor(const cv::Mat& bgr, cv::Rect& barra) {
    barra = cv::Rect();
    if (bgr.empty()) return false;

    std::vector<cv::Mat> canales;
    cv::split(bgr, canales);

    cv::Mat difRG;
    cv::absdiff(canales[2], canales[1], difRG);
    double maxDif = 0.0;
    cv::minMaxLoc(difRG, nullptr, &maxDif);
    bool esBlancoYNegro = (maxDif < 5.0);

    std::vector<std::vector<cv::Point>> contornos;

    if (!esBlancoYNegro) {
        // --- 1) Rayas magenta/rojas de los bordes de la barra ---
        cv::Mat diferencia, mascara, rojo;
        cv::subtract(canales[2], canales[1], diferencia);      // R - G
        cv::threshold(diferencia, diferencia, 25, 255, cv::THRESH_BINARY);
        cv::threshold(canales[2], rojo, 90, 255, cv::THRESH_BINARY);
        cv::bitwise_and(diferencia, rojo, mascara);

        // La raya es larga: se cierra en horizontal
        cv::morphologyEx(mascara, mascara, cv::MORPH_CLOSE,
                         cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 1)));

        cv::findContours(mascara, contornos, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        const int minAncho = (int)(0.20 * bgr.cols);
        for (const auto& c : contornos) {
            cv::Rect r = cv::boundingRect(c);
            if (r.width < minAncho) continue;   // las franjas del cuello son estrechas
            if (r.height < 2) continue;
            if (r.width < 3 * r.height) continue;   // tiene que ser una linea horizontal
            barra = r;
            return true;
        }

        // --- 2) Cuerpo azul oscuro de la barra ---
        cv::Mat masB, masB2, azul;
        cv::subtract(canales[0], canales[1], masB);            // B - G
        cv::subtract(canales[0], canales[2], masB2);           // B - R
        cv::threshold(masB, masB, 35, 255, cv::THRESH_BINARY);
        cv::threshold(masB2, masB2, 35, 255, cv::THRESH_BINARY);
        cv::bitwise_and(masB, masB2, azul);
        cv::morphologyEx(azul, azul, cv::MORPH_CLOSE,
                         cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 3)));
        cv::findContours(azul, contornos, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (const auto& c : contornos) {
            cv::Rect r = cv::boundingRect(c);
            if (r.width < (int)(0.45 * bgr.cols)) continue;   // tiene que cruzar la pista
            if (r.height < 4 || r.height > (int)(0.30 * bgr.rows)) continue;
            if (r.width < 4 * r.height) continue;              // banda plana
            if (r.y < (int)(0.06 * bgr.rows)) continue;         // el foso (arriba) tambien es azul
            barra = r;
            return true;
        }
    } else {
        // --- 3) Barra en blanco y negro (banda oscura horizontal que cruza la pista) ---
        cv::Mat gris = canales[0];
        cv::Mat oscura;
        cv::threshold(gris, oscura, 70, 255, cv::THRESH_BINARY_INV);
        cv::morphologyEx(oscura, oscura, cv::MORPH_CLOSE,
                         cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 3)));
        cv::findContours(oscura, contornos, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (const auto& c : contornos) {
            cv::Rect r = cv::boundingRect(c);
            if (r.width < (int)(0.45 * bgr.cols)) continue;   // cruza la pista
            if (r.height < 4 || r.height > (int)(0.30 * bgr.rows)) continue;
            if (r.width < 3.5 * r.height) continue;           // banda plana
            if (r.y < (int)(0.06 * bgr.rows)) continue;        // no es el foso superior
            barra = r;
            return true;
        }
    }
    return false;
}

// Hay una franja dentro de la zona del cuello del pino (parte superior).
// Se usa cuando la IA no detecta el pino en ese cuadro pero las franjas
// rojas siguen brillando: el pino sigue en pie.
static bool franjaEnCuello(const cv::Rect& caja, const std::vector<cv::Rect>& franjas) {
    if (franjas.empty() || caja.width <= 0 || caja.height <= 0) return false;
    const int x0 = caja.x + (int)std::lround(0.10 * caja.width);
    const int x1 = caja.x + (int)std::lround(0.90 * caja.width);
    const int y0 = caja.y + (int)std::lround(0.02 * caja.height);
    const int y1 = caja.y + (int)std::lround(0.58 * caja.height);
    for (const cv::Rect& f : franjas) {
        const cv::Point centro(f.x + f.width / 2, f.y + f.height / 2);
        if (centro.x >= x0 && centro.x <= x1 && centro.y >= y0 && centro.y <= y1) return true;
    }
    return false;
}

// ------------------------------------------------------------------
// Validacion temporal de pinos
// Un pino solo se cuenta si la IA lo detecta de forma continua durante
// 1 segundo. Asi se evita contar pinos de mas por falsos positivos.
// El conteo se limita a 10 pinos.
// ------------------------------------------------------------------
class ValidadorPinos {
public:
    static const int MAX_PINOS = 10;
    static const int64_t MS_VALIDACION = 1000;   // 1 s para dar el pino por bueno
    static const int64_t MS_PERDIDA = 2000;      // sin verse 3 s -> se descarta (cayo)
    static const int64_t MS_PERDIDA_CANDIDATO = 1200; // candidato sin confirmar
    static const int64_t MS_HUECO = 700;         // hueco maximo para seguir validando

    struct Pino {
        cv::Rect box;
        int64_t primeraVezMs = 0;
        int64_t ultimaVezMs = 0;
        int64_t ultimaVistaIAms = 0; // ultima vez visto de verdad (IA o franjas)
        int64_t vistoMs = 0;        // tiempo real que lleva detectado
        bool confirmado = false;    // ya valido los 1 s: cuenta como pino
        float progreso = 0.0f;      // 0..1 de la validacion
    };

    // detecciones: cajas de la clase "Bolo" del cuadro actual.
    // franjas: cajas de las franjas rojas del cuello (bajo luz azul).
    // pausar = true durante el barrido: la barra tapa los pinos y no se
    // deben descartar por no verlos.
    int actualizar(const std::vector<cv::Rect>& detecciones, int64_t ahoraMs, bool pausar = false,
                   const std::vector<cv::Rect>& franjas = {}) {
        // Asociacion global: se empareja cada pino conocido con la deteccion
        // mas cercana (o con mas solape). Antes se exigia un radio muy corto
        // (0.6 * el lado menor) y con el temblor normal de la caja cada cuadro
        // se creaba un candidato nuevo: la validacion nunca llegaba a 1 s.
        struct Par { double puntaje; int pino; int det; };
        std::vector<Par> pares;
        for (size_t pi = 0; pi < pines_.size(); ++pi) {
            const Pino& p = pines_[pi];
            const double limite =
                std::max(20.0, 0.62 * (double)std::max(p.box.width, p.box.height));
            for (size_t di = 0; di < detecciones.size(); ++di) {
                const double dist = cv::norm(centro(p.box) - centro(detecciones[di]));
                const double solape = iou(p.box, detecciones[di]);
                if (solape >= 0.10 || dist <= limite)
                    pares.push_back({dist - 300.0 * solape, (int)pi, (int)di});
            }
        }
        std::sort(pares.begin(), pares.end(),
                  [](const Par& a, const Par& b) { return a.puntaje < b.puntaje; });

        std::vector<bool> pinoUsado(pines_.size(), false);
        std::vector<bool> detUsada(detecciones.size(), false);
        for (const Par& par : pares) {
            if (pinoUsado[par.pino] || detUsada[par.det]) continue;
            pinoUsado[par.pino] = true;
            detUsada[par.det] = true;
            Pino& p = pines_[par.pino];
            sumarPresencia(p, ahoraMs);
            p.box = detecciones[par.det];
            p.ultimaVistaIAms = ahoraMs;
        }

        for (size_t pi = 0; pi < pines_.size(); ++pi) {
            if (pinoUsado[pi]) continue;
            Pino& p = pines_[pi];
            if (pausar) {
                // La barra tapa los pinos ya validados: no se descartan.
                // A los candidatos sin confirmar no se les renueva la vista:
                // si la barra se queda aparcada, no bloquean la lista.
                if (p.confirmado) p.ultimaVezMs = ahoraMs;
            } else if (franjaEnCuello(p.box, franjas)) {
                // La IA no lo vio en este cuadro, pero las franjas rojas del
                // cuello siguen brillando en su sitio: el pino sigue en pie.
                sumarPresencia(p, ahoraMs);
                p.ultimaVistaIAms = ahoraMs;
            }
        }

        for (Pino& p : pines_)
            p.progreso = p.confirmado ? 1.0f
                        : std::min(1.0f, (float)p.vistoMs / (float)MS_VALIDACION);

        // Descartar los que llevan demasiado tiempo sin verse
        pines_.erase(std::remove_if(pines_.begin(), pines_.end(),
            [&](const Pino& p) {
                const int64_t sinVer = ahoraMs - p.ultimaVezMs;
                return sinVer > (p.confirmado ? MS_PERDIDA : MS_PERDIDA_CANDIDATO);
            }), pines_.end());

        // Candidatos nuevos
        for (size_t i = 0; i < detecciones.size(); i++) {
            if (detUsada[i]) continue;
            bool duplicado = false;
            for (const Pino& p : pines_) {
                if (mismaCaja(p.box, detecciones[i])) { duplicado = true; break; }
            }
            if (duplicado) continue;
            if ((int)pines_.size() >= MAX_PINOS) break;   // ya estan los 10
            Pino nuevo;
            nuevo.box = detecciones[i];
            nuevo.primeraVezMs = nuevo.ultimaVezMs = ahoraMs;
            nuevo.ultimaVistaIAms = ahoraMs;
            pines_.push_back(nuevo);
        }

        return enPie();
    }

    // Pinos validados (1 s) que siguen en pie
    int enPie() const {
        int n = 0;
        for (const Pino& p : pines_) if (p.confirmado) ++n;
        return std::min(n, MAX_PINOS);
    }

    // Pinos que se estan validando (aun no llegan a 1 s)
    int validando() const {
        int n = 0;
        for (const Pino& p : pines_) if (!p.confirmado) ++n;
        return n;
    }

    // Pinos confirmados vistos hace muy poco (sin la tolerancia de 3 s).
    // Es la medida fiable del rack en el momento en que entra la barra: los
    // pinos caidos dejan de verse y no se cuentan, aunque sigan en la lista.
    int enPieRecientes(int64_t ahoraMs, int64_t ventanaMs = MS_CAIDA) const {
        int n = 0;
        for (const Pino& p : pines_)
            if (p.confirmado && p.ultimaVistaIAms > 0 &&
                (ahoraMs - p.ultimaVistaIAms) <= ventanaMs) ++n;
        return std::min(n, MAX_PINOS);
    }

    const std::vector<Pino>& pines() const { return pines_; }

    // Instante de la ultima vez que se vio cualquier pino (IA o franjas).
    // 0 = todavia no se ha visto ninguno.
    int64_t ultimaVistaMs() const {
        int64_t t = 0;
        for (const Pino& p : pines_) t = std::max(t, p.ultimaVistaIAms);
        return t;
    }

private:
    static cv::Point2f centro(const cv::Rect& r) {
        return cv::Point2f(r.x + r.width * 0.5f, r.y + r.height * 0.5f);
    }

    static double iou(const cv::Rect& a, const cv::Rect& b) {
        const int x1 = std::max(a.x, b.x);
        const int y1 = std::max(a.y, b.y);
        const int x2 = std::min(a.x + a.width, b.x + b.width);
        const int y2 = std::min(a.y + a.height, b.y + b.height);
        if (x2 <= x1 || y2 <= y1) return 0.0;
        const double inter = (double)(x2 - x1) * (double)(y2 - y1);
        const double uni = (double)a.width * a.height + (double)b.width * b.height - inter;
        return uni > 0.0 ? inter / uni : 0.0;
    }

    static bool mismaCaja(const cv::Rect& a, const cv::Rect& b) {
        const double limite =
            std::max(14.0, 0.35 * (double)std::max(a.height, b.height));
        return iou(a, b) >= 0.25 || cv::norm(centro(a) - centro(b)) <= limite;
    }

    // Suma el tiempo real transcurrido desde la ultima vista. Si el hueco es
    // grande (el pino desaparecio) se reinicia: no se valida un pino sumando
    // apariciones sueltas separadas por segundos.
    static void sumarPresencia(Pino& p, int64_t ahoraMs) {
        const int64_t hueco = ahoraMs - p.ultimaVezMs;
        if (hueco > MS_HUECO) p.vistoMs = 0;
        else if (hueco > 0) p.vistoMs += hueco;
        p.ultimaVezMs = ahoraMs;
        if (!p.confirmado && p.vistoMs >= MS_VALIDACION) {
            p.confirmado = true;
            p.vistoMs = MS_VALIDACION;
        }
    }

    std::vector<Pino> pines_;
};

// ------------------------------------------------------------------
// Seguimiento de los pinos: deduce cuantos pinos cayeron en cada tiro
// ------------------------------------------------------------------
class SeguimientoTiros {
public:
    // La bola tiene que haber salido de escena este tiempo para cerrar un
    // tiro por canaleta cuando la barra ya estaba a la vista (aparcada).
    static const int64_t MS_POST_BOLA = 1200;
    static const int64_t MS_ENFRIAMIENTO = 900;    // respiro tras un barrido sin tiro
    static const int64_t MS_MAX_PENDIENTE = 2500;  // maxima espera para cerrar
    static const int64_t MS_REARME_RACK = 800;
    static const int64_t MS_MAX_BARRIDO = 8000;

    // Actualiza el estado con el cuadro actual.
    // Devuelve los pinos caidos si se acaba de completar un tiro, o -1 si no hay tiro nuevo.
    // El unico cierre de tiro es la barra barredora.
    //
    // pinosEnPie: conteo normal del validador (mantiene 3 s a los pinos caidos).
    // pinosRack: pinos vistos en los ultimos MS_RACK; establece el rack en pie.
    // pinosAhora: pinos vistos en los ultimos MS_CAIDA; es lo que sigue en pie.
    // msUltimoPino: ms desde la ultima vez que se vio un pino (-1 = sin dato).
    // maxEnPie: pinos que deberian seguir en pie segun el marcador. Es un tope
    // para el rack: las detecciones fantasma de pinos ya caidos no deben
    // inflar el conteo de la siguiente tirada.
    int actualizar(int pinosEnPie, bool hayBola, bool hayBarra, int64_t ahoraMs,
                   int pinosRack = -1, int pinosAhora = -1, int64_t msUltimoPino = -1,
                   int maxEnPie = 10) {
        if (pinosEnPie < 0) return -1;      // sin informacion de pinos
        if (maxEnPie < 0 || maxEnPie > 10) maxEnPie = 10;

        // ------------------------------------------------------------------
        // Conteo del rack y pinos que siguen en pie
        // ------------------------------------------------------------------
        int rack = (pinosRack >= 0) ? pinosRack : pinosEnPie;
        rack = std::min(rack, maxEnPie);
        int ahoraCuenta = (pinosAhora >= 0) ? pinosAhora : rack;
        ahoraCuenta = std::min(ahoraCuenta, maxEnPie);
        pinosVistos_ = ahoraCuenta;

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
                    pinosVistos_ = despues;
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

    bool tiroEnCurso() const { return tiroEnCurso_; }
    bool enBarrido() const { return enBarrido_; }
    bool pendiente() const { return pendiente_; }
    int rack() const { return pinosAntes_; }
    int ultimoSinBarra() const { return ultimoSinBarra_; }
    int antesPendiente() const { return antesBarrido_; }
    int despuesPendiente() const { return despuesBarrido_; }

    // Diagnostico: entrega el detalle del ultimo barrido confirmado (una vez).
    bool tomarUltimoCierre(std::string& texto) {
        if (!cierreNuevo_) return false;
        texto = cierreTexto_;
        cierreNuevo_ = false;
        return true;
    }

private:
    void aviso(const std::string& texto, int64_t ahoraMs) {
        if (ahoraMs - ultimoAvisoMs_ < 3000) return;
        ultimoAvisoMs_ = ahoraMs;
        cierreTexto_ = texto;
        cierreNuevo_ = true;
    }

    bool tiroEnCurso_ = false;
    bool enBarrido_ = false;          // la barra esta bajada, tiro ya cerrado
    bool bolaVista_ = false;          // se vio la bola desde el ultimo barrido
    int pinosAntes_ = 0;              // rack en pie antes del barrido
    int pinosVistos_ = -1;            // pinos vistos en el ultimo cuadro
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

// Maximo de pinos que se pueden registrar en el tiro actual segun las reglas
static int pinosMaximosPermitidos(const Jugador& j) {
    int fi = j.frameActual();
    if (fi < 0) return -1;
    const std::vector<int>& f = j.frames[fi];
    if (fi == 9) {
        if (f.size() == 1 && f[0] != 10) return 10 - f[0];
        if (f.size() == 2 && f[0] == 10 && f[1] != 10) return 10 - f[1];
        return 10;
    }
    if (f.size() == 1 && f[0] != 10) return 10 - f[0];
    return 10;
}

// Registra los pinos caidos en el marcador, ajustandolos a las reglas del juego
static bool registrarTiro(BowlingGame& juego, int pinos, std::string& mensaje) {
    std::string error;
    if (juego.agregarTiro(pinos, error)) {
        mensaje = "Tiro registrado: " + std::to_string(pinos) + " pino(s)";
        return true;
    }
    int maximo = pinosMaximosPermitidos(juego.jugadorActual());
    std::string error2;
    if (maximo >= 0 && maximo != pinos && juego.agregarTiro(maximo, error2)) {
        mensaje = "Tiro ajustado a " + std::to_string(maximo) + " pino(s) (detectados " +
                  std::to_string(pinos) + ": " + error + ")";
        return true;
    }
    mensaje = "Tiro descartado: " + error;
    return false;
}

#ifdef _WIN32
struct InfoMonitor {
    cv::Rect rect;       // area completa del monitor (para el fullscreen)
    cv::Rect trabajo;    // area util (sin barra de tareas)
    bool primario = false;
};
static std::vector<InfoMonitor> enumerarMonitores() {
    std::vector<InfoMonitor> monitores;
    EnumDisplayMonitors(
        nullptr, nullptr,
        [](HMONITOR hMon, HDC, LPRECT, LPARAM lp) -> BOOL {
            MONITORINFO mi{};
            mi.cbSize = sizeof(MONITORINFO);
            if (GetMonitorInfoW(hMon, &mi)) {
                auto* v = reinterpret_cast<std::vector<InfoMonitor>*>(lp);
                InfoMonitor m;
                m.rect = cv::Rect(mi.rcMonitor.left, mi.rcMonitor.top,
                                  mi.rcMonitor.right - mi.rcMonitor.left,
                                  mi.rcMonitor.bottom - mi.rcMonitor.top);
                m.trabajo = cv::Rect(mi.rcWork.left, mi.rcWork.top,
                                     mi.rcWork.right - mi.rcWork.left,
                                     mi.rcWork.bottom - mi.rcWork.top);
                m.primario = (mi.dwFlags & MONITORINFOF_PRIMARY) != 0;
                v->push_back(m);
            }
            return TRUE;
        },
        reinterpret_cast<LPARAM>(&monitores));
    return monitores;
}
#endif

// ------------------------------------------------------------------
// Correccion de color para la inferencia.
// En el boliche la pista se ilumina con LED azules: la imagen entera toma
// un dominante azul y el modelo (entrenado con luz normal) pierde confianza.
// Antes de la IA se neutraliza ese dominante (balance de blancos + mezcla
// parcial con escala de grises) y se recupera el contraste local. La ventana
// de la camara sigue mostrando la imagen real, sin tocar.
// ------------------------------------------------------------------
static cv::Mat prepararParaIA(const cv::Mat& frame, bool& luzAzul) {
    cv::Mat salida = frame.clone();

    cv::Scalar medias = cv::mean(salida);
    const double b = medias[0], g = medias[1], r = medias[2];
    const double rg = (g + r) * 0.5;
    luzAzul = (b > 1.12 * std::max(1.0, rg)) && (b - rg > 6.0);

    if (!luzAzul) return salida;

    // 1) Balance de blancos "gray world": iguala la media de los 3 canales
    const double objetivo = (b + g + r) / 3.0;
    const double gb = std::clamp(objetivo / std::max(1.0, b), 0.55, 2.50);
    const double gg = std::clamp(objetivo / std::max(1.0, g), 0.55, 2.50);
    const double gr = std::clamp(objetivo / std::max(1.0, r), 0.55, 2.50);

    std::vector<cv::Mat> canales;
    cv::split(salida, canales);
    canales[0].convertTo(canales[0], CV_8U, gb);
    canales[1].convertTo(canales[1], CV_8U, gg);
    canales[2].convertTo(canales[2], CV_8U, gr);
    cv::merge(canales, salida);

    // 2) Quitar el tinte azul residual mezclando parte de la luminancia
    cv::Mat gris, gris3;
    cv::cvtColor(salida, gris, cv::COLOR_BGR2GRAY);
    cv::cvtColor(gris, gris3, cv::COLOR_GRAY2BGR);
    cv::addWeighted(salida, 0.70, gris3, 0.30, 0.0, salida);

    // 3) Contraste local: los pinos blancos resaltan sobre el fondo oscuro
    cv::Mat lab;
    cv::cvtColor(salida, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> labCanales;
    cv::split(lab, labCanales);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(labCanales[0], labCanales[0]);
    cv::merge(labCanales, lab);
    cv::cvtColor(lab, salida, cv::COLOR_Lab2BGR);

    return salida;
}

static void esperarMs(int ms) {
#ifdef _WIN32
    Sleep((DWORD)ms);
#else
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
#endif
}

// Abre una fuente de camara y comprueba que entrega imagenes reales.
// Algunos drivers (MSMF) informan que abrieron el dispositivo pero luego
// no pueden leer ningun cuadro; en ese caso se prueba la siguiente fuente.
static bool abrirFuenteCamara(cv::VideoCapture& cam, const std::string& fuente) {
    cam.release();
    try {
        if (fuente == "0" || fuente == "1" || fuente == "2") {
            cam.open(std::stoi(fuente));
        } else {
            cam.open(fuente);
        }
    } catch (const cv::Exception& e) {
        std::cout << "  error al abrir: " << e.what() << std::endl;
        return false;
    }
    if (!cam.isOpened()) return false;

    cv::Mat prueba;
    for (int i = 0; i < 10; i++) {
        if (cam.read(prueba) && !prueba.empty()) return true;
        esperarMs(250);
    }
    cam.release();
    return false;
}

// Prueba la camara USB (indices 1 y 0) y, si no hay, el streaming del celular.
// Devuelve true solo si la fuente elegida entrego al menos un cuadro real.
static bool conectarCamara(cv::VideoCapture& cam, const std::string& ipCelular) {
    const std::vector<std::string> fuentes = {"1", "0", ipCelular};
    for (const std::string& fuente : fuentes) {
        std::cout << "Probando camara: " << fuente << " ... " << std::flush;
        if (abrirFuenteCamara(cam, fuente)) {
            std::cout << "OK" << std::endl;
            return true;
        }
        std::cout << "sin imagen" << std::endl;
    }
    return false;
}

// ------------------------------------------------------------------
// Configuracion de la partida en la interfaz:
//   - cantidad de jugadores (minimo 4, maximo 8)
//   - nombre de cada jugador (se pide confirmacion de cada uno)
// Se escribe con el teclado: ENTER confirma, BACKSPACE borra y ESC cancela
// (en ese caso se juega con 4 jugadores y los nombres por defecto).
// ------------------------------------------------------------------
static std::vector<std::string> pedirConfiguracionInterfaz(const cv::Rect& monitor) {
    const std::string VENTANA_CFG = "Configuracion - Boliche IA";
    const int ANCHO = 900, ALTO = 300;

    // Mismos colores que el marcador (marcador.cpp)
    const cv::Scalar FONDO(120, 24, 24);
    const cv::Scalar CABECERA(220, 52, 52);
    const cv::Scalar CAJA(90, 20, 20);
    const cv::Scalar CAJA_BORDE(220, 130, 130);
    const cv::Scalar MARCO(240, 150, 150);
    const cv::Scalar TEXTO(255, 255, 255);
    const cv::Scalar TEXTO_SUAVE(255, 200, 200);
    const cv::Scalar ACENTO(0, 251, 161);
    const cv::Scalar AVISO(224, 60, 60);

    cv::namedWindow(VENTANA_CFG, cv::WINDOW_NORMAL);
    cv::resizeWindow(VENTANA_CFG, ANCHO, ALTO);
    // Centrada en el monitor donde aparece
    cv::moveWindow(VENTANA_CFG, monitor.x + (monitor.width - ANCHO) / 2,
                   monitor.y + (monitor.height - ALTO) / 2);
#ifdef _WIN32
    // Se trae al frente una sola vez para que el usuario pueda escribir
    HWND hCfg = FindWindowW(nullptr, L"Configuracion - Boliche IA");
    if (hCfg) SetForegroundWindow(hCfg);
#endif

    std::vector<std::string> nombres;
    std::string texto, error, nombreTemporal;
    int paso = 0;             // 0 = cantidad de jugadores, 1..N = nombre del jugador
    int cantidad = 4;
    bool confirmando = false; // el nombre escrito esta esperando confirmacion

    while (true) {
        cv::Mat lienzo(ALTO, ANCHO, CV_8UC3, FONDO);

        // Marco exterior y cabecera, como el marcador
        cv::rectangle(lienzo, cv::Rect(2, 2, ANCHO - 4, ALTO - 4), MARCO, 2);
        cv::rectangle(lienzo, cv::Rect(20, 20, ANCHO - 40, 48), CABECERA, cv::FILLED);
        cv::putText(lienzo, "CONFIGURACION DE LA PARTIDA", cv::Point(36, 52),
                    cv::FONT_HERSHEY_SIMPLEX, 0.62, TEXTO, 1, cv::LINE_AA);

        std::string pregunta;
        std::string contenido;
        std::string ayuda = "ENTER: confirmar    BACKSPACE: borrar    ESC: cancelar";

        if (paso == 0) {
            pregunta = "Cantidad de jugadores? (4 - 8)";
            contenido = texto + "|";
        } else if (!confirmando) {
            pregunta = "Nombre del jugador " + std::to_string(paso) + ":";
            contenido = texto + "|";
        } else {
            pregunta = "Confirme el nombre del jugador " + std::to_string(paso) + ":";
            contenido = nombreTemporal;
            ayuda = "ENTER: confirmar el nombre    B: escribir otro nombre";
        }

        cv::putText(lienzo, pregunta, cv::Point(40, 120), cv::FONT_HERSHEY_SIMPLEX, 0.85,
                    TEXTO, 2, cv::LINE_AA);

        // Caja de texto
        cv::rectangle(lienzo, cv::Rect(38, 145, ANCHO - 76, 58), CAJA, cv::FILLED);
        cv::rectangle(lienzo, cv::Rect(38, 145, ANCHO - 76, 58), CAJA_BORDE, 2);
        cv::putText(lienzo, contenido, cv::Point(54, 185), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                    ACENTO, 2, cv::LINE_AA);

        if (!error.empty()) {
            cv::putText(lienzo, error, cv::Point(40, 238), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        AVISO, 2, cv::LINE_AA);
        }
        cv::putText(lienzo, ayuda, cv::Point(40, 272), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    TEXTO_SUAVE, 1, cv::LINE_AA);

        cv::imshow(VENTANA_CFG, lienzo);

        int tecla = cv::waitKey(30);
        if (tecla == 27) {                    // ESC: cancelar
            cv::destroyWindow(VENTANA_CFG);
            std::cout << "Configuracion cancelada: 4 jugadores por defecto." << std::endl;
            return {"Jugador 1", "Jugador 2", "Jugador 3", "Jugador 4"};
        }

        if (confirmando) {
            // Solo se acepta la confirmacion o volver a escribir el nombre
            if (tecla == 13 || tecla == 10 || tecla == 's' || tecla == 'S') {
                nombres.push_back(nombreTemporal);
                if ((int)nombres.size() >= cantidad) break;
                paso++;
                confirmando = false;
                nombreTemporal.clear();
                texto.clear();
                error.clear();
            } else if (tecla == 'b' || tecla == 'B' || tecla == 8 || tecla == 127) {
                confirmando = false;
                texto = nombreTemporal;      // se edita el nombre anterior
                nombreTemporal.clear();
                error.clear();
            }
            continue;
        }

        if (tecla == 8 || tecla == 127) {     // BACKSPACE
            if (!texto.empty()) texto.pop_back();
            continue;
        }
        if (tecla == 13 || tecla == 10) {     // ENTER
            if (paso == 0) {
                int n = 0;
                try {
                    n = std::stoi(texto);
                } catch (const std::exception&) {
                    n = 0;
                }
                if (n < 4 || n > 8) {
                    error = "La cantidad debe ser un numero entre 4 y 8.";
                    continue;
                }
                cantidad = n;
                paso = 1;
                texto.clear();
                error.clear();
            } else {
                if (texto.empty()) {
                    error = "El nombre no puede estar vacio.";
                    continue;
                }
                // Se pide confirmar el nombre antes de tomarlo
                nombreTemporal = texto;
                texto.clear();
                confirmando = true;
                error.clear();
            }
            continue;
        }
        if (tecla >= 32 && tecla < 127) {     // caracter normal
            texto += (char)tecla;
            error.clear();
        }
    }

    cv::destroyWindow(VENTANA_CFG);
    std::cout << "Partida para " << std::to_string(cantidad) << " jugadores." << std::endl;
    return nombres;
}

void procesarVideo() {

#ifdef _WIN32
    // Fullscreen nitido: trabajamos en pixeles reales de la pantalla (sin
    // que Windows escale la ventana por el aumento de DPI del 125%).
    SetProcessDPIAware();

    // Anade la carpeta de CUDA al buscador de DLL del proceso, asi no hace falta
    // copiar ~1.6 GB de DLLs (cuBLAS/cuDNN) junto al ejecutable
    SetDllDirectoryW(aWide(BOLICHE_CUDA_BIN_PATH).c_str());
#endif

    // ------------------------------------------------------------------
    // Motor de inferencia: ONNX Runtime con CUDA Execution Provider
    // (la inferencia corre en la RTX 3050, no en la grafica integrada)
    // ------------------------------------------------------------------
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "boliche");

    Ort::SessionOptions sessionOptions;
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    sessionOptions.SetIntraOpNumThreads(4);

    std::string dispositivo = "CPU";
    try {
        OrtCUDAProviderOptions cuda{};
        cuda.device_id = 0;                                 // RTX 3050
        cuda.gpu_mem_limit = 0;                             // sin limite de memoria
        cuda.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        cuda.do_copy_in_default_stream = true;
        sessionOptions.AppendExecutionProvider_CUDA(cuda);
        dispositivo = "CUDA (GPU dedicada, device_id=0)";
    } catch (const std::exception& e) {
        std::cout << "AVISO: no se pudo activar el proveedor CUDA (" << e.what()
                  << "). Se usara la CPU." << std::endl;
    }

    std::cout << "ONNX Runtime " << OrtGetApiBase()->GetVersionString() << " | Execution Providers: ";
    for (const auto& p : Ort::GetAvailableProviders()) std::cout << p << " ";
    std::cout << "\n" << "Dispositivo de inferencia: " << dispositivo << std::endl;

    // Cargar el modelo YOLOv11 entrenado en formato ONNX
    std::string rutaModelo = BOLICHE_MODEL_PATH;
    Ort::Session session(env, aWide(rutaModelo).c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;
    std::string inputName = session.GetInputNameAllocated(0, allocator).get();
    std::string outputName = session.GetOutputNameAllocated(0, allocator).get();

    std::vector<int64_t> inShape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    const int ALTO = (int)inShape[2];
    const int ANCHO = (int)inShape[3];
    std::cout << "Modelo: " << rutaModelo << " | entrada " << inputName << " [1,3," << ALTO << "," << ANCHO
              << "] | salida " << outputName << std::endl;

    // Marcador de la partida: los jugadores los define el usuario al iniciar
    std::vector<std::string> nombres;
    BowlingGame juego;
    SeguimientoTiros seguimiento;
    ValidadorPinos validador;
    int ultimoTiro = -1;          // pinos caidos en el ultimo tiro
    int pinosEnPie = 0;            // pinos detectados en pie ahora mismo
    bool hayBarra = false;         // la barra barredora esta en escena
    bool hayBola = false;          // la bola (cuadro azul) esta en escena
    bool luzAzul = false;          // hay dominante azul (luces LED de la pista)
    int franjasRojas = 0;          // franjas del cuello vistas en el ultimo cuadro
    cv::Rect barraDetectada;       // barra vista por color (la IA no la detecta)

    // Buffer de entrada (NCHW) y tensor asociado al mismo
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> tensorShape = {1, 3, ALTO, ANCHO};
    const char* nombresEntrada[] = {inputName.c_str()};
    const char* nombresSalida[] = {outputName.c_str()};

    cv::Mat frame;
    int contador_cuadros = 0;
    double ms_ia_promedio = 0.0;

    // Diagnostico en vivo: cambios de estado que se registran en consola
    int pinosValidadosPrevios = -1;
    bool barraPrevia = false;
    bool bolaPrevia = false;

    // ------------------------------------------------------------------
    // Reparto de ventanas entre monitores (proceso DPI-aware, pixeles reales):
    // el tablero va fullscreen al monitor secundario (el conectado) y la camara
    // queda en el monitor principal.
    // ------------------------------------------------------------------
    cv::Rect monitorTablero(0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN));
    cv::Rect monitorCamara = monitorTablero;
    bool tableroEnSecundario = false;
    std::vector<InfoMonitor> monitores = enumerarMonitores();
    for (const auto& m : monitores) {
        if (m.primario) monitorCamara = m.trabajo;
    }
    for (const auto& m : monitores) {
        if (!m.primario) {
            monitorTablero = m.rect;
            tableroEnSecundario = true;
            break;
        }
    }
    if (!monitores.empty() && !tableroEnSecundario) {
        monitorTablero = monitores[0].rect;   // un solo monitor
    }
    if (monitorTablero.width < 800 || monitorTablero.height < 600) {
        monitorTablero = cv::Rect(0, 0, 1920, 1080);
    }
    if (monitorCamara.width < 640 || monitorCamara.height < 480) {
        monitorCamara = cv::Rect(0, 0, 1536, 864);
    }

    int PANT_W = monitorTablero.width;
    int PANT_H = monitorTablero.height;
    std::cout << "Tablero fullscreen en "
              << (tableroEnSecundario ? "monitor secundario" : "monitor principal")
              << ": " << PANT_W << "x" << PANT_H << " @ " << monitorTablero.x << ","
              << monitorTablero.y << " | camara en monitor principal ("
              << monitorCamara.width << "x" << monitorCamara.height << ")" << std::endl;

    const std::string VENTANA_TABLERO = "Tablero - Boliche IA";
    const std::string VENTANA_CAMARA = "Camara - Boliche IA";

    // Tablero a pantalla completa en el monitor elegido. Primero lo movemos a ese
    // monitor para que OpenCV aplique el fullscreen sobre el correcto.
    cv::namedWindow(VENTANA_TABLERO, cv::WINDOW_NORMAL);
    cv::moveWindow(VENTANA_TABLERO, monitorTablero.x, monitorTablero.y);
    cv::resizeWindow(VENTANA_TABLERO, PANT_W, PANT_H);
    try {
        cv::setWindowProperty(VENTANA_TABLERO, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    } catch (const cv::Exception& e) {
        std::cout << "No se pudo activar pantalla completa: " << e.what() << std::endl;
    }

    // Camara: ventana normal en el monitor principal
    const int ANCHO_CAMARA = 1280, ALTO_CAMARA = 960;

    // Lienzos fijos (el tablero ocupa toda la pantalla, sin imagen inferior)
    cv::Mat lienzoCam(ALTO_CAMARA, ANCHO_CAMARA, CV_8UC3, cv::Scalar(18, 18, 22));
    cv::Mat lienzoTab(PANT_H, PANT_W, CV_8UC3, cv::Scalar(120, 24, 24));

    // ------------------------------------------------------------------
    // El tablero debe ser lo primero que se ve: se muestra antes de
    // preguntar la configuracion y antes de conectar la camara.
    // ------------------------------------------------------------------
    cv::imshow(VENTANA_TABLERO, lienzoTab);
    cv::waitKey(1);

    // ------------------------------------------------------------------
    // Preguntas de inicio en la interfaz: cantidad de jugadores (de 4 a 8)
    // y el nombre de cada uno.
    // ------------------------------------------------------------------
    nombres = pedirConfiguracionInterfaz(monitorCamara);
    juego = BowlingGame(nombres);

    // El marcador se escala para que quepa completo segun la cantidad de
    // jugadores (una fila por jugador) y quede centrado en la pantalla
    const int filasMarcador = std::max(4, (int)nombres.size());
    const double anchoBaseMarcador = 1040.0;
    const double altoBaseMarcador = 121.0 + 86.0 * filasMarcador;   // alto a escala 1.0
    double escalaTablero = (PANT_W - 32.0) / anchoBaseMarcador;
    double escalaPorAlto = (PANT_H - 32.0) / altoBaseMarcador;
    if (escalaPorAlto < escalaTablero) escalaTablero = escalaPorAlto;

    // Ventana de la camara (normal) en el monitor principal
    int ventanaCamW = std::min(ANCHO_CAMARA, std::max(320, monitorCamara.width - 40));
    int ventanaCamH = std::min(ALTO_CAMARA, std::max(240, monitorCamara.height - 40));
    cv::namedWindow(VENTANA_CAMARA, cv::WINDOW_NORMAL);
    cv::resizeWindow(VENTANA_CAMARA, ventanaCamW, ventanaCamH);
    cv::moveWindow(VENTANA_CAMARA, monitorCamara.x + 24, monitorCamara.y + 48);

    // ------------------------------------------------------------------
    // El tablero debe ser lo primero que se vea: se dibuja y muestra antes
    // de conectar la camara. Si la camara falla, el tablero sigue visible.
    // ------------------------------------------------------------------
    auto actualizarTablero = [&]() {
        cv::Mat tablero = dibujarMarcador(juego, pinosEnPie, ultimoTiro, escalaTablero);
        int xTablero = (PANT_W - tablero.cols) / 2;
        int yTablero = std::max(0, (PANT_H - tablero.rows) / 2);   // centrado en la pantalla
        tablero.copyTo(lienzoTab(cv::Rect(xTablero, yTablero, tablero.cols, tablero.rows)));
    };
    actualizarTablero();
    cv::imshow(VENTANA_TABLERO, lienzoTab);
    cv::waitKey(1);

    // Conectar a la cámara (0 para webcam USB, cambia por ip_celular si usas el teléfono)
    std::string ip_celular = "http://192.168.137.229:8080/video";

    // Se abre con DirectShow (DSHOW): el backend MSMF de OpenCV usa Media
    // Foundation con un hilo asincrono y, cuando el dispositivo falla o se
    // desconecta, deja el proceso sin respuesta (fallo fatal). Con DSHOW la
    // perdida de la camara solo devuelve un error y se puede reconectar.
    cv::VideoCapture cam;
    cam.open(1, cv::CAP_DSHOW);
    if (!cam.isOpened()) {
        std::cout << "DSHOW no disponible, se usa el backend por defecto." << std::endl;
        cam.open(1);
    }

    if (!cam.isOpened()) {
        std::cout << "Error: No se pudo conectar a la camara. Verifica la IP." << std::endl;
        cv::destroyAllWindows();
        return;
    }

    // EL DRIVER DE LA CAMARA PUEDE FALLAR (se desconecta el dispositivo o el
    // driver reinicia): deja de entregar cuadros y en algun caso lanza
    // excepciones de OpenCV. Por eso la lectura va protegida y, si se pierde
    // la camara, se reconecta sin tumbar el programa; el tablero sigue visible.
    bool salir = false;
    bool camaraViva = true;
    int64_t ultimoIntento = 0;

    // Captura de depuracion: se activa con la variable de entorno
    // BOLICHEIA_DEBUG=1. Guarda en la carpeta "depuracion" los cuadros del
    // tiro y del barrido (con 60 cuadros de pre-roll) para poder ajustar la
    // deteccion de la barra sin adivinar. No modifica la logica normal.
    bool depurarCuadros = false;
    {
        const char* envDebug = std::getenv("BOLICHEIA_DEBUG");
        depurarCuadros = (envDebug != nullptr) && (std::string(envDebug) == "1");
        if (depurarCuadros) {
            std::filesystem::create_directories("depuracion");
            std::cout << "DEBUG: captura de cuadros activada (carpeta depuracion)" << std::endl;
        }
    }
    std::deque<std::vector<uchar>> bufferDebug;
    bool grabandoDebug = false;
    int64_t inicioDebugMs = 0;
    int64_t finDebugMs = 0;
    int capturaDebug = 0;

    // Vuelca los cuadros acumulados a disco (tambien al salir, para no perder
    // la captura si se cierra la app antes del final del barrido).
    auto volcarCapturaDebug = [&](const char* motivo) {
        if (bufferDebug.empty()) return;
        char carpeta[64];
        std::snprintf(carpeta, sizeof(carpeta), "depuracion\\captura_%03d", ++capturaDebug);
        std::error_code ec;
        std::filesystem::create_directories(carpeta, ec);
        if (ec) {
            std::cout << "DEBUG: no se pudo crear " << carpeta << std::endl;
            return;
        }
        int n = 0;
        for (const std::vector<uchar>& j : bufferDebug) {
            char ruta[160];
            std::snprintf(ruta, sizeof(ruta), "%s\\cuadro_%04d.jpg", carpeta, n++);
            std::ofstream salidaJpg(ruta, std::ios::binary);
            salidaJpg.write(reinterpret_cast<const char*>(j.data()),
                            (std::streamsize)j.size());
        }
        std::cout << "DEBUG: " << n << " cuadros guardados en " << carpeta
                  << " (" << motivo << ")" << std::endl;
        bufferDebug.clear();
    };

    while (!salir) {
        if (!camaraViva) {
            // Aviso en la ventana de la camara mientras se reintenta
            cv::Mat aviso = lienzoCam.clone();
            cv::putText(aviso, "CAMARA NO DISPONIBLE", cv::Point(40, 220),
                        cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);
            cv::putText(aviso, "Reconectando...", cv::Point(40, 260),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 1);
            actualizarTablero();
            cv::imshow(VENTANA_TABLERO, lienzoTab);
            cv::imshow(VENTANA_CAMARA, aviso);

            int t = cv::waitKey(200);
            if (t == 27) { salir = true; break; }

            int64_t ahora = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now().time_since_epoch()).count();
            if (ahora - ultimoIntento < 2000) continue;
            ultimoIntento = ahora;

            try {
                cam.release();
            } catch (const cv::Exception& e) {
                std::cout << "Camara: " << e.what() << std::endl;
            } catch (...) {
                std::cout << "Camara: error desconocido al liberar" << std::endl;
            }
            try {
                cam.open(1, cv::CAP_DSHOW);
                if (!cam.isOpened()) cam.open(1);
                camaraViva = cam.isOpened();
            } catch (const cv::Exception& e) {
                std::cout << "Camara: error al reconectar: " << e.what() << std::endl;
                camaraViva = false;
            } catch (...) {
                std::cout << "Camara: error desconocido al reconectar" << std::endl;
                camaraViva = false;
            }
            if (camaraViva) std::cout << "Camara reconectada." << std::endl;
            continue;
        }

        // Lectura protegida del cuadro
        bool leido = false;
        try {
            leido = cam.read(frame) && !frame.empty();
        } catch (const cv::Exception& e) {
            std::cout << "Camara: error al leer el cuadro: " << e.what() << std::endl;
        }
        if (!leido) {
            std::cout << "Camara: se perdio la senal. Reconectando..." << std::endl;
            camaraViva = false;
            continue;
        }
        // El driver puede cambiar el formato al reiniciar: se asegura CV_8UC3
        if (frame.type() != CV_8UC3) frame.convertTo(frame, CV_8UC3);

        // Filtro permanente: la camara trabaja siempre en blanco y negro
        if (CAMARA_SIEMPRE_BLANCO_Y_NEGRO) {
            cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
            cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
        }

        contador_cuadros++;

        bool procesar_IA = (contador_cuadros % PROCESAR_CADA_N_CUADROS == 0);

        int width = frame.cols;
        int height = frame.rows;

        std::vector<Deteccion> detecciones;

        if (procesar_IA) {
            // La correccion se aplica sobre la imagen ya reducida al tamano de
            // entrada del modelo: mismo resultado con mucho menos coste.
            cv::Mat frameReducida;
            cv::resize(frame, frameReducida, cv::Size(ANCHO, ALTO), 0, 0, cv::INTER_LINEAR);
            cv::Mat frameIA = prepararParaIA(frameReducida, luzAzul);

            // Franjas del cuello: en blanco y negro (Caso 2: Black-Hat) o bajo luz azul
            // se detectan y se usan como evidencia extra para validar el pino.
            std::vector<cv::Rect> franjas;
            if (CAMARA_SIEMPRE_BLANCO_Y_NEGRO || luzAzul) {
                franjas = detectarFranjasRojas(frameReducida);
                const double sx = frame.cols / (double)ANCHO;
                const double sy = frame.rows / (double)ALTO;
                for (cv::Rect& f : franjas) {
                    f.x = (int)std::lround(f.x * sx);
                    f.y = (int)std::lround(f.y * sy);
                    f.width = (int)std::lround(f.width * sx);
                    f.height = (int)std::lround(f.height * sy);
                }
            }
            franjasRojas = (int)franjas.size();

            // Preparar la imagen para YOLO (640x640) con el mismo preprocesado
            // que usaba OpenCV DNN: escala 1/255 y cambio de BGR a RGB
            cv::Mat blob = cv::dnn::blobFromImage(frameIA, 1.0 / 255.0, cv::Size(ANCHO, ALTO),
                                                   cv::Scalar(), true, false);

            // El tensor de entrada apunta al buffer del blob (vive durante la inferencia)
            Ort::Value tensorEntrada = Ort::Value::CreateTensor<float>(memInfo, blob.ptr<float>(),
                                                                      (size_t)blob.total(),
                                                                      tensorShape.data(), tensorShape.size());

            // Ejecutar la deteccion en la GPU
            auto t0 = std::chrono::high_resolution_clock::now();
            std::vector<Ort::Value> salida = session.Run(Ort::RunOptions{nullptr}, nombresEntrada,
                                                         &tensorEntrada, 1, nombresSalida, 1);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms_ia = std::chrono::duration<double, std::milli>(t1 - t0).count();
            ms_ia_promedio = (ms_ia_promedio == 0.0) ? ms_ia : (ms_ia_promedio * 0.9 + ms_ia * 0.1);

            // Decodificar salida de YOLOv11 Multi-Clase
            std::vector<int64_t> forma = salida[0].GetTensorTypeAndShapeInfo().GetShape();
            const float* data = salida[0].GetTensorData<float>();

            // Formato habitual [1, 4+clases, anchors]; algunos exportes usan [1, anchors, 4+clases]
            const bool canalPrimero = (forma[1] == 4 + CLASES);
            const int num_anchors = canalPrimero ? (int)forma[2] : (int)forma[1];

            auto valor = [&](int canal, int ancla) -> float {
                return canalPrimero ? data[canal * num_anchors + ancla]
                                    : data[ancla * (4 + CLASES) + canal];
            };

            // NMS por clase: si se hiciese global, la caja de la bola podria
            // eliminar la de un pino y contariamos menos pinos de los reales.
            // Con luz azul el contraste baja: umbrales algo mas permisivos
            // (la validacion temporal filtra los falsos positivos).
            const float umbrales[CLASES] = {
                luzAzul ? 0.38f : UMBRAL_BOLA,
                luzAzul ? 0.32f : UMBRAL_BOLO,
                luzAzul ? 0.38f : UMBRAL_BARRA
            };
            for (int clase = 0; clase < CLASES; ++clase) {
                std::vector<cv::Rect> cajas;
                std::vector<float> confianzas;
                for (int i = 0; i < num_anchors; ++i) {
                    float conf = valor(4 + clase, i);
                    if (conf < umbrales[clase]) continue;
                    float cx = valor(0, i) * (width / (float)ANCHO);
                    float cy = valor(1, i) * (height / (float)ALTO);
                    float w = valor(2, i) * (width / (float)ANCHO);
                    float h = valor(3, i) * (height / (float)ALTO);
                    cajas.push_back(cv::Rect((int)(cx - w / 2), (int)(cy - h / 2), (int)w, (int)h));
                    confianzas.push_back(conf);
                }
                if (cajas.empty()) continue;

                std::vector<int> idx;
                cv::dnn::NMSBoxes(cajas, confianzas, umbrales[clase], NMS_THRESHOLD, idx);
                for (int k : idx) {
                    detecciones.push_back({cajas[k], confianzas[k], clase});
                }
            }

            // Cajas de pinos, bola en escena y barra
            std::vector<cv::Rect> cajasPinos;
            hayBola = false;
            hayBarra = false;
            barraDetectada = cv::Rect();
            for (const Deteccion& d : detecciones) {
                if (d.clase == 1) cajasPinos.push_back(d.box);
                else if (d.clase == 0) hayBola = true;
                else if (d.clase == 2) hayBarra = true;
            }

            // La barra real es azul con rayas magenta: si la IA no la ve,
            // se busca su firma de color en la imagen real.
            if (!hayBarra) {
                cv::Rect barraColor;
                if (detectarBarraPorColor(frameReducida, barraColor)) {
                    hayBarra = true;
                    // El detector de color trabaja en 640x640: se lleva a las
                    // coordenadas del cuadro real para dibujarla bien.
                    const double bx = width / (double)ANCHO;
                    const double by = height / (double)ALTO;
                    barraDetectada = cv::Rect((int)std::lround(barraColor.x * bx),
                                              (int)std::lround(barraColor.y * by),
                                              (int)std::lround(barraColor.width * bx),
                                              (int)std::lround(barraColor.height * by));
                }
            }

            int64_t ahora = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now().time_since_epoch()).count();

            // Solo cuentan los pinos validados durante 1 s, maximo 10.
            // Durante el barrido no se descartan (la barra los tapa).
            pinosEnPie = validador.actualizar(cajasPinos, ahora, hayBarra, franjas);

            // Pinos que se estan viendo de verdad ahora: es lo que mide el rack
            // cuando baja la barra (el conteo normal sigue 3 s con los caidos).
            int pinosRack = validador.enPieRecientes(ahora, MS_RACK);
            int pinosAhora = validador.enPieRecientes(ahora, MS_CAIDA);
            const int64_t ultimaVista = validador.ultimaVistaMs();
            const int64_t msUltimoPino = (ultimaVista > 0) ? (ahora - ultimaVista) : -1;

            // Cambios de estado: sirven para ver en consola por que no registra
            if (pinosEnPie != pinosValidadosPrevios) {
                std::cout << "Pinos validados: " << pinosEnPie << "/10 (validando "
                          << validador.validando() << ", cajas IA " << cajasPinos.size()
                          << ")" << std::endl;
                pinosValidadosPrevios = pinosEnPie;
            }
            if (hayBarra != barraPrevia) {
                std::cout << "Barra: " << (hayBarra ? "visible" : "fuera")
                          << (barraDetectada.empty() ? "" : " (por color)") << std::endl;
                barraPrevia = hayBarra;
            }
            if (hayBola != bolaPrevia) {
                std::cout << "Bola: " << (hayBola ? "visible" : "fuera") << std::endl;
                bolaPrevia = hayBola;
            }

            // Tope de pinos en pie segun el marcador: impide que pinos ya
            // caidos (detecciones fantasma) inflen el rack del tiro actual.
            int maxEnPie = juego.pinosEnPieEsperados();
            if (maxEnPie < 0 || maxEnPie > 10) maxEnPie = 10;
            int pinosCaidos = seguimiento.actualizar(pinosEnPie, hayBola, hayBarra, ahora,
                                                     pinosRack, pinosAhora, msUltimoPino,
                                                     maxEnPie);
            std::string cierre;
            if (seguimiento.tomarUltimoCierre(cierre)) std::cout << cierre << std::endl;
            if (pinosCaidos >= 0) {
                std::string mensaje;
                if (registrarTiro(juego, pinosCaidos, mensaje)) {
                    ultimoTiro = pinosCaidos;
                    std::cout << mensaje << std::endl;
                } else {
                    std::cout << mensaje << std::endl;
                }
            }

            // ----------------------------------------------------------
            // Captura de depuracion (BOLICHEIA_DEBUG=1). Guarda el cuadro
            // reducido en JPEG: 60 cuadros de pre-roll desde que aparece la
            // bola y luego hasta 4 s despues de ver la barra. Sirve para
            // revisar como se ve realmente el barrido y ajustar el detector.
            // ----------------------------------------------------------
            if (depurarCuadros) {
                std::vector<uchar> jpgDebug;
                cv::imencode(".jpg", frameReducida, jpgDebug,
                             {cv::IMWRITE_JPEG_QUALITY, 88});
                bufferDebug.push_back(std::move(jpgDebug));

                if (hayBola) {
                    if (!grabandoDebug) {
                        grabandoDebug = true;
                        inicioDebugMs = ahora;
                        std::cout << "DEBUG: captura iniciada" << std::endl;
                    }
                    finDebugMs = ahora + 12000;   // 12 s despues de la ultima bola
                } else if (grabandoDebug && hayBarra) {
                    finDebugMs = ahora + 4000;    // 4 s despues de ver la barra
                }
                if (grabandoDebug && (ahora - inicioDebugMs) > 25000) {
                    finDebugMs = ahora;           // tope por si la bola no sale
                }

                // Fuera de la grabacion solo se conservan los ultimos 60 cuadros.
                if (!grabandoDebug) {
                    while (bufferDebug.size() > 60) bufferDebug.pop_front();
                }

                if (grabandoDebug && ahora >= finDebugMs) {
                    char carpeta[64];
                    std::snprintf(carpeta, sizeof(carpeta), "depuracion\\captura_%03d",
                                  ++capturaDebug);
                    std::filesystem::create_directories(carpeta);
                    int n = 0;
                    for (const std::vector<uchar>& j : bufferDebug) {
                        char ruta[160];
                        std::snprintf(ruta, sizeof(ruta), "%s\\cuadro_%04d.jpg", carpeta, n++);
                        std::ofstream salidaJpg(ruta, std::ios::binary);
                        salidaJpg.write(reinterpret_cast<const char*>(j.data()),
                                        (std::streamsize)j.size());
                    }
                    std::cout << "DEBUG: " << n << " cuadros guardados en " << carpeta
                              << std::endl;
                    bufferDebug.clear();
                    grabandoDebug = false;
                }
            }

            if (contador_cuadros % 30 == 0) {
                std::cout << "IA: " << ms_ia_promedio << " ms/frame (" << 1000.0 / ms_ia_promedio
                          << " FPS) | " << dispositivo << " | pinos validados: " << pinosEnPie << "/10"
                          << " | validando: " << validador.validando()
                          << " | cajas IA: " << detecciones.size()
                          << " (pino " << cajasPinos.size() << ", bola " << (hayBola ? 1 : 0)
                          << ", barra " << (hayBarra ? 1 : 0) << ")"
                          << " | rack: " << seguimiento.rack()
                          << " sinBarra: " << seguimiento.ultimoSinBarra()
                          << " pendiente: " << (seguimiento.pendiente() ? "si" : "no")
                          << " tiro: " << (seguimiento.tiroEnCurso() ? "si" : "no")
                          << " | franjas rojas: " << franjasRojas
                          << " | luz azul: " << (luzAzul ? "si (correccion activa)" : "no");
                if (!barraDetectada.empty()) std::cout << " | barra por color";
                std::cout << std::endl;
                if (seguimiento.pendiente()) {
                    std::cout << "   Barrido pendiente: antes=" << seguimiento.antesPendiente()
                              << " despues=" << seguimiento.despuesPendiente() << std::endl;
                }
                // Cajas de los pinos que sigue el validador (validados y en curso)
                for (const ValidadorPinos::Pino& p : validador.pines()) {
                    std::cout << "   pino " << (p.confirmado ? "OK " : ".. ")
                              << "(" << p.box.x << "," << p.box.y << ","
                              << p.box.width << "," << p.box.height << ")"
                              << " visto=" << p.vistoMs << "ms"
                              << " prog=" << (int)(p.progreso * 100) << "%" << std::endl;
                }
            }
        }

        // ------------------------------------------------------------------
        // Ventana de la camara (normal): video, detecciones y estado arriba
        // ------------------------------------------------------------------
        double escVista = std::min(ANCHO_CAMARA / (double)frame.cols,
                                   ALTO_CAMARA / (double)frame.rows);
        int anchoVista = std::max(1, (int)std::lround(frame.cols * escVista));
        int altoVista = std::max(1, (int)std::lround(frame.rows * escVista));
        int dxVista = (ANCHO_CAMARA - anchoVista) / 2;
        int dyVista = (ALTO_CAMARA - altoVista) / 2;

        cv::Mat vista;
        cv::resize(frame, vista, cv::Size(anchoVista, altoVista));
        vista.copyTo(lienzoCam(cv::Rect(dxVista, dyVista, anchoVista, altoVista)));

        const float sx = anchoVista / (float)frame.cols;
        const float sy = altoVista / (float)frame.rows;
        const double fEtiqueta = 0.7 * (anchoVista / 1024.0);
        const double fEtiquetaValidando = 0.6 * (anchoVista / 1024.0);

        // Estado en una franja oscura arriba (deja libres los pinos de abajo)
        const Jugador& actual = juego.jugadorActual();
        std::string infoTurno = "Turno: " + actual.nombre;
        if (actual.frameActual() >= 0) {
            infoTurno += "   Frame " + std::to_string(actual.frameActual() + 1) +
                         "   Tiro " + std::to_string(actual.numeroTiro());
        }
        int esperados = juego.pinosEnPieEsperados();
        std::string infoPinos = "Pinos en pie: " + std::to_string(pinosEnPie) + "/10" +
                                "   validando: " + std::to_string(validador.validando()) +
                                "   (esperados: " + (esperados >= 0 ? std::to_string(esperados) : "-") + ")";
        if (CAMARA_SIEMPRE_BLANCO_Y_NEGRO || luzAzul) infoPinos += "   franjas: " + std::to_string(franjasRojas);
        std::string infoTiro = "Ultimo tiro: " +
                               (ultimoTiro >= 0 ? std::to_string(ultimoTiro) + " pinos caidos" : "-");
        std::string infoIA = "IA: " + std::to_string(ms_ia_promedio) + " ms (" +
                             std::to_string((int)(1000.0 / std::max(0.01, ms_ia_promedio))) + " FPS) | " + dispositivo;
        if (CAMARA_SIEMPRE_BLANCO_Y_NEGRO) infoIA += " | B/N ON (morfologia)";
        else if (luzAzul) infoIA += " | luz azul ON";

        cv::Rect franja(0, 0, ANCHO_CAMARA, 304);
        cv::Mat zona = lienzoCam(franja);
        cv::Mat oscuro(zona.size(), CV_8UC3, cv::Scalar(8, 10, 14));
        cv::addWeighted(oscuro, 0.62, zona, 0.38, 0, zona);

        const int xTexto = 24, yTexto = 40, dyLinea = 44;
        cv::putText(lienzoCam, infoIA, cv::Point(xTexto, yTexto), cv::FONT_HERSHEY_SIMPLEX,
                    0.75, cv::Scalar(170, 170, 190), 2);
        cv::putText(lienzoCam, infoTurno, cv::Point(xTexto, yTexto + dyLinea), cv::FONT_HERSHEY_SIMPLEX,
                    0.8, cv::Scalar(255, 255, 255), 2);
        cv::putText(lienzoCam, infoPinos, cv::Point(xTexto, yTexto + 2 * dyLinea), cv::FONT_HERSHEY_SIMPLEX,
                    0.8, cv::Scalar(0, 255, 255), 2);
        cv::putText(lienzoCam, infoTiro, cv::Point(xTexto, yTexto + 3 * dyLinea), cv::FONT_HERSHEY_SIMPLEX,
                    0.8, cv::Scalar(0, 255, 0), 2);
        // La barra es el unico cierre del tiro: cuenta lo que cayo desde el ultimo barrido
        if (seguimiento.tiroEnCurso()) {
            cv::putText(lienzoCam, "Tiro en curso: esperando la barra para cerrar",
                        cv::Point(xTexto, yTexto + 4 * dyLinea), cv::FONT_HERSHEY_SIMPLEX,
                        0.8, cv::Scalar(255, 160, 0), 2);
        } else {
            cv::putText(lienzoCam, "Esperando bola o caida de pinos",
                        cv::Point(xTexto, yTexto + 4 * dyLinea), cv::FONT_HERSHEY_SIMPLEX,
                        0.8, cv::Scalar(140, 140, 150), 2);
        }
        if (hayBarra) {
            cv::putText(lienzoCam, "BARRA DETECTADA: cambio de tiro", cv::Point(xTexto, yTexto + 5 * dyLinea),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        }

        for (const Deteccion& d : detecciones) {
            if (d.clase == 1) continue;   // los pinos se dibujan segun su validacion
            cv::Rect box((int)(d.box.x * sx) + dxVista, (int)(d.box.y * sy) + dyVista,
                         (int)(d.box.width * sx), (int)(d.box.height * sy));
            cv::Scalar color;
            std::string nombre;
            if (d.clase == 0) { color = cv::Scalar(255, 0, 0); nombre = "Bola"; }           // Azul
            else { color = cv::Scalar(0, 0, 255); nombre = "Barra Barredora"; }            // Rojo
            cv::rectangle(lienzoCam, box, color, 3);
            std::string label = nombre + ": " + std::to_string(int(d.confianza * 100)) + "%";
            cv::putText(lienzoCam, label, cv::Point(box.x, std::max(24, box.y - 8)),
                        cv::FONT_HERSHEY_SIMPLEX, fEtiqueta, color, 2);
        }

        // Barra vista por color (la IA no la reconoce): se dibuja en rojo
        if (!barraDetectada.empty()) {
            const float sxBarra = anchoVista / (float)frame.cols;
            const float syBarra = altoVista / (float)frame.rows;
            cv::Rect box((int)(barraDetectada.x * sxBarra) + dxVista,
                         (int)(barraDetectada.y * syBarra) + dyVista,
                         (int)(barraDetectada.width * sxBarra),
                         (int)(barraDetectada.height * syBarra));
            cv::rectangle(lienzoCam, box, cv::Scalar(0, 0, 255), 3);
            cv::putText(lienzoCam, "Barra Barredora (color)", cv::Point(box.x, std::max(24, box.y - 8)),
                        cv::FONT_HERSHEY_SIMPLEX, fEtiqueta, cv::Scalar(0, 0, 255), 2);
        }

        // Pinos: verde = validado (1 s), amarillo = en validacion con su progreso
        for (const ValidadorPinos::Pino& p : validador.pines()) {
            cv::Rect box((int)(p.box.x * sx) + dxVista, (int)(p.box.y * sy) + dyVista,
                         (int)(p.box.width * sx), (int)(p.box.height * sy));
            if (p.confirmado) {
                cv::rectangle(lienzoCam, box, cv::Scalar(0, 255, 0), 3);
                cv::putText(lienzoCam, "Pino", cv::Point(box.x, std::max(24, box.y - 8)),
                            cv::FONT_HERSHEY_SIMPLEX, fEtiqueta, cv::Scalar(0, 255, 0), 2);
            } else {
                cv::rectangle(lienzoCam, box, cv::Scalar(0, 255, 255), 2);
                int pct = (int)(p.progreso * 100);
                cv::putText(lienzoCam, "Validando " + std::to_string(pct) + "%",
                            cv::Point(box.x, std::max(24, box.y - 8)),
                            cv::FONT_HERSHEY_SIMPLEX, fEtiquetaValidando, cv::Scalar(0, 255, 255), 2);
                // Barra de progreso debajo de la caja
                cv::Rect fondo(box.x, box.y + box.height + 4, box.width, 6);
                cv::rectangle(lienzoCam, fondo, cv::Scalar(80, 80, 80), cv::FILLED);
                cv::Rect lleno(fondo.x, fondo.y, (int)(fondo.width * p.progreso), fondo.height);
                cv::rectangle(lienzoCam, lleno, cv::Scalar(0, 255, 255), cv::FILLED);
            }
        }

        // (la camara se muestra al final del bucle, por encima del tablero)

        // ------------------------------------------------------------------
        // Ventana del tablero (fullscreen): marcador centrado + barra inferior
        // ------------------------------------------------------------------
        actualizarTablero();
        cv::imshow(VENTANA_TABLERO, lienzoTab);
        // La camara se muestra al final para que quede por encima del tablero
        cv::imshow(VENTANA_CAMARA, lienzoCam);
        if (contador_cuadros == 1 || contador_cuadros == 30) {
            // Prioridad al tablero: debe ser lo primero que se vea al iniciar
            HWND hTab = FindWindowW(nullptr, L"Tablero - Boliche IA");
            if (hTab) {
                SetForegroundWindow(hTab);
                SetWindowPos(hTab, HWND_TOP, 0, 0, 0, 0,
                             SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
            }
        }

        int tecla = cv::waitKey(1);
        if (tecla == 27) { salir = true; break; } // ESC para salir

        if (tecla == 'r' || tecla == 'R') {
            juego.reiniciar(nombres);
            ultimoTiro = -1;
            std::cout << "Partida reiniciada" << std::endl;
        } else if (tecla == 't' || tecla == 'T') {
            std::string mensaje;
            registrarTiro(juego, 10, mensaje);
            ultimoTiro = 10;
            std::cout << "Tiro manual (strike): " << mensaje << std::endl;
        } else if (tecla >= '0' && tecla <= '9') {
            std::string mensaje;
            registrarTiro(juego, tecla - '0', mensaje);
            ultimoTiro = tecla - '0';
            std::cout << "Tiro manual: " << mensaje << std::endl;
        }
    }

    try {
        cam.release();
    } catch (const cv::Exception& e) {
        std::cout << "Camara: " << e.what() << std::endl;
    }
    cv::destroyAllWindows();
}
