#include "vision/detector_barra.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

// Movimiento: diferencia entre cuadros consecutivos.
constexpr int DIF_UMBRAL = 20;

// Perfil horizontal: fraccion de la fila que debe estar en movimiento para
// considerarla parte de la banda.
constexpr double FILA_MIN = 0.50;

// Zona de la pista donde puede estar la barra. Se excluye el foso (arriba) y
// el borde inferior del cuadro, que son estaticos y oscuros.
constexpr double X_INI = 0.04;
constexpr double X_FIN = 0.96;
constexpr double Y_INI = 0.30;
constexpr double Y_FIN = 0.92;

// Geometria de la barra: banda ancha, plana y maciza.
constexpr double ANCHO_MIN = 0.55;    // del ancho del cuadro
constexpr double ALTO_MIN = 0.03;     // del alto del cuadro
constexpr double ALTO_MAX = 0.30;     // del alto del cuadro
constexpr double RATIO_MIN = 3.5;     // ancho / alto
constexpr double RELLENO_MIN = 0.35;  // pixeles en movimiento dentro de la caja
constexpr double FONDO_MIN = 0.65;    // la barra barre hasta la parte baja
constexpr double DOMINANCIA_MIN = 0.25; // su movimiento domina sobre el resto

// Confirmacion temporal: el seguimiento del tiro exige varios cuadros de
// barra, asi que el detector entrega cada cuadro valido y no se auto-confirma.
constexpr int HUECO_MAX = 2;          // filas sin movimiento dentro de la banda
constexpr double SALTO_Y_MAX = 0.18;  // salto vertical permitido entre cuadros

// Destellos de exposicion: la camara cambia el brillo de todo el cuadro y
// aparece una banda oscura falsa en el borde inferior. En los videos reales
// los barridos tienen cuadros de hasta 12.3 de cambio y los destellos
// conocidos de 14.9 a 43. Con 12 se descarta algun cuadro aislado de un
// barrido (el seguimiento exige 3 y cada barrido aporta varios) pero no se
// cuela ningun destello observado.
constexpr double BRILLO_DELTA_MAX = 12.0;
constexpr double MOV_GLOBAL_MAX = 0.55;

struct Banda {
    cv::Rect caja;
    double anchoRel = 0.0;
    double altoRel = 0.0;
    double relacion = 0.0;
    double relleno = 0.0;
    double dominancia = 0.0;
    const char* motivo = "sin banda";
    double puntaje = -1.0;
    bool valida = false;
};

// Devuelve la banda de movimiento mas creible del cuadro, si la hay.
Banda buscarBanda(const cv::Mat& movimiento, int ancho, int alto, DiagnosticoBarra& diag) {
    Banda mejor;

    const int x0 = (int)std::lround(X_INI * ancho);
    const int x1 = (int)std::lround(X_FIN * ancho);
    const int y0 = (int)std::lround(Y_INI * alto);
    const int y1 = (int)std::lround(Y_FIN * alto);
    if (x1 - x0 < 32 || y1 - y0 < 32) return mejor;

    // Perfil: que fraccion de cada fila esta en movimiento.
    const int anchoFila = x1 - x0;
    std::vector<double> filas((size_t)y1, 0.0);
    for (int y = y0; y < y1; ++y) {
        const int cuenta = cv::countNonZero(movimiento(cv::Rect(x0, y, anchoFila, 1)));
        filas[(size_t)y] = cuenta / (double)anchoFila;
    }

    // Rachas de filas activas, tolerando huecos pequenos (barra desenfocada).
    std::vector<std::pair<int, int>> rachas;
    int inicio = -1, finAnterior = -1;
    for (int y = y0; y <= y1; ++y) {
        const bool activa = (y < y1) && (filas[(size_t)y] >= FILA_MIN);
        if (activa) {
            if (inicio < 0) {
                inicio = y;
            } else if (y - finAnterior - 1 > HUECO_MAX) {
                rachas.emplace_back(inicio, finAnterior + 1);
                inicio = y;
            }
            finAnterior = y;
        }
    }
    if (inicio >= 0) rachas.emplace_back(inicio, finAnterior + 1);
    if (rachas.empty()) return mejor;

    // Movimiento total de referencia para exigir dominancia.
    const cv::Rect zona(x0, (int)std::lround(0.06 * alto), x1 - x0,
                        (int)std::lround(0.94 * alto) - (int)std::lround(0.06 * alto));
    const int movimientoTotal = cv::countNonZero(movimiento(zona));
    if (movimientoTotal <= 0) return mejor;

    // Se guarda como diagnostico la banda mas grande aunque no cumpla todo,
    // para poder ajustar los umbrales con videos reales.
    double mejorCandidata = -1.0;

    for (const auto& [desde, hasta] : rachas) {
        const int altoRacha = hasta - desde;
        if (altoRacha <= 0) continue;

        // Caja real de los pixeles en movimiento de la racha.
        std::vector<cv::Point> puntos;
        cv::findNonZero(movimiento(cv::Rect(x0, desde, anchoFila, altoRacha)), puntos);
        if (puntos.empty()) continue;
        for (cv::Point& p : puntos) p += cv::Point(x0, desde);
        const cv::Rect caja = cv::boundingRect(puntos);

        const double anchoRel = caja.width / (double)ancho;
        const double altoRel = caja.height / (double)alto;
        const double relacion = caja.width / (double)std::max(1, caja.height);
        const double relleno = puntos.size() / (double)std::max(1, caja.width * caja.height);
        const double dominancia =
            cv::countNonZero(movimiento(caja)) / (double)std::max(1, movimientoTotal);

        Banda b;
        b.caja = caja;
        b.anchoRel = anchoRel;
        b.altoRel = altoRel;
        b.relacion = relacion;
        b.relleno = relleno;
        b.dominancia = dominancia;

        if (anchoRel + relleno > mejorCandidata) {
            mejorCandidata = anchoRel + relleno;
            diag.geometria = false;
            diag.caja = caja;
            diag.anchoRel = anchoRel;
            diag.altoRel = altoRel;
            diag.relacion = relacion;
            diag.relleno = relleno;
            diag.dominancia = dominancia;
            diag.motivo = "";
        }

        if (anchoRel < ANCHO_MIN) {
            b.motivo = "ancho";
        } else if (altoRel < ALTO_MIN || altoRel > ALTO_MAX) {
            b.motivo = "alto";
        } else if (relacion < RATIO_MIN) {
            b.motivo = "relacion";
        } else if (relleno < RELLENO_MIN) {
            b.motivo = "relleno";
        } else if (caja.y < (int)std::lround(0.06 * alto) ||
                   caja.y + caja.height < (int)std::lround(FONDO_MIN * alto) ||
                   caja.y + caja.height > (int)std::lround(0.94 * alto)) {
            b.motivo = "posicion";
        } else if (dominancia < DOMINANCIA_MIN) {
            b.motivo = "dominancia";
        } else {
            b.motivo = "ok";
            b.valida = true;
            b.puntaje = anchoRel + relleno + 0.5 * dominancia - 0.2 * altoRel;
            if (b.puntaje > mejor.puntaje) {
                mejor = b;
                diag.geometria = true;
                diag.caja = caja;
                diag.anchoRel = anchoRel;
                diag.altoRel = altoRel;
                diag.relacion = relacion;
                diag.relleno = relleno;
                diag.dominancia = dominancia;
                diag.motivo = "ok";
            }
        }
        // Si la banda mas grande no es valida, el motivo queda en el
        // diagnostico para saber por que se descarto.
        if (b.caja == diag.caja) diag.motivo = b.motivo;
    }
    return mejor;
}

cv::Mat aGris(const cv::Mat& imagen) {
    if (imagen.channels() == 1) return imagen;
    cv::Mat gris;
    cv::cvtColor(imagen, gris, cv::COLOR_BGR2GRAY);
    return gris;
}

} // namespace

void DetectorBarra::reiniciar() {
    anterior_.release();
    ultima_ = cv::Rect();
    diagnostico_ = DiagnosticoBarra();
    racha_ = 0;
}

bool DetectorBarra::actualizar(const cv::Mat& gris, cv::Rect& barra) {
    barra = cv::Rect();
    if (gris.empty()) {
        reiniciar();
        return false;
    }

    const cv::Mat actual = aGris(gris);
    if (anterior_.empty() || anterior_.size() != actual.size() ||
        anterior_.type() != actual.type()) {
        actual.copyTo(anterior_);
        racha_ = 0;
        ultima_ = cv::Rect();
        diagnostico_ = DiagnosticoBarra();
        return false;
    }

    cv::Mat diferencia;
    cv::absdiff(actual, anterior_, diferencia);
    cv::Mat movimiento;
    cv::threshold(diferencia, movimiento, DIF_UMBRAL, 255, cv::THRESH_BINARY);

    static const cv::Mat cierre =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 3));
    static const cv::Mat apertura =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 3));
    cv::morphologyEx(movimiento, movimiento, cv::MORPH_CLOSE, cierre);
    cv::morphologyEx(movimiento, movimiento, cv::MORPH_OPEN, apertura);

    diagnostico_ = DiagnosticoBarra();
    diagnostico_.brilloDelta =
        std::abs(cv::mean(actual)[0] - cv::mean(anterior_)[0]);
    diagnostico_.movimientoGlobal =
        cv::countNonZero(movimiento) / (double)(actual.cols * actual.rows);
    diagnostico_.destello = diagnostico_.brilloDelta > BRILLO_DELTA_MAX ||
                            diagnostico_.movimientoGlobal > MOV_GLOBAL_MAX;
    if (diagnostico_.destello) {
        // Cuadro de exposicion: no se busca barra y se rompe la racha.
        racha_ = 0;
        ultima_ = cv::Rect();
        diagnostico_.motivo = "destello";
        actual.copyTo(anterior_);
        return false;
    }

    const Banda banda = buscarBanda(movimiento, actual.cols, actual.rows, diagnostico_);

    if (!diagnostico_.caja.empty()) {
        const cv::Rect caja = diagnostico_.caja & cv::Rect(0, 0, actual.cols, actual.rows);
        if (caja.width > 0 && caja.height > 0) {
            diagnostico_.brilloDentro = cv::mean(actual(caja))[0];
            const int yb0 = caja.y + caja.height;
            const int yb1 = std::min(actual.rows,
                                     yb0 + std::max(2, (int)std::lround(0.05 * actual.rows)));
            if (yb1 - yb0 >= 2) {
                diagnostico_.brilloDebajo =
                    cv::mean(actual(cv::Rect(caja.x, yb0, caja.width, yb1 - yb0)))[0];
            }
        }
    }

    if (banda.valida) {
        const bool continua =
            !ultima_.empty() &&
            std::abs((banda.caja.y + banda.caja.height / 2) -
                     (ultima_.y + ultima_.height / 2)) <=
                (int)std::lround(SALTO_Y_MAX * actual.rows);
        racha_ = continua ? racha_ + 1 : 1;
        ultima_ = banda.caja;
    } else {
        racha_ = 0;
        ultima_ = cv::Rect();
    }
    diagnostico_.racha = racha_;

    actual.copyTo(anterior_);
    if (!banda.valida) return false;
    barra = banda.caja;
    return true;
}

bool barraIAPlausible(const cv::Rect& caja, int anchoCuadro, int altoCuadro) {
    if (anchoCuadro <= 0 || altoCuadro <= 0) return false;
    if (caja.width < (int)std::lround(0.45 * anchoCuadro)) return false;
    if (caja.height < (int)std::lround(0.025 * altoCuadro)) return false;
    if (caja.height > (int)std::lround(0.30 * altoCuadro)) return false;
    if (caja.width < 4 * caja.height) return false;
    if (caja.y < (int)std::lround(0.06 * altoCuadro)) return false;
    // La banda oscura inferior (apoyo del canal) es estatica: la IA la
    // confunde con la barra. La barra real nunca llega al borde inferior.
    if (caja.y + caja.height > (int)std::lround(0.88 * altoCuadro)) return false;
    return true;
}
