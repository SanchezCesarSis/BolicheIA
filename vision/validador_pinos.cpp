#include "vision/validador_pinos.h"

#include "vision/preprocesado.h"

#include <algorithm>

int ValidadorPinos::actualizar(const std::vector<cv::Rect>& detecciones, int64_t ahoraMs,
                               bool pausar, const std::vector<cv::Rect>& franjas) {
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
        ultimaVistaGlobalMs_ = ahoraMs;
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
            ultimaVistaGlobalMs_ = ahoraMs;
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

int ValidadorPinos::enPie() const {
    int n = 0;
    for (const Pino& p : pines_) if (p.confirmado) ++n;
    return std::min(n, MAX_PINOS);
}

int ValidadorPinos::validando() const {
    int n = 0;
    for (const Pino& p : pines_) if (!p.confirmado) ++n;
    return n;
}

int ValidadorPinos::enPieRecientes(int64_t ahoraMs, int64_t ventanaMs) const {
    int n = 0;
    for (const Pino& p : pines_)
        if (p.confirmado && p.ultimaVistaIAms > 0 &&
            (ahoraMs - p.ultimaVistaIAms) <= ventanaMs) ++n;
    return std::min(n, MAX_PINOS);
}

int64_t ValidadorPinos::ultimaVistaMs() const {
    int64_t t = ultimaVistaGlobalMs_;
    for (const Pino& p : pines_) t = std::max(t, p.ultimaVistaIAms);
    return t;
}

cv::Point2f ValidadorPinos::centro(const cv::Rect& r) {
    return cv::Point2f(r.x + r.width * 0.5f, r.y + r.height * 0.5f);
}

double ValidadorPinos::iou(const cv::Rect& a, const cv::Rect& b) {
    const int x1 = std::max(a.x, b.x);
    const int y1 = std::max(a.y, b.y);
    const int x2 = std::min(a.x + a.width, b.x + b.width);
    const int y2 = std::min(a.y + a.height, b.y + b.height);
    if (x2 <= x1 || y2 <= y1) return 0.0;
    const double inter = (double)(x2 - x1) * (double)(y2 - y1);
    const double uni = (double)a.width * a.height + (double)b.width * b.height - inter;
    return uni > 0.0 ? inter / uni : 0.0;
}

bool ValidadorPinos::mismaCaja(const cv::Rect& a, const cv::Rect& b) {
    const double limite =
        std::max(14.0, 0.35 * (double)std::max(a.height, b.height));
    return iou(a, b) >= 0.25 || cv::norm(centro(a) - centro(b)) <= limite;
}

void ValidadorPinos::sumarPresencia(Pino& p, int64_t ahoraMs) {
    const int64_t hueco = ahoraMs - p.ultimaVezMs;
    if (hueco > MS_HUECO) p.vistoMs = 0;
    else if (hueco > 0) p.vistoMs += hueco;
    p.ultimaVezMs = ahoraMs;
    if (!p.confirmado && p.vistoMs >= MS_VALIDACION) {
        p.confirmado = true;
        p.vistoMs = MS_VALIDACION;
    }
}
