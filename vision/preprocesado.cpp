#include "vision/preprocesado.h"

#include <algorithm>
#include <cmath>

InfoGris analizarGris(const cv::Mat& bgr, bool forzarBlancoYNegro) {
    InfoGris info;
    if (bgr.empty()) return info;

    if (bgr.channels() == 1) {
        info.gris = bgr;
        info.esBlancoYNegro = true;
        info.valida = true;
        return info;
    }
    if (bgr.channels() != 3) return info;

    if (forzarBlancoYNegro) {
        // La camara ya entrega la imagen en blanco y negro: el canal 0 es la
        // luminancia y no hace falta comparar canales.
        cv::extractChannel(bgr, info.gris, 0);
        info.esBlancoYNegro = true;
        info.valida = true;
        return info;
    }

    cv::split(bgr, info.canales);

    // Comprobar si la imagen es monocromatica / blanco y negro
    cv::Mat difRG;
    cv::absdiff(info.canales[2], info.canales[1], difRG);
    double maxDif = 0.0;
    cv::minMaxLoc(difRG, nullptr, &maxDif);
    info.esBlancoYNegro = (maxDif < 5.0);
    if (info.esBlancoYNegro) info.gris = info.canales[0];
    info.valida = true;
    return info;
}

std::vector<cv::Rect> detectarFranjasCuello(const cv::Mat& bgr) {
    if (bgr.empty()) return {};
    return detectarFranjasCuello(bgr, analizarGris(bgr));
}

std::vector<cv::Rect> detectarFranjasCuello(const cv::Mat& bgr, const InfoGris& info) {
    std::vector<cv::Rect> franjas;
    if (bgr.empty() || !info.valida) return franjas;

    cv::Mat mascara;

    if (!info.esBlancoYNegro) {
        if (info.canales.size() < 3) return franjas;

        // --- Metodo por Color (R - G) ---
        cv::Mat rojo, diferencia, diferenciaUmbral;
        cv::threshold(info.canales[2], rojo, 130, 255, cv::THRESH_BINARY);
        cv::subtract(info.canales[2], info.canales[1], diferencia); // R - G (saturado a 0)
        cv::threshold(diferencia, diferenciaUmbral, 18, 255, cv::THRESH_BINARY);
        cv::bitwise_and(rojo, diferenciaUmbral, mascara);
    } else {
        // --- Metodo para Blanco y Negro (Caso 2: Filtro Morfologico Black-Hat) ---
        // En escala de grises, el pino es blanco brillante y la franja roja es una banda
        // horizontal oscura en el cuello. Black-Hat (Close - Original) realza esas bandas oscuras.
        if (info.gris.empty()) return franjas;
        static const cv::Mat elemFranja =
            cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 3));

        cv::Mat blackhat;
        cv::morphologyEx(info.gris, blackhat, cv::MORPH_BLACKHAT, elemFranja);

        // Mascara de brillo: asegurarse de que la franja este sobre el pino iluminado (brillo > 80)
        cv::Mat mascaraBrillo;
        cv::threshold(info.gris, mascaraBrillo, 80, 255, cv::THRESH_BINARY);

        cv::threshold(blackhat, mascara, 16, 255, cv::THRESH_BINARY);
        cv::bitwise_and(mascara, mascaraBrillo, mascara);
    }

    // Unir la franja horizontal y eliminar pixeles aislados
    static const cv::Mat elemCierre =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 1));
    static const cv::Mat elemApertura =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 1));
    cv::morphologyEx(mascara, mascara, cv::MORPH_CLOSE, elemCierre);
    cv::morphologyEx(mascara, mascara, cv::MORPH_OPEN, elemApertura);

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

bool detectarBarra(const cv::Mat& bgr, cv::Rect& barra) {
    barra = cv::Rect();
    if (bgr.empty()) return false;
    return detectarBarra(bgr, barra, analizarGris(bgr));
}

bool detectarBarra(const cv::Mat& bgr, cv::Rect& barra, const InfoGris& info) {
    barra = cv::Rect();
    if (bgr.empty() || !info.valida) return false;

    std::vector<std::vector<cv::Point>> contornos;

    if (!info.esBlancoYNegro) {
        if (info.canales.size() < 3) return false;

        // --- 1) Rayas magenta/rojas de los bordes de la barra ---
        cv::Mat diferencia, mascara, rojo;
        cv::subtract(info.canales[2], info.canales[1], diferencia); // R - G
        cv::threshold(diferencia, diferencia, 25, 255, cv::THRESH_BINARY);
        cv::threshold(info.canales[2], rojo, 90, 255, cv::THRESH_BINARY);
        cv::bitwise_and(diferencia, rojo, mascara);

        // La raya es larga: se cierra en horizontal
        static const cv::Mat elemRaya =
            cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 1));
        cv::morphologyEx(mascara, mascara, cv::MORPH_CLOSE, elemRaya);

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
        cv::subtract(info.canales[0], info.canales[1], masB);   // B - G
        cv::subtract(info.canales[0], info.canales[2], masB2);  // B - R
        cv::threshold(masB, masB, 35, 255, cv::THRESH_BINARY);
        cv::threshold(masB2, masB2, 35, 255, cv::THRESH_BINARY);
        cv::bitwise_and(masB, masB2, azul);
        static const cv::Mat elemAzul =
            cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 3));
        cv::morphologyEx(azul, azul, cv::MORPH_CLOSE, elemAzul);
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
        if (info.gris.empty()) return false;
        cv::Mat oscura;
        cv::threshold(info.gris, oscura, 70, 255, cv::THRESH_BINARY_INV);
        static const cv::Mat elemOscura =
            cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 3));
        cv::morphologyEx(oscura, oscura, cv::MORPH_CLOSE, elemOscura);
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

bool franjaEnCuello(const cv::Rect& caja, const std::vector<cv::Rect>& franjas) {
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

cv::Mat prepararParaIA(const cv::Mat& frame, bool& luzAzul) {
    luzAzul = false;
    if (frame.empty() || frame.channels() != 3) return frame;

    cv::Scalar medias = cv::mean(frame);
    const double b = medias[0], g = medias[1], r = medias[2];
    const double rg = (g + r) * 0.5;
    luzAzul = (b > 1.12 * std::max(1.0, rg)) && (b - rg > 6.0);

    // Sin dominante azul no hay nada que corregir: se devuelve la misma
    // imagen (sin clonar) para no gastar memoria en cada cuadro.
    if (!luzAzul) return frame;

    cv::Mat salida = frame.clone();

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
