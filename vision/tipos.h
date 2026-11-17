#pragma once

#include <opencv2/core.hpp>

// Una deteccion ya filtrada por NMS. El NMS se hace por clase para que un pin
// no sea eliminado por la caja de la bola.
struct Deteccion {
    cv::Rect box;
    float confianza = 0.0f;
    int clase = -1;
};
