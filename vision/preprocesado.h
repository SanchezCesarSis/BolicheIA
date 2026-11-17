#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

// Datos de color calculados una sola vez por cuadro. Evita repetir el split y
// la comprobacion de blanco y negro en cada detector.
struct InfoGris {
    std::vector<cv::Mat> canales;  // B, G, R cuando la entrada es de 3 canales
    cv::Mat gris;                  // luminancia usada por las ramas en blanco y negro
    bool esBlancoYNegro = false;
    bool valida = false;
};

// Analiza la imagen una vez. Si se sabe de antemano que la camara esta en
// blanco y negro, forzarBlancoYNegro evita el split y la comparacion de canales.
InfoGris analizarGris(const cv::Mat& bgr, bool forzarBlancoYNegro = false);

// Franjas del cuello del pino, en color (R - G) o en blanco y negro (Black-Hat).
std::vector<cv::Rect> detectarFranjasCuello(const cv::Mat& bgr);
std::vector<cv::Rect> detectarFranjasCuello(const cv::Mat& bgr, const InfoGris& info);

// Barra de barrido: rayas magenta, cuerpo azul o banda oscura horizontal (B/N).
bool detectarBarra(const cv::Mat& bgr, cv::Rect& barra);
bool detectarBarra(const cv::Mat& bgr, cv::Rect& barra, const InfoGris& info);

// Hay una franja dentro de la zona del cuello del pino (parte superior)?
bool franjaEnCuello(const cv::Rect& caja, const std::vector<cv::Rect>& franjas);

// Corrige el dominante azul de las luces LED antes de la inferencia.
// Si la imagen no tiene dominante azul devuelve la misma imagen (sin copia).
cv::Mat prepararParaIA(const cv::Mat& frame, bool& luzAzul);
