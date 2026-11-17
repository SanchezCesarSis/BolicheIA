#pragma once

#include "vision/tipos.h"

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <optional>
#include <string>
#include <vector>

// Detector YOLO (ONNX Runtime). Encapsula la sesion, el proveedor CUDA y la
// decodificacion/NMS de la salida. El preprocesado de color se hace fuera:
// el detector recibe la imagen ya reducida y corregida.
class DetectorYolo {
public:
    explicit DetectorYolo(const std::string& rutaModelo, bool usarCuda = true);

    // frameEntrada: imagen BGR del tamano de entrada del modelo.
    // anchoOriginal/altoOriginal: tamano del cuadro real para escalar las cajas.
    std::vector<Deteccion> detectar(const cv::Mat& frameEntrada, int anchoOriginal,
                                    int altoOriginal, bool luzAzul);

    int anchoEntrada() const { return anchoEntrada_; }
    int altoEntrada() const { return altoEntrada_; }
    const std::string& dispositivo() const { return dispositivo_; }
    double ultimoMs() const { return ultimoMs_; }

private:
    Ort::Env env_;
    std::optional<Ort::Session> sesion_;
    Ort::MemoryInfo memoria_;
    std::string nombreEntrada_;
    std::string nombreSalida_;
    std::vector<int64_t> formaTensor_;
    cv::Mat blob_;
    int anchoEntrada_ = 0;
    int altoEntrada_ = 0;
    double ultimoMs_ = 0.0;
    std::string dispositivo_ = "CPU";
};
