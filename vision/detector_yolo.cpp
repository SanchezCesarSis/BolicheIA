#include "vision/detector_yolo.h"

#include "configuracion.h"

#include <opencv2/dnn.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#endif

namespace {

std::wstring aWide(const std::string& ruta) {
    return std::wstring(ruta.begin(), ruta.end());
}

} // namespace

DetectorYolo::DetectorYolo(const std::string& rutaModelo, bool usarCuda)
    : env_(ORT_LOGGING_LEVEL_WARNING, "boliche"),
      memoria_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {

#ifdef _WIN32
    // Anade la carpeta de CUDA al buscador de DLL del proceso, asi no hace falta
    // copiar ~1.6 GB de DLLs (cuBLAS/cuDNN) junto al ejecutable.
    SetDllDirectoryW(aWide(BOLICHE_CUDA_BIN_PATH).c_str());
#endif

    Ort::SessionOptions sessionOptions;
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    sessionOptions.SetIntraOpNumThreads(4);

    if (usarCuda) {
        try {
            OrtCUDAProviderOptions cuda{};
            cuda.device_id = 0;                                 // RTX 3050
            cuda.gpu_mem_limit = 0;                             // sin limite de memoria
            cuda.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
            cuda.do_copy_in_default_stream = true;
            sessionOptions.AppendExecutionProvider_CUDA(cuda);
            dispositivo_ = "CUDA (GPU dedicada, device_id=0)";
        } catch (const std::exception& e) {
            std::cout << "AVISO: no se pudo activar el proveedor CUDA (" << e.what()
                      << "). Se usara la CPU." << std::endl;
        }
    }

    std::cout << "ONNX Runtime " << OrtGetApiBase()->GetVersionString()
              << " | Execution Providers: ";
    for (const auto& p : Ort::GetAvailableProviders()) std::cout << p << " ";
    std::cout << "\nDispositivo de inferencia: " << dispositivo_ << std::endl;

    // Cargar el modelo YOLOv11 entrenado en formato ONNX
    sesion_.emplace(env_, aWide(rutaModelo).c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;
    nombreEntrada_ = sesion_->GetInputNameAllocated(0, allocator).get();
    nombreSalida_ = sesion_->GetOutputNameAllocated(0, allocator).get();

    std::vector<int64_t> inShape =
        sesion_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    altoEntrada_ = (int)inShape[2];
    anchoEntrada_ = (int)inShape[3];
    formaTensor_ = {1, 3, altoEntrada_, anchoEntrada_};

    std::cout << "Modelo: " << rutaModelo << " | entrada " << nombreEntrada_
              << " [1,3," << altoEntrada_ << "," << anchoEntrada_ << "] | salida "
              << nombreSalida_ << std::endl;
}

std::vector<Deteccion> DetectorYolo::detectar(const cv::Mat& frameEntrada, int anchoOriginal,
                                              int altoOriginal, bool luzAzul) {
    std::vector<Deteccion> detecciones;
    if (!sesion_ || frameEntrada.empty() || anchoOriginal <= 0 || altoOriginal <= 0)
        return detecciones;

    // El blob se reutiliza entre cuadros: evita reservar memoria en cada inferencia.
    cv::dnn::blobFromImage(frameEntrada, blob_, 1.0 / 255.0,
                           cv::Size(anchoEntrada_, altoEntrada_), cv::Scalar(), true, false);

    // El tensor de entrada apunta al buffer del blob (vive durante la inferencia)
    Ort::Value tensorEntrada = Ort::Value::CreateTensor<float>(
        memoria_, blob_.ptr<float>(), (size_t)blob_.total(),
        formaTensor_.data(), formaTensor_.size());

    const char* nombresEntrada[] = {nombreEntrada_.c_str()};
    const char* nombresSalida[] = {nombreSalida_.c_str()};

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<Ort::Value> salida = sesion_->Run(
        Ort::RunOptions{nullptr}, nombresEntrada, &tensorEntrada, 1, nombresSalida, 1);
    auto t1 = std::chrono::high_resolution_clock::now();
    ultimoMs_ = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (salida.empty() || !salida[0].IsTensor()) return detecciones;

    // Decodificar salida de YOLOv11 Multi-Clase
    std::vector<int64_t> forma = salida[0].GetTensorTypeAndShapeInfo().GetShape();
    if (forma.size() < 3) return detecciones;
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

    detecciones.reserve(16);
    for (int clase = 0; clase < CLASES; ++clase) {
        std::vector<cv::Rect> cajas;
        std::vector<float> confianzas;
        for (int i = 0; i < num_anchors; ++i) {
            float conf = valor(4 + clase, i);
            if (conf < umbrales[clase]) continue;
            float cx = valor(0, i) * (anchoOriginal / (float)anchoEntrada_);
            float cy = valor(1, i) * (altoOriginal / (float)altoEntrada_);
            float w = valor(2, i) * (anchoOriginal / (float)anchoEntrada_);
            float h = valor(3, i) * (altoOriginal / (float)altoEntrada_);
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

    return detecciones;
}
