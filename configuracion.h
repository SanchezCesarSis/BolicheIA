#pragma once

#include <cstdint>

// ------------------------------------------------------------------
// Configuracion del sistema de deteccion
// ------------------------------------------------------------------

// NMS por clase (asi la caja de la bola no elimina la de un pino)
inline constexpr float NMS_THRESHOLD = 0.4f;

// Con la GPU dedicada la inferencia tarda ~10 ms: se puede detectar en cada cuadro
inline constexpr int PROCESAR_CADA_N_CUADROS = 1;
inline constexpr int CLASES = 3; // 0 = Bola, 1 = Bolo (pino), 2 = Barra barredora

// Umbral de confianza por clase
inline constexpr float UMBRAL_BOLA = 0.45f;
inline constexpr float UMBRAL_BOLO = 0.45f;
inline constexpr float UMBRAL_BARRA = 0.45f;

// Filtro permanente: la camara y todo el procesamiento operan siempre en blanco y negro
inline constexpr bool CAMARA_SIEMPRE_BLANCO_Y_NEGRO = true;

// Seguimiento del tiro (para saber cuando termina y cuantos pinos cayeron)
inline constexpr int CUADROS_BARRA = 3;          // detecciones de barra para confirmar el barrido
inline constexpr int64_t MS_VENTANA_BARRA = 1200; // margen para juntar esas detecciones
inline constexpr int64_t MS_FIN_BARRIDO = 600;    // sin ver la barra este tiempo: el barrido termino

// Mientras la barra esta bajada (aunque el detector solo la vea a ratos) no se
// actualiza el conteo de pinos: la barredora los tapa y el conteo caeria a 0.
inline constexpr int64_t MS_CONGELAR_CONTEO = 2000;

// Tras registrar un tiro la maquina baja a recolocar los pinos que quedaron.
// Durante ese ciclo no se buscan pinos: si se cuentan los que baja la maquina,
// el tiro siguiente arranca con un rack inflado y registra mal los caidos.
inline constexpr int64_t MS_PAUSA_MAQUINA = 13000;

inline constexpr int64_t MS_ESTABLE_CAIDA = 600;  // pinos quietos este tiempo antes de cerrar el tiro

// Tiempos del registro (independientes de la tolerancia visual del validador).
// Un pino se da por caido para el marcador cuando deja de verse este tiempo:
// es corto a proposito, porque la barra barredora puede bajar enseguida.
inline constexpr int64_t MS_CAIDA = 450;

// Ventana para medir el rack en pie (los pinos estan quietos, se toleran
// parpadeos de la deteccion). El rack solo crece, nunca se resta.
inline constexpr int64_t MS_RACK = 1200;

// Jugadores que permite la interfaz de inicio
inline constexpr int MIN_JUGADORES = 4;
inline constexpr int MAX_JUGADORES = 8;

// La ruta la define CMakeLists.txt para no depender del directorio de trabajo
#ifndef BOLICHE_MODEL_PATH
#define BOLICHE_MODEL_PATH "bolos.onnx"
#endif

// Carpeta con las DLL de CUDA (cuBLAS / cuDNN) que necesita el Execution Provider
#ifndef BOLICHE_CUDA_BIN_PATH
#define BOLICHE_CUDA_BIN_PATH "C:/ort/cuda/bin"
#endif
