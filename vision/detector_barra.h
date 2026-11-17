#pragma once

#include <opencv2/opencv.hpp>

// Datos del ultimo cuadro analizado. Sirven para ajustar el detector con
// videos reales y para el diagnostico en vivo.
struct DiagnosticoBarra {
    bool geometria = false;   // la banda cumple tamano y posicion
    cv::Rect caja;
    double anchoRel = 0.0;
    double altoRel = 0.0;
    double relacion = 0.0;
    double relleno = 0.0;
    double dominancia = 0.0;
    double brilloDelta = 0.0;      // cambio de brillo medio entre cuadros
    double movimientoGlobal = 0.0; // fraccion del cuadro con movimiento
    double brilloDentro = 0.0;     // brillo medio de la banda
    double brilloDebajo = 0.0;     // brillo medio de la pista bajo la banda
    bool destello = false;         // cambio global de exposicion
    int racha = 0;
    const char* motivo = "";
};

// Detector de la barra barredora para camara en blanco y negro.
//
// La barra es la unica banda ancha y plana que recorre la parte baja de la
// pista. Se compara el cuadro con el anterior y se exige:
//   - banda horizontal en la zona de la pista (ni foso ni borde inferior),
//   - ancho, altura, relacion de aspecto y relleno de barra,
//   - que el movimiento de la banda domine sobre el resto del cuadro
//     (un temblor de camara mueve todo y no domina).
//
// El detector entrega cada cuadro valido; la confirmacion temporal la hace
// SeguimientoTiros con CUADROS_BARRA. Los destellos de exposicion se
// descartan por cambio de brillo global.
//
// El impacto de la bola y el pisador de pinos quedan fuera por geometria y
// por zona: su movimiento es pequeno, disperso o esta en la parte alta.
class DetectorBarra {
public:
    // gris: cuadro reducido de un canal (el mismo que usa la app).
    // barra: caja detectada, en coordenadas de gris.
    // Devuelve true en cada cuadro con una banda de barra valida. El
    // seguimiento del tiro es quien exige varios cuadros para confirmar.
    bool actualizar(const cv::Mat& gris, cv::Rect& barra);

    // Se llama al reconectar la camara: no hay cuadro anterior valido.
    void reiniciar();

    const DiagnosticoBarra& diagnostico() const { return diagnostico_; }

private:
    cv::Mat anterior_;
    cv::Rect ultima_;
    DiagnosticoBarra diagnostico_;
    int racha_ = 0;
};

// La IA a veces marca como barra la banda oscura estatica del borde inferior.
// Solo se acepta una caja de clase 2 con geometria de barra real.
bool barraIAPlausible(const cv::Rect& caja, int anchoCuadro, int altoCuadro);
