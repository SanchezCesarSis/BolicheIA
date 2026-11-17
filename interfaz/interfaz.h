#pragma once

#include "marcador.h"
#include "vision/tipos.h"
#include "vision/validador_pinos.h"

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Reparto de ventanas entre monitores (proceso DPI-aware, pixeles reales):
// el tablero va fullscreen al monitor secundario (el conectado) y la camara
// queda en el monitor principal.
struct RepartoPantallas {
    cv::Rect monitorTablero;
    cv::Rect monitorCamara;
    bool tableroEnSecundario = false;
};

RepartoPantallas repartirPantallas();

// Preguntas de inicio en la interfaz: cantidad de jugadores (4 a 8) y el
// nombre de cada uno. ESC cancela y juega con 4 jugadores por defecto.
std::vector<std::string> pedirConfiguracionInterfaz(const cv::Rect& monitor);

// Estado que la aplicacion entrega a la ventana de la camara para el HUD.
struct EstadoVista {
    int pinosEnPie = 0;
    int validando = 0;
    int esperados = -1;
    int franjasRojas = 0;
    int ultimoTiro = -1;
    double msIA = 0.0;
    bool hayBola = false;
    bool hayBarra = false;
    bool tiroEnCurso = false;
    bool pausaMaquina = false;
    bool luzAzul = false;
};

// Ventanas del juego: tablero fullscreen y camara en ventana normal.
// Toda la parte visual queda aqui, fuera de la logica de deteccion.
class VentanasJuego {
public:
    void crear(const RepartoPantallas& reparto);
    void ajustarEscala(int filasMarcador);

    void mostrarTablero(const BowlingGame& juego, int pinosEnPie, int ultimoTiro);
    void mostrarCamara(const cv::Mat& frame,
                       const std::vector<Deteccion>& detecciones,
                       const std::vector<ValidadorPinos::Pino>& pinos,
                       const cv::Rect& barraDetectada,
                       const BowlingGame& juego,
                       const EstadoVista& estado,
                       const std::string& dispositivo);
    void mostrarAvisoCamara();
    void traerTableroAlFrente() const;
    int esperarTecla(int ms) const;
    void cerrar() const;

private:
    std::string ventanaTablero_;
    std::string ventanaCamara_;
    cv::Mat lienzoCam_;
    cv::Mat lienzoTab_;
    cv::Mat vistaCam_;
    int anchoTab_ = 0;
    int altoTab_ = 0;
    int filasMarcador_ = 4;
    double escalaTablero_ = 1.0;
};
