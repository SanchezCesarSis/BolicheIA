#ifndef BOLICHE_INTELIGENTE_MARCADOR_H
#define BOLICHE_INTELIGENTE_MARCADOR_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

class Jugador {
public:
    explicit Jugador(const std::string& nombre = "Jugador");

    std::string nombre;
    std::vector<std::vector<int>> frames;

    int frameActual() const;
    bool frameNecesitaTiro(int fi) const;
    bool necesitaTiro() const;
    bool terminado() const;
    std::string validarTiro(int pinos) const;
    void registrarTiro(int pinos);
    int pinosEnPieEsperados() const;
    int numeroTiro() const;
    std::vector<int> calcularPuntuacion() const;
    int total() const;
};

class BowlingGame {
public:
    explicit BowlingGame(const std::vector<std::string>& nombres = {});

    bool agregarTiro(int pinos, std::string& error);
    const Jugador& jugadorActual() const;
    int indiceJugadorActual() const;
    bool juegoTerminado() const;
    const std::vector<Jugador>& jugadores() const;
    void reiniciar(const std::vector<std::string>& nombres = {});
    int pinosEnPieEsperados() const;

private:
    std::vector<Jugador> jugadores_;
    int actual_ = 0;
    bool terminado_ = false;
};

std::string formatearTiros(const std::vector<int>& tiros);

// Maximo de pinos que se pueden registrar en el tiro actual segun las reglas
int pinosMaximosPermitidos(const Jugador& j);

// Registra los pinos caidos en el marcador, ajustandolos a las reglas del juego
bool registrarTiro(BowlingGame& juego, int pinos, std::string& mensaje);

// Dibuja el marcador. pinosEnPie y ultimoTiro son opcionales: si se pasan,
// se muestran en el pie de pagina junto a los puntos del ultimo tiro.
// escala agranda/reduce todo el marcador (1.0 = tamano original).
cv::Mat dibujarMarcador(const BowlingGame& juego, int pinosEnPie = -1, int ultimoTiro = -1,
                        double escala = 1.0);

#endif //BOLICHE_INTELIGENTE_MARCADOR_H
