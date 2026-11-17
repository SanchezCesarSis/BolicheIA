#include "marcador.h"
#include <algorithm>
#include <cmath>

Jugador::Jugador(const std::string& nombre) : nombre(nombre) {
    frames.resize(10);
}

int Jugador::frameActual() const {
    for (int i = 0; i < 10; i++) {
        if (frameNecesitaTiro(i)) return i;
    }
    return -1;
}

bool Jugador::frameNecesitaTiro(int fi) const {
    const auto& f = frames[fi];
    if (fi == 9) {
        if (f.size() < 2) return true;
        if (f.size() == 2) return f[0] == 10 || f[0] + f[1] == 10;
        return false;
    }
    if (f.empty()) return true;
    if (f.size() == 1) return f[0] != 10;
    return false;
}

bool Jugador::necesitaTiro() const {
    return frameActual() >= 0;
}

bool Jugador::terminado() const {
    return !necesitaTiro();
}

std::string Jugador::validarTiro(int pinos) const {
    if (pinos < 0 || pinos > 10) return "Los pinos deben estar entre 0 y 10";
    int fi = frameActual();
    if (fi < 0) return "Este jugador ya termino";
    const auto& f = frames[fi];
    if (fi == 9) {
        if (f.size() == 1 && f[0] != 10 && f[0] + pinos > 10)
            return "Invalido: " + std::to_string(f[0]) + "+" + std::to_string(pinos) + " supera 10";
        if (f.size() == 2 && f[0] == 10 && f[1] != 10 && f[1] + pinos > 10)
            return "Invalido: " + std::to_string(f[1]) + "+" + std::to_string(pinos) + " supera 10";
    } else {
        if (f.size() == 1 && f[0] != 10 && f[0] + pinos > 10)
            return "Invalido: " + std::to_string(f[0]) + "+" + std::to_string(pinos) + " supera 10";
    }
    return "";
}

void Jugador::registrarTiro(int pinos) {
    int fi = frameActual();
    if (fi < 0) return;
    frames[fi].push_back(pinos);
}

int Jugador::pinosEnPieEsperados() const {
    int fi = frameActual();
    if (fi < 0) return -1;
    const auto& f = frames[fi];
    if (f.empty()) return 10;
    if (fi < 9) {
        if (f.size() == 1) return 10 - f[0];
        return 10;
    }
    if (f.size() == 1) return f[0] == 10 ? 10 : 10 - f[0];
    if (f.size() == 2) {
        if (f[0] == 10) return f[1] == 10 ? 10 : 10 - f[1];
        if (f[0] + f[1] == 10) return 10;
    }
    return -1;
}

int Jugador::numeroTiro() const {
    int fi = frameActual();
    if (fi < 0) return 0;
    return (int)frames[fi].size() + 1;
}

std::vector<int> Jugador::calcularPuntuacion() const {
    std::vector<int> scores(10, -1);
    std::vector<int> todos;
    for (const auto& f : frames)
        todos.insert(todos.end(), f.begin(), f.end());

    int acumulado = 0;
    size_t idx = 0;

    for (int fn = 0; fn < 10; fn++) {
        const auto& f = frames[fn];
        if (f.empty()) break;

        if (fn == 9) {
            if (!frameNecesitaTiro(9)) {
                for (int t : f) acumulado += t;
                scores[fn] = acumulado;
            }
            break;
        }

        if (f[0] == 10) {
            if (idx + 2 < todos.size()) {
                acumulado += 10 + todos[idx + 1] + todos[idx + 2];
                scores[fn] = acumulado;
            }
            idx += 1;
        } else if (f.size() == 2) {
            if (f[0] + f[1] == 10) {
                // Un spare puntua 10 + los dos siguientes tiros
                if (idx + 3 < todos.size()) {
                    acumulado += 10 + todos[idx + 2] + todos[idx + 3];
                    scores[fn] = acumulado;
                }
            } else {
                acumulado += f[0] + f[1];
                scores[fn] = acumulado;
            }
            idx += 2;
        } else {
            break;
        }
    }
    return scores;
}

int Jugador::total() const {
    auto s = calcularPuntuacion();
    for (int i = 9; i >= 0; i--)
        if (s[i] >= 0) return s[i];
    return 0;
}

BowlingGame::BowlingGame(const std::vector<std::string>& nombres) {
    reiniciar(nombres);
}

void BowlingGame::reiniciar(const std::vector<std::string>& nombres) {
    jugadores_.clear();
    if (nombres.empty()) {
        for (int i = 1; i <= 4; i++)
            jugadores_.emplace_back("Jugador " + std::to_string(i));
    } else {
        for (const auto& n : nombres) jugadores_.emplace_back(n);
    }
    actual_ = 0;
    terminado_ = false;
}

bool BowlingGame::agregarTiro(int pinos, std::string& error) {
    if (terminado_) {
        error = "La partida ya termino";
        return false;
    }
    Jugador& j = jugadores_[actual_];
    error = j.validarTiro(pinos);
    if (!error.empty()) return false;

    int fi = j.frameActual();
    j.registrarTiro(pinos);

    if (!j.frameNecesitaTiro(fi)) {
        if (actual_ == (int)jugadores_.size() - 1) {
            actual_ = 0;
            bool todos = true;
            for (const auto& p : jugadores_)
                if (!p.terminado()) { todos = false; break; }
            if (todos) terminado_ = true;
        } else {
            actual_++;
        }
    }
    error.clear();
    return true;
}

const Jugador& BowlingGame::jugadorActual() const {
    return jugadores_[actual_];
}

int BowlingGame::indiceJugadorActual() const {
    return actual_;
}

bool BowlingGame::juegoTerminado() const {
    return terminado_;
}

const std::vector<Jugador>& BowlingGame::jugadores() const {
    return jugadores_;
}

int BowlingGame::pinosEnPieEsperados() const {
    if (terminado_) return -1;
    return jugadores_[actual_].pinosEnPieEsperados();
}

std::string formatearTiros(const std::vector<int>& tiros) {
    std::string r;
    int rack = 0;
    for (size_t i = 0; i < tiros.size(); i++) {
        if (i > 0) r += " ";
        if ((int)i > rack && tiros[i - 1] + tiros[i] == 10) {
            r += "/";
            rack = (int)i + 1;
        } else if (tiros[i] == 10 && (int)i == rack) {
            r += "X";
            rack = (int)i + 1;
        } else if (tiros[i] == 10) {
            r += "X";
            rack = (int)i + 1;
        } else if (tiros[i] == 0) {
            r += "-";
        } else {
            r += std::to_string(tiros[i]);
        }
    }
    return r;
}

int pinosMaximosPermitidos(const Jugador& j) {
    int fi = j.frameActual();
    if (fi < 0) return -1;
    const std::vector<int>& f = j.frames[fi];
    if (fi == 9) {
        if (f.size() == 1 && f[0] != 10) return 10 - f[0];
        if (f.size() == 2 && f[0] == 10 && f[1] != 10) return 10 - f[1];
        return 10;
    }
    if (f.size() == 1 && f[0] != 10) return 10 - f[0];
    return 10;
}

bool registrarTiro(BowlingGame& juego, int pinos, std::string& mensaje) {
    std::string error;
    if (juego.agregarTiro(pinos, error)) {
        mensaje = "Tiro registrado: " + std::to_string(pinos) + " pino(s)";
        return true;
    }
    int maximo = pinosMaximosPermitidos(juego.jugadorActual());
    std::string error2;
    if (maximo >= 0 && maximo != pinos && juego.agregarTiro(maximo, error2)) {
        mensaje = "Tiro ajustado a " + std::to_string(maximo) + " pino(s) (detectados " +
                  std::to_string(pinos) + ": " + error + ")";
        return true;
    }
    mensaje = "Tiro descartado: " + error;
    return false;
}

static void putCentrado(cv::Mat& img, const std::string& text, cv::Rect area,
                        double scale, cv::Scalar color, int thickness = 1) {
    int baseline = 0;
    auto size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
    cv::Point pos(area.x + (area.width - size.width) / 2,
                  area.y + (area.height + size.height) / 2 - baseline);
    cv::putText(img, text, pos, cv::FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv::LINE_AA);
}

// Simbolos de cada caja (uno por tiro) a partir del texto formateado ("7 /", "X 5 /"...)
static std::vector<std::string> dividirSimbolos(const std::string& texto) {
    std::vector<std::string> out;
    std::string actual;
    for (char c : texto) {
        if (c == ' ') {
            if (!actual.empty()) {
                out.push_back(actual);
                actual.clear();
            }
        } else {
            actual += c;
        }
    }
    if (!actual.empty()) out.push_back(actual);
    return out;
}

cv::Mat dibujarMarcador(const BowlingGame& juego, int pinosEnPie, int ultimoTiro, double escala) {
    if (escala <= 0.0) escala = 1.0;
    auto px = [escala](double v) { return (int)std::lround(v * escala); };
    auto fuente = [escala](double v) { return v * escala; };
    auto grosor = [escala](int t) { return std::max(1, (int)std::lround(t * escala)); };

    const int ancho_nom = px(155), ancho_frame = px(78), ancho_total = px(85);
    const int alto_header = px(45), alto_fila = px(86), margen = px(10), alto_pie = px(56);

    // Paleta tomada del banner del evento (BGR): azul electrico, verde lima y blanco
    const cv::Scalar FONDO(120, 24, 24);
    const cv::Scalar CABECERA(220, 52, 52);
    const cv::Scalar CELDA_A(192, 40, 40);
    const cv::Scalar CELDA_B(172, 34, 34);
    const cv::Scalar CELDA_ACTUAL(224, 60, 60);
    const cv::Scalar LINEA(230, 110, 110);
    const cv::Scalar TEXTO(255, 255, 255);
    const cv::Scalar TEXTO_SUAVE(255, 200, 200);
    const cv::Scalar ACENTO(0, 251, 161);
    const cv::Scalar CAJA(90, 20, 20);
    const cv::Scalar CAJA_BORDE(220, 130, 130);
    const cv::Scalar MARCO(240, 150, 150);

    int ancho = margen * 2 + ancho_nom + 10 * ancho_frame + ancho_total;
    // Una fila por jugador (de 4 a 8): el alto de la tabla crece con ellos
    const int nFilas = std::max(4, std::min(8, (int)juego.jugadores().size()));
    int alto = margen * 2 + alto_header + nFilas * alto_fila + alto_pie;

    cv::Mat img(alto, ancho, CV_8UC3, FONDO);
    auto& jugadores = juego.jugadores();

    int x0 = margen, y0 = margen;
    int anchoTabla = ancho - 2 * margen;
    int altoTabla = alto_header + nFilas * alto_fila;

    // Marco exterior frio, como la tabla de la imagen
    cv::rectangle(img, cv::Rect(1, 1, ancho - 2, alto - 2), MARCO, grosor(2));

    cv::rectangle(img, cv::Rect(x0, y0, anchoTabla, alto_header), CABECERA, cv::FILLED);

    putCentrado(img, "JUGADOR", cv::Rect(x0, y0, ancho_nom, alto_header), fuente(0.5),
                TEXTO, grosor(1));

    int x = x0 + ancho_nom;
    for (int i = 0; i < 10; i++) {
        putCentrado(img, std::to_string(i + 1), cv::Rect(x, y0, ancho_frame, alto_header),
                    fuente(0.52), TEXTO_SUAVE, grosor(1));
        x += ancho_frame;
    }
    putCentrado(img, "TOTAL", cv::Rect(x, y0, ancho_total, alto_header), fuente(0.5),
                TEXTO, grosor(1));

    for (int row = 0; row < nFilas && row < (int)jugadores.size(); row++) {
        int y = y0 + alto_header + row * alto_fila;
        bool esActual = (row == juego.indiceJugadorActual()) && !juego.juegoTerminado();
        cv::rectangle(img, cv::Rect(x0, y, anchoTabla, alto_fila),
                      esActual ? CELDA_ACTUAL : (row % 2 == 0 ? CELDA_A : CELDA_B), cv::FILLED);
        if (esActual) {
            cv::rectangle(img, cv::Rect(x0, y, px(5), alto_fila), ACENTO, cv::FILLED);
        }

        const Jugador& j = jugadores[row];
        std::string nom = esActual ? "> " + j.nombre : j.nombre;
        int baseNom = 0;
        cv::Size tamNom = cv::getTextSize(nom, cv::FONT_HERSHEY_SIMPLEX, fuente(0.52),
                                          grosor(1), &baseNom);
        cv::putText(img, nom, cv::Point(x0 + px(10), y + (alto_fila + tamNom.height) / 2),
                    cv::FONT_HERSHEY_SIMPLEX, fuente(0.52), esActual ? ACENTO : TEXTO,
                    grosor(1), cv::LINE_AA);

        auto scores = j.calcularPuntuacion();
        x = x0 + ancho_nom;
        for (int fi = 0; fi < 10; fi++) {
            cv::Rect celda(x, y, ancho_frame, alto_fila);
            std::vector<std::string> simbolos = dividirSimbolos(formatearTiros(j.frames[fi]));

            // Cajitas de tiros arriba (2 por frame, 3 en el frame 10)
            int numCajas = (fi == 9) ? 3 : 2;
            int cajaW = (fi == 9) ? px(17) : px(21);
            int cajaH = px(20), gap = px(2);
            int baseNum = 0;
            cv::Size tamNum = cv::getTextSize("0", cv::FONT_HERSHEY_SIMPLEX, fuente(0.52),
                                              grosor(1), &baseNum);
            int altoGrupo = cajaH + px(2) + tamNum.height;
            int inicioX = celda.x + (celda.width - (numCajas * cajaW + (numCajas - 1) * gap)) / 2;
            int inicioY = celda.y + (alto_fila - altoGrupo) / 2;
            for (int c = 0; c < numCajas; c++) {
                cv::Rect caja(inicioX + c * (cajaW + gap), inicioY, cajaW, cajaH);
                cv::rectangle(img, caja, CAJA, cv::FILLED);
                cv::rectangle(img, caja, CAJA_BORDE, grosor(1));
                if (c < (int)simbolos.size()) {
                    putCentrado(img, simbolos[c], caja, fuente(fi == 9 ? 0.38 : 0.44),
                                ACENTO, grosor(1));
                }
            }

            // Puntuacion acumulada debajo de las cajitas
            if (scores[fi] >= 0) {
                int inicioScore = inicioY + cajaH + px(2);
                cv::Rect areaScore(celda.x, inicioScore, celda.width, tamNum.height);
                putCentrado(img, std::to_string(scores[fi]), areaScore, fuente(0.52),
                            TEXTO, grosor(1));
            }

            cv::line(img, cv::Point(x, y), cv::Point(x, y + alto_fila), LINEA, grosor(1));
            // Resaltamos el frame en curso del jugador actual
            if (esActual && fi == j.frameActual()) {
                cv::rectangle(img, cv::Rect(x + px(2), y + px(2), ancho_frame - px(4), alto_fila - px(4)),
                              ACENTO, grosor(1));
            }
            x += ancho_frame;
        }

        putCentrado(img, std::to_string(j.total()), cv::Rect(x, y, ancho_total, alto_fila),
                    fuente(0.7), ACENTO, grosor(2));
        cv::line(img, cv::Point(x0, y + alto_fila), cv::Point(x0 + anchoTabla, y + alto_fila),
                 LINEA, grosor(1));
    }

    // Separadores verticales de las columnas de nombre y total
    cv::line(img, cv::Point(x0 + ancho_nom, y0), cv::Point(x0 + ancho_nom, y0 + altoTabla),
             LINEA, grosor(1));
    cv::line(img, cv::Point(x0 + ancho_nom + 10 * ancho_frame, y0),
             cv::Point(x0 + ancho_nom + 10 * ancho_frame, y0 + altoTabla), LINEA, grosor(1));

    // Linea de acento bajo la cabecera
    cv::line(img, cv::Point(x0, y0 + alto_header), cv::Point(x0 + anchoTabla, y0 + alto_header),
             ACENTO, grosor(2));

    // Pie de pagina: turno actual, pinos en pie y puntos del ultimo tiro
    int yPie = y0 + altoTabla;
    cv::rectangle(img, cv::Rect(x0, yPie, anchoTabla, alto_pie), CABECERA, cv::FILLED);
    cv::line(img, cv::Point(x0, yPie), cv::Point(x0 + anchoTabla, yPie), LINEA, grosor(1));

    std::string izquierda;
    if (juego.juegoTerminado()) {
        izquierda = "PARTIDA TERMINADA  (R para reiniciar)";
    } else {
        const Jugador& ja = juego.jugadorActual();
        izquierda = "Turno: " + ja.nombre;
        if (ja.frameActual() >= 0) {
            izquierda += "   Frame " + std::to_string(ja.frameActual() + 1) +
                         "   Tiro " + std::to_string(ja.numeroTiro());
        }
    }
    int basePie = 0;
    cv::Size tamPie = cv::getTextSize(izquierda, cv::FONT_HERSHEY_SIMPLEX, fuente(0.5),
                                      grosor(1), &basePie);
    int yPieTexto = yPie + (alto_pie + tamPie.height) / 2;
    cv::putText(img, izquierda, cv::Point(x0 + px(10), yPieTexto), cv::FONT_HERSHEY_SIMPLEX,
                fuente(0.5), TEXTO, grosor(1), cv::LINE_AA);

    std::string derecha;
    if (ultimoTiro >= 0) derecha = "Ultimo tiro: " + std::to_string(ultimoTiro) + " pinos";
    if (pinosEnPie >= 0) derecha += "   Pinos en pie: " + std::to_string(pinosEnPie);
    if (!derecha.empty()) {
        int baseline = 0;
        int tam = cv::getTextSize(derecha, cv::FONT_HERSHEY_SIMPLEX, fuente(0.5), grosor(1), &baseline).width;
        cv::putText(img, derecha, cv::Point(x0 + anchoTabla - tam - px(10), yPieTexto),
                    cv::FONT_HERSHEY_SIMPLEX, fuente(0.5), ACENTO, grosor(1), cv::LINE_AA);
    }

    return img;
}
