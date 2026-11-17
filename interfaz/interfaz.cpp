#ifdef _WIN32
#define NOMINMAX
#endif

#include "interfaz/interfaz.h"

#include "configuracion.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#ifdef _WIN32
#include <Windows.h>
#pragma comment(lib, "user32.lib")
#endif

namespace {

constexpr int ANCHO_CAMARA = 1280;
constexpr int ALTO_CAMARA = 960;
constexpr int ANCHO_CONFIG = 900;
constexpr int ALTO_CONFIG = 300;

#ifdef _WIN32
struct InfoMonitor {
    cv::Rect rect;       // area completa del monitor (para el fullscreen)
    cv::Rect trabajo;    // area util (sin barra de tareas)
    bool primario = false;
};

std::vector<InfoMonitor> enumerarMonitores() {
    std::vector<InfoMonitor> monitores;
    EnumDisplayMonitors(
        nullptr, nullptr,
        [](HMONITOR hMon, HDC, LPRECT, LPARAM lp) -> BOOL {
            MONITORINFO mi{};
            mi.cbSize = sizeof(MONITORINFO);
            if (GetMonitorInfoW(hMon, &mi)) {
                auto* v = reinterpret_cast<std::vector<InfoMonitor>*>(lp);
                InfoMonitor m;
                m.rect = cv::Rect(mi.rcMonitor.left, mi.rcMonitor.top,
                                  mi.rcMonitor.right - mi.rcMonitor.left,
                                  mi.rcMonitor.bottom - mi.rcMonitor.top);
                m.trabajo = cv::Rect(mi.rcWork.left, mi.rcWork.top,
                                     mi.rcWork.right - mi.rcWork.left,
                                     mi.rcWork.bottom - mi.rcWork.top);
                m.primario = (mi.dwFlags & MONITORINFOF_PRIMARY) != 0;
                v->push_back(m);
            }
            return TRUE;
        },
        reinterpret_cast<LPARAM>(&monitores));
    return monitores;
}
#endif

} // namespace

RepartoPantallas repartirPantallas() {
#ifdef _WIN32
    // Fullscreen nitido: trabajamos en pixeles reales de la pantalla (sin
    // que Windows escale la ventana por el aumento de DPI del 125%).
    SetProcessDPIAware();
#endif

    RepartoPantallas reparto;
    reparto.monitorTablero = cv::Rect(0, 0, 1920, 1080);
    reparto.monitorCamara = reparto.monitorTablero;

#ifdef _WIN32
    reparto.monitorTablero =
        cv::Rect(0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN));
    reparto.monitorCamara = reparto.monitorTablero;

    const std::vector<InfoMonitor> monitores = enumerarMonitores();
    for (const auto& m : monitores) {
        if (m.primario) reparto.monitorCamara = m.trabajo;
    }
    for (const auto& m : monitores) {
        if (!m.primario) {
            reparto.monitorTablero = m.rect;
            reparto.tableroEnSecundario = true;
            break;
        }
    }
    if (!monitores.empty() && !reparto.tableroEnSecundario) {
        reparto.monitorTablero = monitores[0].rect;   // un solo monitor
    }
#endif

    if (reparto.monitorTablero.width < 800 || reparto.monitorTablero.height < 600) {
        reparto.monitorTablero = cv::Rect(0, 0, 1920, 1080);
    }
    if (reparto.monitorCamara.width < 640 || reparto.monitorCamara.height < 480) {
        reparto.monitorCamara = cv::Rect(0, 0, 1536, 864);
    }
    return reparto;
}

std::vector<std::string> pedirConfiguracionInterfaz(const cv::Rect& monitor) {
    const std::string VENTANA_CFG = "Configuracion - Boliche IA";

    // Mismos colores que el marcador (marcador.cpp)
    const cv::Scalar FONDO(120, 24, 24);
    const cv::Scalar CABECERA(220, 52, 52);
    const cv::Scalar CAJA(90, 20, 20);
    const cv::Scalar CAJA_BORDE(220, 130, 130);
    const cv::Scalar MARCO(240, 150, 150);
    const cv::Scalar TEXTO(255, 255, 255);
    const cv::Scalar TEXTO_SUAVE(255, 200, 200);
    const cv::Scalar ACENTO(0, 251, 161);
    const cv::Scalar AVISO(224, 60, 60);

    cv::namedWindow(VENTANA_CFG, cv::WINDOW_NORMAL);
    cv::resizeWindow(VENTANA_CFG, ANCHO_CONFIG, ALTO_CONFIG);
    // Centrada en el monitor donde aparece
    cv::moveWindow(VENTANA_CFG, monitor.x + (monitor.width - ANCHO_CONFIG) / 2,
                   monitor.y + (monitor.height - ALTO_CONFIG) / 2);
#ifdef _WIN32
    // Se trae al frente una sola vez para que el usuario pueda escribir
    HWND hCfg = FindWindowW(nullptr, L"Configuracion - Boliche IA");
    if (hCfg) SetForegroundWindow(hCfg);
#endif

    std::vector<std::string> nombres;
    std::string texto, error, nombreTemporal;
    int paso = 0;             // 0 = cantidad de jugadores, 1..N = nombre del jugador
    int cantidad = MIN_JUGADORES;
    bool confirmando = false; // el nombre escrito esta esperando confirmacion

    while (true) {
        cv::Mat lienzo(ALTO_CONFIG, ANCHO_CONFIG, CV_8UC3, FONDO);

        // Marco exterior y cabecera, como el marcador
        cv::rectangle(lienzo, cv::Rect(2, 2, ANCHO_CONFIG - 4, ALTO_CONFIG - 4), MARCO, 2);
        cv::rectangle(lienzo, cv::Rect(20, 20, ANCHO_CONFIG - 40, 48), CABECERA, cv::FILLED);
        cv::putText(lienzo, "CONFIGURACION DE LA PARTIDA", cv::Point(36, 52),
                    cv::FONT_HERSHEY_SIMPLEX, 0.62, TEXTO, 1, cv::LINE_AA);

        std::string pregunta;
        std::string contenido;
        std::string ayuda = "ENTER: confirmar    BACKSPACE: borrar    ESC: cancelar";

        if (paso == 0) {
            pregunta = "Cantidad de jugadores? (" + std::to_string(MIN_JUGADORES) +
                       " - " + std::to_string(MAX_JUGADORES) + ")";
            contenido = texto + "|";
        } else if (!confirmando) {
            pregunta = "Nombre del jugador " + std::to_string(paso) + ":";
            contenido = texto + "|";
        } else {
            pregunta = "Confirme el nombre del jugador " + std::to_string(paso) + ":";
            contenido = nombreTemporal;
            ayuda = "ENTER: confirmar el nombre    B: escribir otro nombre";
        }

        cv::putText(lienzo, pregunta, cv::Point(40, 120), cv::FONT_HERSHEY_SIMPLEX, 0.85,
                    TEXTO, 2, cv::LINE_AA);

        // Caja de texto
        cv::rectangle(lienzo, cv::Rect(38, 145, ANCHO_CONFIG - 76, 58), CAJA, cv::FILLED);
        cv::rectangle(lienzo, cv::Rect(38, 145, ANCHO_CONFIG - 76, 58), CAJA_BORDE, 2);
        cv::putText(lienzo, contenido, cv::Point(54, 185), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                    ACENTO, 2, cv::LINE_AA);

        if (!error.empty()) {
            cv::putText(lienzo, error, cv::Point(40, 238), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        AVISO, 2, cv::LINE_AA);
        }
        cv::putText(lienzo, ayuda, cv::Point(40, 272), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    TEXTO_SUAVE, 1, cv::LINE_AA);

        cv::imshow(VENTANA_CFG, lienzo);

        int tecla = cv::waitKey(30);
        if (tecla == 27) {                    // ESC: cancelar
            cv::destroyWindow(VENTANA_CFG);
            std::cout << "Configuracion cancelada: " << MIN_JUGADORES
                      << " jugadores por defecto." << std::endl;
            std::vector<std::string> porDefecto;
            for (int i = 1; i <= MIN_JUGADORES; ++i)
                porDefecto.push_back("Jugador " + std::to_string(i));
            return porDefecto;
        }

        if (confirmando) {
            // Solo se acepta la confirmacion o volver a escribir el nombre
            if (tecla == 13 || tecla == 10 || tecla == 's' || tecla == 'S') {
                nombres.push_back(nombreTemporal);
                if ((int)nombres.size() >= cantidad) break;
                paso++;
                confirmando = false;
                nombreTemporal.clear();
                texto.clear();
                error.clear();
            } else if (tecla == 'b' || tecla == 'B' || tecla == 8 || tecla == 127) {
                confirmando = false;
                texto = nombreTemporal;      // se edita el nombre anterior
                nombreTemporal.clear();
                error.clear();
            }
            continue;
        }

        if (tecla == 8 || tecla == 127) {     // BACKSPACE
            if (!texto.empty()) texto.pop_back();
            continue;
        }
        if (tecla == 13 || tecla == 10) {     // ENTER
            if (paso == 0) {
                int n = 0;
                try {
                    n = std::stoi(texto);
                } catch (const std::exception&) {
                    n = 0;
                }
                if (n < MIN_JUGADORES || n > MAX_JUGADORES) {
                    error = "La cantidad debe ser un numero entre " +
                            std::to_string(MIN_JUGADORES) + " y " +
                            std::to_string(MAX_JUGADORES) + ".";
                    continue;
                }
                cantidad = n;
                paso = 1;
                texto.clear();
                error.clear();
            } else {
                if (texto.empty()) {
                    error = "El nombre no puede estar vacio.";
                    continue;
                }
                // Se pide confirmar el nombre antes de tomarlo
                nombreTemporal = texto;
                texto.clear();
                confirmando = true;
                error.clear();
            }
            continue;
        }
        if (tecla >= 32 && tecla < 127) {     // caracter normal
            texto += (char)tecla;
            error.clear();
        }
    }

    cv::destroyWindow(VENTANA_CFG);
    std::cout << "Partida para " << cantidad << " jugadores." << std::endl;
    return nombres;
}

void VentanasJuego::crear(const RepartoPantallas& reparto) {
    anchoTab_ = reparto.monitorTablero.width;
    altoTab_ = reparto.monitorTablero.height;
    ventanaTablero_ = "Tablero - Boliche IA";
    ventanaCamara_ = "Camara - Boliche IA";

    // Tablero a pantalla completa en el monitor elegido. Primero lo movemos a
    // ese monitor para que OpenCV aplique el fullscreen sobre el correcto.
    cv::namedWindow(ventanaTablero_, cv::WINDOW_NORMAL);
    cv::moveWindow(ventanaTablero_, reparto.monitorTablero.x, reparto.monitorTablero.y);
    cv::resizeWindow(ventanaTablero_, anchoTab_, altoTab_);
    try {
        cv::setWindowProperty(ventanaTablero_, cv::WND_PROP_FULLSCREEN,
                              cv::WINDOW_FULLSCREEN);
    } catch (const cv::Exception& e) {
        std::cout << "No se pudo activar pantalla completa: " << e.what() << std::endl;
    }

    lienzoTab_ = cv::Mat(altoTab_, anchoTab_, CV_8UC3, cv::Scalar(120, 24, 24));

    // Camara: ventana normal en el monitor principal
    const int ventanaCamW =
        std::min(ANCHO_CAMARA, std::max(320, reparto.monitorCamara.width - 40));
    const int ventanaCamH =
        std::min(ALTO_CAMARA, std::max(240, reparto.monitorCamara.height - 40));
    cv::namedWindow(ventanaCamara_, cv::WINDOW_NORMAL);
    cv::resizeWindow(ventanaCamara_, ventanaCamW, ventanaCamH);
    cv::moveWindow(ventanaCamara_, reparto.monitorCamara.x + 24,
                   reparto.monitorCamara.y + 48);

    lienzoCam_ = cv::Mat(ALTO_CAMARA, ANCHO_CAMARA, CV_8UC3, cv::Scalar(18, 18, 22));
    ajustarEscala(MIN_JUGADORES);
}

void VentanasJuego::ajustarEscala(int filasMarcador) {
    filasMarcador_ = std::max(MIN_JUGADORES, filasMarcador);

    // El marcador se escala para que quepa completo segun la cantidad de
    // jugadores (una fila por jugador) y quede centrado en la pantalla.
    const double anchoBaseMarcador = 1040.0;
    const double altoBaseMarcador = 121.0 + 86.0 * filasMarcador_;
    escalaTablero_ = (anchoTab_ - 32.0) / anchoBaseMarcador;
    const double escalaPorAlto = (altoTab_ - 32.0) / altoBaseMarcador;
    if (escalaPorAlto < escalaTablero_) escalaTablero_ = escalaPorAlto;
    if (escalaTablero_ <= 0.0) escalaTablero_ = 1.0;
}

void VentanasJuego::mostrarTablero(const BowlingGame& juego, int pinosEnPie, int ultimoTiro) {
    lienzoTab_.setTo(cv::Scalar(120, 24, 24));

    cv::Mat tablero = dibujarMarcador(juego, pinosEnPie, ultimoTiro, escalaTablero_);
    const int xTablero = (anchoTab_ - tablero.cols) / 2;
    const int yTablero = std::max(0, (altoTab_ - tablero.rows) / 2);   // centrado
    if (xTablero >= 0 && yTablero >= 0 && xTablero + tablero.cols <= anchoTab_ &&
        yTablero + tablero.rows <= altoTab_) {
        tablero.copyTo(lienzoTab_(cv::Rect(xTablero, yTablero, tablero.cols, tablero.rows)));
    }
    cv::imshow(ventanaTablero_, lienzoTab_);
}

void VentanasJuego::mostrarCamara(const cv::Mat& frame,
                                  const std::vector<Deteccion>& detecciones,
                                  const std::vector<ValidadorPinos::Pino>& pinos,
                                  const cv::Rect& barraDetectada,
                                  const BowlingGame& juego,
                                  const EstadoVista& estado,
                                  const std::string& dispositivo) {
    if (frame.empty()) return;

    const double escVista = std::min(ANCHO_CAMARA / (double)frame.cols,
                                     ALTO_CAMARA / (double)frame.rows);
    const int anchoVista = std::max(1, (int)std::lround(frame.cols * escVista));
    const int altoVista = std::max(1, (int)std::lround(frame.rows * escVista));
    const int dxVista = (ANCHO_CAMARA - anchoVista) / 2;
    const int dyVista = (ALTO_CAMARA - altoVista) / 2;

    lienzoCam_.setTo(cv::Scalar(18, 18, 22));
    cv::resize(frame, vistaCam_, cv::Size(anchoVista, altoVista));
    vistaCam_.copyTo(lienzoCam_(cv::Rect(dxVista, dyVista, anchoVista, altoVista)));

    const float sx = anchoVista / (float)frame.cols;
    const float sy = altoVista / (float)frame.rows;
    const double fEtiqueta = 0.7 * (anchoVista / 1024.0);
    const double fEtiquetaValidando = 0.6 * (anchoVista / 1024.0);

    // Estado en una franja oscura arriba (deja libres los pinos de abajo)
    const Jugador& actual = juego.jugadorActual();
    std::string infoTurno = "Turno: " + actual.nombre;
    if (actual.frameActual() >= 0) {
        infoTurno += "   Frame " + std::to_string(actual.frameActual() + 1) +
                     "   Tiro " + std::to_string(actual.numeroTiro());
    }
    std::string infoPinos = "Pinos en pie: " + std::to_string(estado.pinosEnPie) +
                            "/10   validando: " + std::to_string(estado.validando) +
                            "   (esperados: " +
                            (estado.esperados >= 0 ? std::to_string(estado.esperados) : "-") +
                            ")";
    if (CAMARA_SIEMPRE_BLANCO_Y_NEGRO || estado.luzAzul)
        infoPinos += "   franjas: " + std::to_string(estado.franjasRojas);
    std::string infoTiro = "Ultimo tiro: " +
                           (estado.ultimoTiro >= 0
                                ? std::to_string(estado.ultimoTiro) + " pinos caidos"
                                : "-");
    std::string infoIA = "IA: " + std::to_string(estado.msIA) + " ms (" +
                         std::to_string((int)(1000.0 / std::max(0.01, estado.msIA))) +
                         " FPS) | " + dispositivo;
    if (CAMARA_SIEMPRE_BLANCO_Y_NEGRO) infoIA += " | B/N ON (morfologia)";
    else if (estado.luzAzul) infoIA += " | luz azul ON";

    cv::Rect franja(0, 0, ANCHO_CAMARA, 304);
    cv::Mat zona = lienzoCam_(franja);
    cv::Mat oscuro(zona.size(), CV_8UC3, cv::Scalar(8, 10, 14));
    cv::addWeighted(oscuro, 0.62, zona, 0.38, 0, zona);

    const int xTexto = 24, yTexto = 40, dyLinea = 44;
    cv::putText(lienzoCam_, infoIA, cv::Point(xTexto, yTexto), cv::FONT_HERSHEY_SIMPLEX,
                0.75, cv::Scalar(170, 170, 190), 2);
    cv::putText(lienzoCam_, infoTurno, cv::Point(xTexto, yTexto + dyLinea),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    cv::putText(lienzoCam_, infoPinos, cv::Point(xTexto, yTexto + 2 * dyLinea),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
    cv::putText(lienzoCam_, infoTiro, cv::Point(xTexto, yTexto + 3 * dyLinea),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

    bool escrito = false;
    if (estado.pausaMaquina) {
        cv::putText(lienzoCam_, "Pausa: la maquina recoloca los pinos",
                    cv::Point(xTexto, yTexto + 4 * dyLinea), cv::FONT_HERSHEY_SIMPLEX,
                    0.8, cv::Scalar(0, 200, 255), 2);
        escrito = true;
    } else if (estado.tiroEnCurso) {
        // La barra es el unico cierre del tiro: cuenta lo que cayo desde el ultimo barrido
        cv::putText(lienzoCam_, "Tiro en curso: esperando la barra para cerrar",
                    cv::Point(xTexto, yTexto + 4 * dyLinea), cv::FONT_HERSHEY_SIMPLEX,
                    0.8, cv::Scalar(255, 160, 0), 2);
        escrito = true;
    }
    if (!escrito) {
        cv::putText(lienzoCam_, "Esperando bola o caida de pinos",
                    cv::Point(xTexto, yTexto + 4 * dyLinea), cv::FONT_HERSHEY_SIMPLEX,
                    0.8, cv::Scalar(140, 140, 150), 2);
    }
    if (estado.hayBarra) {
        cv::putText(lienzoCam_, "BARRA DETECTADA: cambio de tiro",
                    cv::Point(xTexto, yTexto + 5 * dyLinea), cv::FONT_HERSHEY_SIMPLEX,
                    0.8, cv::Scalar(0, 0, 255), 2);
    }

    for (const Deteccion& d : detecciones) {
        if (d.clase == 1) continue;   // los pinos se dibujan segun su validacion
        cv::Rect box((int)(d.box.x * sx) + dxVista, (int)(d.box.y * sy) + dyVista,
                     (int)(d.box.width * sx), (int)(d.box.height * sy));
        cv::Scalar color;
        std::string nombre;
        if (d.clase == 0) {
            color = cv::Scalar(255, 0, 0);          // Azul
            nombre = "Bola";
        } else {
            color = cv::Scalar(0, 0, 255);          // Rojo
            nombre = "Barra Barredora";
        }
        cv::rectangle(lienzoCam_, box, color, 3);
        std::string label = nombre + ": " + std::to_string(int(d.confianza * 100)) + "%";
        cv::putText(lienzoCam_, label, cv::Point(box.x, std::max(24, box.y - 8)),
                    cv::FONT_HERSHEY_SIMPLEX, fEtiqueta, color, 2);
    }

    // Barra vista por color (la IA no la reconoce): se dibuja en rojo
    if (!barraDetectada.empty()) {
        cv::Rect box((int)(barraDetectada.x * sx) + dxVista,
                     (int)(barraDetectada.y * sy) + dyVista,
                     (int)(barraDetectada.width * sx),
                     (int)(barraDetectada.height * sy));
        cv::rectangle(lienzoCam_, box, cv::Scalar(0, 0, 255), 3);
        const std::string etiquetaBarra = CAMARA_SIEMPRE_BLANCO_Y_NEGRO
                                              ? "Barra Barredora (B/N)"
                                              : "Barra Barredora (color)";
        cv::putText(lienzoCam_, etiquetaBarra,
                    cv::Point(box.x, std::max(24, box.y - 8)),
                    cv::FONT_HERSHEY_SIMPLEX, fEtiqueta, cv::Scalar(0, 0, 255), 2);
    }

    // Pinos: verde = validado (0.5 s), amarillo = en validacion con su progreso
    for (const ValidadorPinos::Pino& p : pinos) {
        cv::Rect box((int)(p.box.x * sx) + dxVista, (int)(p.box.y * sy) + dyVista,
                     (int)(p.box.width * sx), (int)(p.box.height * sy));
        if (p.confirmado) {
            cv::rectangle(lienzoCam_, box, cv::Scalar(0, 255, 0), 3);
            cv::putText(lienzoCam_, "Pino", cv::Point(box.x, std::max(24, box.y - 8)),
                        cv::FONT_HERSHEY_SIMPLEX, fEtiqueta, cv::Scalar(0, 255, 0), 2);
        } else {
            cv::rectangle(lienzoCam_, box, cv::Scalar(0, 255, 255), 2);
            int pct = (int)(p.progreso * 100);
            cv::putText(lienzoCam_, "Validando " + std::to_string(pct) + "%",
                        cv::Point(box.x, std::max(24, box.y - 8)),
                        cv::FONT_HERSHEY_SIMPLEX, fEtiquetaValidando,
                        cv::Scalar(0, 255, 255), 2);
            // Barra de progreso debajo de la caja
            cv::Rect fondo(box.x, box.y + box.height + 4, box.width, 6);
            cv::rectangle(lienzoCam_, fondo, cv::Scalar(80, 80, 80), cv::FILLED);
            cv::Rect lleno(fondo.x, fondo.y, (int)(fondo.width * p.progreso), fondo.height);
            cv::rectangle(lienzoCam_, lleno, cv::Scalar(0, 255, 255), cv::FILLED);
        }
    }

    cv::imshow(ventanaCamara_, lienzoCam_);
}

void VentanasJuego::mostrarAvisoCamara() {
    if (lienzoCam_.empty()) return;
    cv::Mat aviso = lienzoCam_.clone();
    cv::putText(aviso, "CAMARA NO DISPONIBLE", cv::Point(40, 220),
                cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);
    cv::putText(aviso, "Reconectando...", cv::Point(40, 260),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 1);
    cv::imshow(ventanaCamara_, aviso);
}

void VentanasJuego::traerTableroAlFrente() const {
#ifdef _WIN32
    HWND hTab = FindWindowW(nullptr, L"Tablero - Boliche IA");
    if (hTab) {
        SetForegroundWindow(hTab);
        SetWindowPos(hTab, HWND_TOP, 0, 0, 0, 0,
                     SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
    }
#endif
}

int VentanasJuego::esperarTecla(int ms) const {
    return cv::waitKey(ms);
}

void VentanasJuego::cerrar() const {
    cv::destroyAllWindows();
}
