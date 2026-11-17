#include "aplicacion/aplicacion.h"

#include "configuracion.h"
#include "interfaz/interfaz.h"
#include "marcador.h"
#include "vision/detector_barra.h"
#include "vision/detector_yolo.h"
#include "vision/preprocesado.h"
#include "vision/seguimiento_tiros.h"
#include "vision/validador_pinos.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#include <dshow.h>
#include <oleauto.h>
#pragma comment(lib, "strmiids.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "oleaut32.lib")
#endif

namespace {

constexpr int CAMARA_ANCHO = 1280;
constexpr int CAMARA_ALTO = 720;

int64_t ahoraMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

#ifdef _WIN32
std::string aUtf8(const wchar_t* texto) {
    if (texto == nullptr) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, texto, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1) return {};
    std::string salida(n - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, texto, -1, salida.data(), n, nullptr, nullptr);
    return salida;
}

// Nombres de las camaras en el mismo orden en que OpenCV las numera: el
// indice de esta lista es el que espera VideoCapture.
std::vector<std::string> nombresCamaras() {
    std::vector<std::string> nombres;
    ICreateDevEnum* enumerador = nullptr;
    if (FAILED(CoCreateInstance(CLSID_SystemDeviceEnum, nullptr, CLSCTX_INPROC_SERVER,
                                IID_PPV_ARGS(&enumerador)))) {
        return nombres;
    }
    IEnumMoniker* monikers = nullptr;
    if (enumerador->CreateClassEnumerator(CLSID_VideoInputDeviceCategory, &monikers, 0) == S_OK &&
        monikers != nullptr) {
        IMoniker* moniker = nullptr;
        while (monikers->Next(1, &moniker, nullptr) == S_OK) {
            IPropertyBag* datos = nullptr;
            if (SUCCEEDED(moniker->BindToStorage(nullptr, nullptr, IID_PPV_ARGS(&datos)))) {
                VARIANT valor;
                VariantInit(&valor);
                if (SUCCEEDED(datos->Read(L"FriendlyName", &valor, nullptr)) &&
                    valor.vt == VT_BSTR) {
                    nombres.push_back(aUtf8(valor.bstrVal));
                }
                VariantClear(&valor);
                datos->Release();
            }
            moniker->Release();
        }
        monikers->Release();
    }
    enumerador->Release();
    return nombres;
}
#endif

// Las camaras virtuales (OBS, Snap, ManyCam...) muestran un logo en vez de la
// pista. Se prueban al final, solo si no hay ninguna camara fisica.
bool esCamaraVirtual(const std::string& nombre) {
    std::string n = nombre;
    std::transform(n.begin(), n.end(), n.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return n.find("obs") != std::string::npos || n.find("virtual") != std::string::npos ||
           n.find("snap camera") != std::string::npos ||
           n.find("manycam") != std::string::npos ||
           n.find("xsplit") != std::string::npos;
}

// Orden de camaras a probar: primero la forzada con BOLICHEIA_CAMARA, luego
// las fisicas y por ultimo las virtuales. Se calcula una sola vez.
const std::vector<int>& ordenCamaras() {
    static const std::vector<int> orden = [] {
        std::vector<int> fisicas, virtuales, todas;
#ifdef _WIN32
        const std::vector<std::string> nombres = nombresCamaras();
        for (int i = 0; i < (int)nombres.size(); ++i) {
            todas.push_back(i);
            const bool virtual_ = esCamaraVirtual(nombres[i]);
            (virtual_ ? virtuales : fisicas).push_back(i);
            std::cout << "Camara " << i << ": " << nombres[i]
                      << (virtual_ ? " (virtual, se prueba al final)" : "") << std::endl;
        }
#endif
        std::vector<int> resultado;
        const char* forzada = std::getenv("BOLICHEIA_CAMARA");
        if (forzada != nullptr) {
            try {
                resultado.push_back(std::stoi(forzada));
                std::cout << "Camara forzada por BOLICHEIA_CAMARA: " << forzada << std::endl;
            } catch (const std::exception&) {
                std::cout << "BOLICHEIA_CAMARA no es un numero: " << forzada << std::endl;
            }
        }
        auto agregar = [&resultado](const std::vector<int>& lista) {
            for (int i : lista)
                if (std::find(resultado.begin(), resultado.end(), i) == resultado.end())
                    resultado.push_back(i);
        };
        agregar(fisicas);
        agregar(virtuales);
        if (todas.empty()) {          // sin enumeracion: indices habituales
            for (int i = 0; i < 4; ++i) resultado.push_back(i);
        }
        return resultado;
    }();
    return orden;
}

// Se abre con DirectShow (DSHOW): el backend MSMF de OpenCV usa Media
// Foundation con un hilo asincrono y, cuando el dispositivo falla o se
// desconecta, deja el proceso sin respuesta (fallo fatal). Con DSHOW la
// perdida de la camara solo devuelve un error y se puede reconectar.
bool abrirCamara(cv::VideoCapture& cam) {
    for (int indice : ordenCamaras()) {
        try {
            cam.release();
            cam.open(indice, cv::CAP_DSHOW);
            if (!cam.isOpened()) cam.open(indice);
            if (!cam.isOpened()) continue;
            cam.set(cv::CAP_PROP_FRAME_WIDTH, CAMARA_ANCHO);
            cam.set(cv::CAP_PROP_FRAME_HEIGHT, CAMARA_ALTO);
            std::cout << "Camara abierta: indice " << indice << " a "
                      << (int)cam.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
                      << (int)cam.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
            return true;
        } catch (const cv::Exception& e) {
            std::cout << "Camara " << indice << ": " << e.what() << std::endl;
        }
    }
    return false;
}

} // namespace

void procesarVideo() {
#ifdef _WIN32
    // COM es necesario para enumerar las camaras por nombre (DirectShow).
    const HRESULT comIniciado = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
#endif
    // ------------------------------------------------------------------
    // El tablero debe ser lo primero que se ve: se crea y muestra antes de
    // cargar el modelo, preguntar la configuracion o conectar la camara.
    // ------------------------------------------------------------------
    const RepartoPantallas reparto = repartirPantallas();
    std::cout << "Tablero fullscreen en "
              << (reparto.tableroEnSecundario ? "monitor secundario" : "monitor principal")
              << ": " << reparto.monitorTablero.width << "x" << reparto.monitorTablero.height
              << " @ " << reparto.monitorTablero.x << "," << reparto.monitorTablero.y
              << " | camara en monitor principal (" << reparto.monitorCamara.width << "x"
              << reparto.monitorCamara.height << ")" << std::endl;

    VentanasJuego ventanas;
    ventanas.crear(reparto);

    int ultimoTiro = -1;          // pinos caidos en el ultimo tiro
    int pinosEnPie = 0;           // pinos detectados en pie ahora mismo
    BowlingGame juego;            // partida por defecto mientras se configura

    ventanas.mostrarTablero(juego, pinosEnPie, ultimoTiro);
    ventanas.esperarTecla(1);

    // ------------------------------------------------------------------
    // Preguntas de inicio en la interfaz: cantidad de jugadores (de 4 a 8)
    // y el nombre de cada uno.
    // ------------------------------------------------------------------
    const std::vector<std::string> nombres =
        pedirConfiguracionInterfaz(reparto.monitorCamara);
    juego = BowlingGame(nombres);
    ventanas.ajustarEscala((int)nombres.size());

    ventanas.mostrarTablero(juego, pinosEnPie, ultimoTiro);
    ventanas.esperarTecla(1);

    // ------------------------------------------------------------------
    // Motor de inferencia: ONNX Runtime con CUDA Execution Provider
    // (la inferencia corre en la RTX 3050, no en la grafica integrada)
    // ------------------------------------------------------------------
    DetectorYolo detector(BOLICHE_MODEL_PATH, true);
    const int ANCHO = detector.anchoEntrada();
    const int ALTO = detector.altoEntrada();

    // Estado del seguimiento y de la deteccion
    SeguimientoTiros seguimiento;
    ValidadorPinos validador;
    DetectorBarra detectorBarra;   // barra por movimiento (camara en B/N)
    bool hayBarra = false;         // la barra barredora esta en escena
    bool barraPorMovimiento = false; // la barra se vio por movimiento
    bool barraPorIA = false;       // la barra se acepto de la IA (clase 2)
    bool hayBola = false;          // la bola (cuadro azul) esta en escena
    bool luzAzul = false;          // hay dominante azul (luces LED de la pista)
    bool pausaMaquina = false;     // la maquina recoloca pinos tras un tiro
    int franjasRojas = 0;          // franjas del cuello vistas en el ultimo cuadro
    cv::Rect barraDetectada;       // barra vista por color (la IA no la detecta)
    double ms_ia_promedio = 0.0;

    // Diagnostico en vivo: cambios de estado que se registran en consola
    int contador_cuadros = 0;
    int pinosValidadosPrevios = -1;
    bool barraPrevia = false;
    bool bolaPrevia = false;

    // ------------------------------------------------------------------
    // Camara. Si no esta disponible al iniciar se reintenta sin cerrar la
    // aplicacion: el tablero debe seguir visible.
    // ------------------------------------------------------------------
    cv::VideoCapture cam;
    bool camaraViva = abrirCamara(cam);
    if (!camaraViva) {
        std::cout << "Camara: no disponible al iniciar. Se reintentara..." << std::endl;
    }
    int64_t ultimoIntento = 0;

    // ------------------------------------------------------------------
    // Captura de depuracion: se activa con la variable de entorno
    // BOLICHEIA_DEBUG=1. Guarda en la carpeta "depuracion" los cuadros del
    // tiro y del barrido (con 60 cuadros de pre-roll) para poder ajustar la
    // deteccion de la barra sin adivinar. No modifica la logica normal.
    // ------------------------------------------------------------------
    bool depurarCuadros = false;
    {
        const char* envDebug = std::getenv("BOLICHEIA_DEBUG");
        depurarCuadros = (envDebug != nullptr) && (std::string(envDebug) == "1");
        if (depurarCuadros) {
            std::filesystem::create_directories("depuracion");
            std::cout << "DEBUG: captura de cuadros activada (carpeta depuracion)" << std::endl;
        }
    }
    std::deque<std::vector<uchar>> bufferDebug;
    std::vector<uchar> jpgDebug;
    bool grabandoDebug = false;
    int64_t inicioDebugMs = 0;
    int64_t finDebugMs = 0;
    int capturaDebug = 0;

    // Tras registrar un tiro se dejan de buscar pinos durante el ciclo de la
    // maquina (MS_PAUSA_MAQUINA): si se cuentan los pinos que baja, el rack
    // queda inflado y el tiro siguiente se registra mal.
    int64_t ignorarPinosHastaMs = 0;
    int64_t reanudarPinosMs = 0;

    // Vuelca los cuadros acumulados a disco (tambien al salir, para no perder
    // la captura si se cierra la app antes del final del barrido).
    auto volcarCapturaDebug = [&](const char* motivo) {
        if (bufferDebug.empty()) return;
        char carpeta[64];
        std::snprintf(carpeta, sizeof(carpeta), "depuracion\\captura_%03d", ++capturaDebug);
        std::error_code ec;
        std::filesystem::create_directories(carpeta, ec);
        if (ec) {
            std::cout << "DEBUG: no se pudo crear " << carpeta << std::endl;
            return;
        }
        int n = 0;
        for (const std::vector<uchar>& j : bufferDebug) {
            char ruta[160];
            std::snprintf(ruta, sizeof(ruta), "%s\\cuadro_%04d.jpg", carpeta, n++);
            std::ofstream salidaJpg(ruta, std::ios::binary);
            salidaJpg.write(reinterpret_cast<const char*>(j.data()),
                            (std::streamsize)j.size());
        }
        std::cout << "DEBUG: " << n << " cuadros guardados en " << carpeta
                  << " (" << motivo << ")" << std::endl;
        bufferDebug.clear();
    };

    bool salir = false;
    cv::Mat frame;
    cv::Mat frameReducida;
    cv::Mat frameIA;
    std::vector<cv::Rect> franjas;
    std::vector<Deteccion> detecciones;

    while (!salir) {
        if (!camaraViva) {
            // Aviso en la ventana de la camara mientras se reintenta
            ventanas.mostrarTablero(juego, pinosEnPie, ultimoTiro);
            ventanas.mostrarAvisoCamara();

            const int t = ventanas.esperarTecla(200);
            if (t == 27) { salir = true; break; }

            const int64_t ahora = ahoraMs();
            if (ahora - ultimoIntento < 2000) continue;
            ultimoIntento = ahora;

            camaraViva = abrirCamara(cam);
            if (camaraViva) {
                detectorBarra.reiniciar();   // no hay cuadro anterior valido
                std::cout << "Camara reconectada." << std::endl;
            }
            continue;
        }

        // Lectura protegida del cuadro
        bool leido = false;
        try {
            leido = cam.read(frame) && !frame.empty();
        } catch (const cv::Exception& e) {
            std::cout << "Camara: error al leer el cuadro: " << e.what() << std::endl;
        }
        if (!leido) {
            std::cout << "Camara: se perdio la senal. Reconectando..." << std::endl;
            camaraViva = false;
            detectorBarra.reiniciar();
            continue;
        }
        // El driver puede cambiar el formato al reiniciar: se asegura CV_8UC3
        if (frame.type() != CV_8UC3) frame.convertTo(frame, CV_8UC3);

        // Filtro permanente: la camara trabaja siempre en blanco y negro
        if (CAMARA_SIEMPRE_BLANCO_Y_NEGRO) {
            cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
            cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
        }

        contador_cuadros++;
        const bool procesar_IA = (contador_cuadros % PROCESAR_CADA_N_CUADROS == 0);

        const int width = frame.cols;
        const int height = frame.rows;
        detecciones.clear();

        if (procesar_IA) {
            // La correccion se aplica sobre la imagen ya reducida al tamano de
            // entrada del modelo: mismo resultado con mucho menos coste.
            cv::resize(frame, frameReducida, cv::Size(ANCHO, ALTO), 0, 0, cv::INTER_LINEAR);
            const InfoGris info = analizarGris(frameReducida, CAMARA_SIEMPRE_BLANCO_Y_NEGRO);
            frameIA = prepararParaIA(frameReducida, luzAzul);

            // Barra barredora por movimiento entre cuadros. Con la camara en
            // blanco y negro es la unica senal fiable: la IA confunde la
            // banda oscura inferior y no ve las barridas rapidas.
            cv::Rect barraMovimiento;
            const bool barraMov = detectorBarra.actualizar(info.gris, barraMovimiento);

            // Franjas del cuello: en blanco y negro (Caso 2: Black-Hat) o bajo luz azul
            // se detectan y se usan como evidencia extra para validar el pino.
            franjas.clear();
            if (CAMARA_SIEMPRE_BLANCO_Y_NEGRO || luzAzul) {
                franjas = detectarFranjasCuello(frameReducida, info);
                const double sx = frame.cols / (double)ANCHO;
                const double sy = frame.rows / (double)ALTO;
                for (cv::Rect& f : franjas) {
                    f.x = (int)std::lround(f.x * sx);
                    f.y = (int)std::lround(f.y * sy);
                    f.width = (int)std::lround(f.width * sx);
                    f.height = (int)std::lround(f.height * sy);
                }
            }
            franjasRojas = (int)franjas.size();

            // Ejecutar la deteccion en la GPU
            detecciones = detector.detectar(frameIA, width, height, luzAzul);
            const double ms_ia = detector.ultimoMs();
            ms_ia_promedio = (ms_ia_promedio == 0.0) ? ms_ia
                                                      : (ms_ia_promedio * 0.9 + ms_ia * 0.1);

            // Cajas de pinos, bola en escena y barra
            std::vector<cv::Rect> cajasPinos;
            cajasPinos.reserve(detecciones.size());
            hayBola = false;
            hayBarra = barraMov;
            barraPorMovimiento = barraMov;
            barraPorIA = false;
            barraDetectada = cv::Rect();

            // El detector de movimiento trabaja en el tamano del modelo: se
            // lleva la caja a las coordenadas del cuadro real.
            if (barraMov) {
                const double bx = width / (double)ANCHO;
                const double by = height / (double)ALTO;
                barraDetectada = cv::Rect((int)std::lround(barraMovimiento.x * bx),
                                          (int)std::lround(barraMovimiento.y * by),
                                          (int)std::lround(barraMovimiento.width * bx),
                                          (int)std::lround(barraMovimiento.height * by));
            }

            for (const Deteccion& d : detecciones) {
                if (d.clase == 1) {
                    cajasPinos.push_back(d.box);
                } else if (d.clase == 0) {
                    hayBola = true;
                } else if (d.clase == 2 && barraIAPlausible(d.box, width, height)) {
                    // La IA solo aporta si la caja parece una barra real.
                    hayBarra = true;
                    barraPorIA = true;
                    if (barraDetectada.empty()) barraDetectada = d.box;
                }
            }

            // La barra real es azul con rayas magenta: si la IA no la ve,
            // se busca su firma de color en la imagen real. Con la camara en
            // blanco y negro esa firma no existe: manda el movimiento.
            if (!hayBarra && !CAMARA_SIEMPRE_BLANCO_Y_NEGRO) {
                cv::Rect barraColor;
                if (detectarBarra(frameReducida, barraColor, info)) {
                    hayBarra = true;
                    // El detector trabaja en el tamano del modelo: se lleva a
                    // las coordenadas del cuadro real para dibujarla bien.
                    const double bx = width / (double)ANCHO;
                    const double by = height / (double)ALTO;
                    barraDetectada = cv::Rect((int)std::lround(barraColor.x * bx),
                                              (int)std::lround(barraColor.y * by),
                                              (int)std::lround(barraColor.width * bx),
                                              (int)std::lround(barraColor.height * by));
                }
            }

            const int64_t ahora = ahoraMs();

            // Pausa posterior al tiro: la maquina baja a recolocar los pinos
            // que quedaron. Durante ese ciclo no se cuentan pinos.
            if (ignorarPinosHastaMs > 0 && ahora >= ignorarPinosHastaMs) {
                ignorarPinosHastaMs = 0;
                reanudarPinosMs = ahora;
            }
            pausaMaquina = (ahora < ignorarPinosHastaMs);

            // Solo cuentan los pinos validados durante 0.5 s, maximo 10.
            // Durante el barrido no se descartan (la barra los tapa).
            int pinosRack = -1;
            int pinosAhora = -1;
            int64_t msUltimoPino = -1;
            if (!pausaMaquina) {
                pinosEnPie = validador.actualizar(cajasPinos, ahora, hayBarra, franjas);

                // Pinos que se estan viendo de verdad ahora: es lo que mide el
                // rack cuando baja la barra (el conteo normal sigue 1.3 s con
                // los caidos).
                pinosRack = validador.enPieRecientes(ahora, MS_RACK);
                pinosAhora = validador.enPieRecientes(ahora, MS_CAIDA);
                const int64_t ultimaVista = validador.ultimaVistaMs();
                msUltimoPino = (ultimaVista > 0) ? (ahora - ultimaVista) : -1;

                // Al reanudar, los pinos que quedaron en pie se vuelven a ver:
                // hasta que pasen los 0.5 s de validacion se usa el conteo
                // valido en vez del conteo "visto hace poco" (que arranca en 0).
                if (reanudarPinosMs > 0 &&
                    (ahora - reanudarPinosMs) < ValidadorPinos::MS_VALIDACION) {
                    pinosRack = pinosEnPie;
                    pinosAhora = pinosEnPie;
                }
            }

            // Cambios de estado: sirven para ver en consola por que no registra
            if (pinosEnPie != pinosValidadosPrevios) {
                std::cout << "Pinos validados: " << pinosEnPie << "/10 (validando "
                          << validador.validando() << ", cajas IA " << cajasPinos.size()
                          << ")" << std::endl;
                pinosValidadosPrevios = pinosEnPie;
            }
            if (hayBarra != barraPrevia) {
                const char* fuente = barraPorMovimiento ? " (movimiento)"
                                     : barraPorIA     ? " (IA)"
                                                      : "";
                std::cout << "Barra: " << (hayBarra ? "visible" : "fuera")
                          << (barraPorMovimiento || barraPorIA ? fuente : "") << std::endl;
                barraPrevia = hayBarra;
            }
            if (hayBola != bolaPrevia) {
                std::cout << "Bola: " << (hayBola ? "visible" : "fuera") << std::endl;
                bolaPrevia = hayBola;
            }

            // Tope de pinos en pie segun el marcador: impide que pinos ya
            // caidos (detecciones fantasma) inflen el rack del tiro actual.
            int maxEnPie = juego.pinosEnPieEsperados();
            if (maxEnPie < 0 || maxEnPie > 10) maxEnPie = 10;
            int pinosCaidos = -1;
            if (!pausaMaquina) {
                pinosCaidos = seguimiento.actualizar(pinosEnPie, hayBola, hayBarra, ahora,
                                                     pinosRack, pinosAhora, msUltimoPino,
                                                     maxEnPie);
            }
            std::string cierre;
            if (seguimiento.tomarUltimoCierre(cierre)) std::cout << cierre << std::endl;
            if (pinosCaidos >= 0) {
                std::string mensaje;
                if (registrarTiro(juego, pinosCaidos, mensaje)) {
                    ultimoTiro = pinosCaidos;
                    ignorarPinosHastaMs = ahora + MS_PAUSA_MAQUINA;
                    reanudarPinosMs = 0;
                    std::cout << mensaje << std::endl;
                } else {
                    std::cout << mensaje << std::endl;
                }
            }

            // ----------------------------------------------------------
            // Captura de depuracion (BOLICHEIA_DEBUG=1). Guarda el cuadro
            // reducido en JPEG: 60 cuadros de pre-roll desde que aparece la
            // bola y luego hasta 4 s despues de ver la barra. Sirve para
            // revisar como se ve realmente el barrido y ajustar el detector.
            // ----------------------------------------------------------
            if (depurarCuadros) {
                jpgDebug.clear();
                cv::imencode(".jpg", frameReducida, jpgDebug,
                             {cv::IMWRITE_JPEG_QUALITY, 88});
                bufferDebug.push_back(std::move(jpgDebug));

                if (hayBola) {
                    if (!grabandoDebug) {
                        grabandoDebug = true;
                        inicioDebugMs = ahora;
                        std::cout << "DEBUG: captura iniciada" << std::endl;
                    }
                    finDebugMs = ahora + 12000;   // 12 s despues de la ultima bola
                } else if (grabandoDebug && hayBarra) {
                    finDebugMs = ahora + 4000;    // 4 s despues de ver la barra
                }
                if (grabandoDebug && (ahora - inicioDebugMs) > 25000) {
                    finDebugMs = ahora;           // tope por si la bola no sale
                }

                // Fuera de la grabacion solo se conservan los ultimos 60 cuadros.
                if (!grabandoDebug) {
                    while (bufferDebug.size() > 60) bufferDebug.pop_front();
                }

                if (grabandoDebug && ahora >= finDebugMs) {
                    volcarCapturaDebug("fin de captura");
                    grabandoDebug = false;
                }
            }

            if (contador_cuadros % 30 == 0) {
                std::cout << "IA: " << ms_ia_promedio << " ms/frame ("
                          << 1000.0 / ms_ia_promedio << " FPS) | " << detector.dispositivo()
                          << " | pinos validados: " << pinosEnPie << "/10"
                          << " | validando: " << validador.validando()
                          << " | cajas IA: " << detecciones.size()
                          << " (pino " << cajasPinos.size() << ", bola " << (hayBola ? 1 : 0)
                          << ", barra " << (hayBarra ? 1 : 0) << ")"
                          << " | rack: " << seguimiento.rack()
                          << " sinBarra: " << seguimiento.ultimoSinBarra()
                          << " pendiente: " << (seguimiento.pendiente() ? "si" : "no")
                          << " tiro: " << (seguimiento.tiroEnCurso() ? "si" : "no")
                          << " | franjas rojas: " << franjasRojas
                          << " | luz azul: "
                          << (luzAzul ? "si (correccion activa)" : "no");
                if (barraPorMovimiento) std::cout << " | barra movimiento";
                else if (!barraDetectada.empty()) std::cout << " | barra visual";
                std::cout << std::endl;
                if (seguimiento.pendiente()) {
                    std::cout << "   Barrido pendiente: antes=" << seguimiento.antesPendiente()
                              << " despues=" << seguimiento.despuesPendiente() << std::endl;
                }
                // Cajas de los pinos que sigue el validador (validados y en curso)
                for (const ValidadorPinos::Pino& p : validador.pines()) {
                    std::cout << "   pino " << (p.confirmado ? "OK " : ".. ")
                              << "(" << p.box.x << "," << p.box.y << ","
                              << p.box.width << "," << p.box.height << ")"
                              << " visto=" << p.vistoMs << "ms"
                              << " prog=" << (int)(p.progreso * 100) << "%" << std::endl;
                }
            }
        }

        // ------------------------------------------------------------------
        // Interfaz: tablero fullscreen y ventana de la camara con el HUD
        // ------------------------------------------------------------------
        EstadoVista estado;
        estado.pinosEnPie = pinosEnPie;
        estado.validando = validador.validando();
        estado.esperados = juego.pinosEnPieEsperados();
        estado.franjasRojas = franjasRojas;
        estado.ultimoTiro = ultimoTiro;
        estado.msIA = ms_ia_promedio;
        estado.hayBola = hayBola;
        estado.hayBarra = hayBarra;
        estado.tiroEnCurso = seguimiento.tiroEnCurso();
        estado.pausaMaquina = pausaMaquina;
        estado.luzAzul = luzAzul;

        ventanas.mostrarTablero(juego, pinosEnPie, ultimoTiro);
        ventanas.mostrarCamara(frame, detecciones, validador.pines(), barraDetectada,
                               juego, estado, detector.dispositivo());
        if (contador_cuadros == 1 || contador_cuadros == 30) {
            // Prioridad al tablero: debe ser lo primero que se vea al iniciar
            ventanas.traerTableroAlFrente();
        }

        const int tecla = ventanas.esperarTecla(1);
        if (tecla == 27) { salir = true; break; } // ESC para salir

        if (tecla == 'r' || tecla == 'R') {
            juego.reiniciar(nombres);
            ultimoTiro = -1;
            std::cout << "Partida reiniciada" << std::endl;
        } else if (tecla == 't' || tecla == 'T') {
            std::string mensaje;
            registrarTiro(juego, 10, mensaje);
            ultimoTiro = 10;
            std::cout << "Tiro manual (strike): " << mensaje << std::endl;
        } else if (tecla >= '0' && tecla <= '9') {
            std::string mensaje;
            registrarTiro(juego, tecla - '0', mensaje);
            ultimoTiro = tecla - '0';
            std::cout << "Tiro manual: " << mensaje << std::endl;
        }
    }

    if (depurarCuadros) volcarCapturaDebug("salida");

    try {
        cam.release();
    } catch (const cv::Exception& e) {
        std::cout << "Camara: " << e.what() << std::endl;
    }
    ventanas.cerrar();

#ifdef _WIN32
    if (SUCCEEDED(comIniciado)) CoUninitialize();
#endif
}
