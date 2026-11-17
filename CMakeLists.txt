cmake_minimum_required(VERSION 3.22)
project(Boliche_inteligente)

set(CMAKE_CXX_STANDARD 17)

# Indicamos la ruta de tu OpenCV actual
set(OpenCV_DIR "C:/opencv/build")

# Buscador automático de CMake
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Ruta absoluta del modelo ONNX: evita que falle si el directorio de trabajo
# del ejecutable no es la raíz del proyecto
set(BOLICHE_MODEL "${CMAKE_CURRENT_SOURCE_DIR}/bolos.onnx")

# ONNX Runtime (con CUDA Execution Provider) para ejecutar el modelo en la GPU dedicada
set(ORT_ROOT "C:/ort/onnxruntime/onnxruntime-win-x64-gpu_cuda12-1.31.0" CACHE PATH "ONNX Runtime root")
set(CUDA_BIN_DIR "C:/ort/cuda/bin" CACHE PATH "Carpeta con las DLL de CUDA (cuBLAS/cuDNN)")

if(NOT EXISTS "${ORT_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "No se encuentra ONNX Runtime en ${ORT_ROOT}. "
                        "Descarga onnxruntime-win-x64-gpu_cuda12 y ajusta ORT_ROOT.")
endif()

add_executable(Boliche_inteligente
        main.cpp
        marcador.cpp
        marcador.h
        configuracion.h
        aplicacion/aplicacion.cpp
        aplicacion/aplicacion.h
        interfaz/interfaz.cpp
        interfaz/interfaz.h
        vision/detector_barra.cpp
        vision/detector_barra.h
        vision/detector_yolo.cpp
        vision/detector_yolo.h
        vision/preprocesado.cpp
        vision/preprocesado.h
        vision/seguimiento_tiros.cpp
        vision/seguimiento_tiros.h
        vision/tipos.h
        vision/validador_pinos.cpp
        vision/validador_pinos.h)

target_include_directories(Boliche_inteligente PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${ORT_ROOT}/include")

target_compile_definitions(Boliche_inteligente PRIVATE
        BOLICHE_MODEL_PATH="${BOLICHE_MODEL}"
        BOLICHE_CUDA_BIN_PATH="${CUDA_BIN_DIR}")

# Enlazamos automáticamente todas las librerías
target_link_libraries(Boliche_inteligente ${OpenCV_LIBS} "${ORT_ROOT}/lib/onnxruntime.lib")

# Al ejecutar desde CLion, la carpeta de CUDA se añade al PATH del proceso
set_target_properties(Boliche_inteligente PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${CUDA_BIN_DIR}")

# Las DLL de OpenCV no se copian solas al carpeta de compilación, por eso el
# .exe terminaba con "exit code -1073741515 (0xC0000135)" (DLL no encontrada).
# Las copiamos junto al ejecutable al terminar la compilación.
set(OpenCV_BIN_CANDIDATES
        "${OpenCV_INSTALL_PATH}/x64/vc16/bin"
        "${OpenCV_INSTALL_PATH}/bin")

set(OpenCV_DLLS)
foreach(candidate ${OpenCV_BIN_CANDIDATES})
    if(IS_DIRECTORY "${candidate}")
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            file(GLOB found "${candidate}/opencv_world*d.dll")
        else()
            file(GLOB found "${candidate}/opencv_world*.dll")
            list(FILTER found EXCLUDE REGEX "d\\.dll$")
        endif()
        file(GLOB found_extra "${candidate}/opencv_videoio_ffmpeg*.dll")
        list(APPEND OpenCV_DLLS ${found} ${found_extra})
    endif()
endforeach()
list(REMOVE_DUPLICATES OpenCV_DLLS)

if(OpenCV_DLLS)
    message(STATUS "OpenCV DLLs a copiar junto al ejecutable: ${OpenCV_DLLS}")
    add_custom_command(TARGET Boliche_inteligente POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${OpenCV_DLLS}
            $<TARGET_FILE_DIR:Boliche_inteligente>
            COMMENT "Copiando DLLs de OpenCV junto al ejecutable")
else()
    message(WARNING "No se encontraron DLLs de OpenCV en ${OpenCV_BIN_CANDIDATES}. "
            "Agrega la carpeta de bin de OpenCV al PATH del sistema o al entorno del run de CLion.")
endif()

# DLLs de ONNX Runtime (incluye el proveedor CUDA) junto al ejecutable.
# Las DLL de cuBLAS/cuDNN (~1.6 GB) no se copian: se localizan en tiempo de
# ejecucion con SetDllDirectory (ver BOLICHE_CUDA_BIN_PATH en detector_yolo.cpp).
set(ORT_DLLS
        "${ORT_ROOT}/lib/onnxruntime.dll"
        "${ORT_ROOT}/lib/onnxruntime_providers_cuda.dll"
        "${ORT_ROOT}/lib/onnxruntime_providers_shared.dll")

# Barra inferior (banner) que se muestra al pie del tablero fullscreen
set(BOLICHE_BANNER "${CMAKE_CURRENT_SOURCE_DIR}/logo.png" CACHE FILEPATH "Imagen de barra inferior")

if(EXISTS "${BOLICHE_BANNER}")
    add_custom_command(TARGET Boliche_inteligente POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${BOLICHE_BANNER}"
            $<TARGET_FILE_DIR:Boliche_inteligente>
            COMMENT "Copiando barra inferior (logo.png) junto al ejecutable")
endif()

set(ORT_DLLS_EXISTENTES)
foreach(dll ${ORT_DLLS})
    if(EXISTS "${dll}")
        list(APPEND ORT_DLLS_EXISTENTES "${dll}")
    endif()
endforeach()

if(ORT_DLLS_EXISTENTES)
    message(STATUS "ONNX Runtime DLLs a copiar junto al ejecutable: ${ORT_DLLS_EXISTENTES}")
    add_custom_command(TARGET Boliche_inteligente POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ORT_DLLS_EXISTENTES}
            $<TARGET_FILE_DIR:Boliche_inteligente>
            COMMENT "Copiando DLLs de ONNX Runtime junto al ejecutable")
else()
    message(WARNING "No se encontraron las DLL de ONNX Runtime en ${ORT_ROOT}/lib")
endif()
