# Paquete de servicio de inferencia

Generado por Wulabel. Obtiene las transmisiones de video de las cámaras, las analiza con el modelo entrenado y envía una alarma cuando detecta un objetivo. También ofrece una API HTTP de detección.

## Ejecución

Requiere Python 3.9 o posterior (no necesita PyTorch ni tarjeta gráfica).

- Windows: haz doble clic en `run.bat`
- Linux: `sh run.sh`

La primera ejecución crea un entorno de Python e instala onnxruntime, opencv y numpy (unos 100 MB). Después arranca directamente.

Para inferir con una GPU NVIDIA: cambia `onnxruntime` por `onnxruntime-gpu` en `requirements.txt`, borra la carpeta `venv` y vuelve a ejecutar.

## Configuración: config.json

```jsonc
{
  "model": "model.onnx",
  "gpu": true,                 // usa la GPU si está instalado onnxruntime-gpu
  "max_side": 1280,            // reduce el lado largo de cada fotograma a este tamaño antes de detectar; más pequeño = más rápido
  "port": 8090,                // puerto de la API HTTP
  "interval": 1.0,             // cada cuántos segundos se analiza un fotograma de cada transmisión
  "streams": [
    {
      "name": "Puerta norte",
      "url": "rtsp://admin:contraseña@192.168.1.64:554/Streaming/Channels/101",
      "region": [[0.1, 0.4], [0.9, 0.4], [0.9, 1.0], [0.1, 1.0]]  // opcional: región de detección, coordenadas en proporción 0–1
    }
  ],
  "alert": {
    "labels": ["sin_casco"],   // solo alarma para estas clases; vacío = todas las clases
    "min_score": 0.5,          // umbral de confianza
    "class_scores": {},        // umbral por clase, p. ej. {"rayón": 0.3}; las demás clases usan min_score
    "cooldown": 60,            // como máximo una alarma cada N segundos por transmisión y clase
    "hours": "07:00-19:00",    // horario activo; vacío = todo el día; si cruza la medianoche, escribe "22:00-06:00"
    "save_image": true         // guarda capturas de las alarmas en alerts/
  },
  "webhook": { "url": "http://tu-plataforma/api/alarm", "with_image": false, "headers": {} },
  "mqtt": { "host": "", "port": 1883, "topic": "xlabel/alarm", "username": "", "password": "" },
  "reflux": { "enabled": true, "low": 0.3, "high": 0.5, "interval": 60 }
}
```

- Las fuentes pueden ser RTSP, HTTP-FLV, HLS (m3u8) o MP4 por http. También se puede indicar la ruta de un archivo de video local, para pruebas (los archivos locales y los videos bajo demanda por http se reproducen en bucle por defecto). Un video local debe apuntar al archivo en sí, por ejemplo `D:\\test\\1.mp4` (en JSON, escribe cada barra invertida dos veces o usa `D:/test/1.mp4`), no solo a la carpeta; si el registro muestra una y otra vez «Error de conexión», revisa esto primero. Para reconocer una carpeta de imágenes, envíalas de una en una con `POST /predict`. Para reproducir un video local solo una vez, añade `"loop": false` a esa transmisión. Para revisar un lote de videos una vez cada uno, con resultados por carpeta, es más cómodo usar el «Reconocimiento por lotes» de la página Desplegar de Wulabel.
- La región de detección se comprueba con el **punto medio del borde inferior** de cada caja (donde está la persona o el vehículo).

## Contenido de la alarma

Mensaje que reciben el Webhook (POST JSON) y MQTT:

```json
{
  "stream": "Puerta norte",
  "time": "2026-11-01T08:30:15",
  "labels": ["sin_casco"],
  "detections": [{ "label": "sin_casco", "score": 0.91, "box": [x, y, w, h] }],
  "width": 1920,
  "height": 1080,
  "image": "alerts/20261101/Puerta norte_083015.jpg"
}
```

Si activas `with_image` en el Webhook, se añade un campo `image_base64` (JPEG con las cajas dibujadas).

## Resultados según el tipo de modelo

Los campos de cada elemento de `detections` dependen del modelo. El `task` que devuelven `GET /health` y `POST /predict` indica de qué tipo es:

| Modelo | task | Campos de cada elemento |
|---|---|---|
| Detección de objetos | `detect` | `label`, `score`, `box` [x, y, w, h] |
| Segmentación de instancias | `segment` | Lo mismo, más `polygon` [x1, y1, x2, y2, …] (contorno del objeto, píxeles de la imagen original) |
| Puntos clave | `pose` | Lo mismo, más `keypoints` [x, y, v, …] (v = 2: encontrado, 0: no encontrado); las capturas de alarma dibujan el esqueleto según `skeleton` en `config.json` |
| Clasificación de imágenes | `classify` | `label`, `score`, sin `box`: un elemento por cada clase cuya probabilidad alcance `min_score` (hasta 5; sin límite en modelos multietiqueta). La clasificación evalúa toda la imagen y no filtra por `region` |

## API HTTP

| Endpoint | Descripción |
|---|---|
| `GET /health` | Estado del servicio y si cada transmisión está en línea |
| `GET /alerts?limit=50` | Alarmas recientes |
| `GET /alerts/20261101/xxx.jpg` | Captura de una alarma |
| `POST /predict?conf=0.4` | Sube una imagen (el archivo de imagen directamente en el cuerpo de la solicitud, o JSON `{"image": "base64"}`) y devuelve las detecciones |

```bash
curl -X POST --data-binary @test.jpg "http://127.0.0.1:8090/predict?conf=0.4"
```

## Retorno de datos de campo

Los fotogramas en los que el modelo duda (confianza entre `low` y `high`) se guardan en `reflux/` (como máximo uno por transmisión cada `interval` segundos).
Copia de vez en cuando la carpeta `reflux`, haz clic en «Importar imágenes de campo» en la página «Desplegar» de Wulabel, anótalas y vuelve a entrenar: el modelo se vuelve más preciso cuanto más se usa.

## Archivos

```
model.onnx / model.json   modelo y clases
service.py                programa del servicio
config.json               configuración
alerts/                   capturas de alarmas
reflux/                   imágenes de campo
logs/service.log          registro
```
