# Pacote de serviço de inferência

Gerado pelo Wulabel. Captura os streams de vídeo das câmeras, roda o modelo treinado sobre eles e envia um alarme quando encontra um alvo. Também oferece uma API HTTP de detecção.

## Execução

Requer Python 3.9 ou superior (não precisa de PyTorch nem de placa de vídeo).

- Windows: clique duas vezes em `run.bat`
- Linux: `sh run.sh`

Na primeira execução, um ambiente Python é criado e onnxruntime, opencv e numpy são instalados (cerca de 100 MB). Depois disso, inicia direto.

Para fazer inferência numa GPU NVIDIA: troque `onnxruntime` por `onnxruntime-gpu` em `requirements.txt`, apague a pasta `venv` e execute novamente.

## Configuração: config.json

```jsonc
{
  "model": "model.onnx",
  "gpu": true,                 // usa a GPU quando onnxruntime-gpu estiver instalado
  "max_side": 1280,            // reduz o lado maior de cada quadro a este tamanho antes da detecção; menor é mais rápido
  "port": 8090,                // porta da API HTTP
  "interval": 1.0,             // detecta um quadro a cada N segundos em cada stream
  "streams": [
    {
      "name": "Portão norte",
      "url": "rtsp://admin:senha@192.168.1.64:554/Streaming/Channels/101",
      "region": [[0.1, 0.4], [0.9, 0.4], [0.9, 1.0], [0.1, 1.0]]  // opcional: região de detecção, coordenadas em proporção 0–1
    }
  ],
  "alert": {
    "labels": ["sem_capacete"],  // só gera alarme para estas classes; vazio = todas as classes
    "min_score": 0.5,          // limiar de confiança
    "class_scores": {},        // limiar por classe, ex.: {"risco": 0.3}; as outras classes usam min_score
    "cooldown": 60,            // no máximo um alarme a cada N segundos para o mesmo stream e classe
    "hours": "07:00-19:00",    // horário ativo; vazio = o dia todo; passando da meia-noite, escreva "22:00-06:00"
    "save_image": true         // salva capturas dos alarmes em alerts/
  },
  "webhook": { "url": "http://sua-plataforma/api/alarm", "with_image": false, "headers": {} },
  "mqtt": { "host": "", "port": 1883, "topic": "xlabel/alarm", "username": "", "password": "" },
  "reflux": { "enabled": true, "low": 0.3, "high": 0.5, "interval": 60 }
}
```

- Os streams podem ser RTSP, HTTP-FLV, HLS (m3u8) ou MP4 via http. Também podem ser o caminho de um arquivo de vídeo local, para testes (arquivos locais e vídeos sob demanda via http repetem em loop por padrão).
- A região de detecção é verificada pelo **ponto central da borda inferior** de cada caixa (onde a pessoa ou o veículo está).

## Mensagem de alarme

Mensagens enviadas ao Webhook (POST JSON) e ao MQTT:

```json
{
  "stream": "Portão norte",
  "time": "2026-11-01T08:30:15",
  "labels": ["sem_capacete"],
  "detections": [{ "label": "sem_capacete", "score": 0.91, "box": [x, y, w, h] }],
  "width": 1920,
  "height": 1080,
  "image": "alerts/20261101/Portão norte_083015.jpg"
}
```

Com `with_image` ativado no Webhook, é incluído um campo extra `image_base64` (JPEG com as caixas desenhadas).

## Resultados por tipo de modelo

Os campos de cada item em `detections` dependem do modelo. O `task` retornado por `GET /health` e `POST /predict` indica qual é o tipo:

| Modelo | task | Campos de cada item |
|---|---|---|
| Detecção de objetos | `detect` | `label`, `score`, `box` [x, y, w, h] |
| Segmentação de instâncias | `segment` | O mesmo, mais `polygon` [x1, y1, x2, y2, …] (contorno do objeto, pixels da imagem original) |
| Pontos-chave | `pose` | O mesmo, mais `keypoints` [x, y, v, …] (v = 2: encontrado, 0: não encontrado); as capturas de alarme desenham o esqueleto conforme `skeleton` no `config.json` |
| Classificação de imagens | `classify` | `label`, `score`, sem `box`: um item para cada classe cuja probabilidade atinge `min_score` (até 5; sem limite em modelos multirrótulo). A classificação avalia o quadro inteiro e não filtra por `region` |

## API HTTP

| Endpoint | Descrição |
|---|---|
| `GET /health` | Status do serviço e se cada stream está online |
| `GET /alerts?limit=50` | Alarmes recentes |
| `GET /alerts/20261101/xxx.jpg` | Captura do alarme |
| `POST /predict?conf=0.4` | Envia uma imagem (o arquivo de imagem direto no corpo da requisição, ou JSON `{"image": "base64"}`) e retorna as detecções |

```bash
curl -X POST --data-binary @test.jpg "http://127.0.0.1:8090/predict?conf=0.4"
```

## Retorno do campo

Quadros em que o modelo fica em dúvida (confiança entre `low` e `high`) são salvos em `reflux/` (no máximo um por stream a cada `interval` segundos).
De tempos em tempos, copie a pasta `reflux` de volta, clique em "Importar imagens de campo" na página "Implantar" do Wulabel, anote as imagens e treine de novo — o modelo fica mais preciso quanto mais você usa.

## Arquivos

```
model.onnx / model.json   modelo e classes
service.py                programa do serviço
config.json               configuração
alerts/                   capturas de alarmes
reflux/                   imagens de campo
logs/service.log          log
```
