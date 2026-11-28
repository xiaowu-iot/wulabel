# Package de service d'inférence

Généré par Wulabel. Il récupère les flux vidéo des caméras, les analyse avec le modèle entraîné et envoie une alerte dès qu'une cible est détectée. Il fournit aussi une API HTTP de détection.

## Exécution

Nécessite Python 3.9 ou plus récent (ni PyTorch ni carte graphique ne sont nécessaires).

- Windows : double-cliquez sur `run.bat`
- Linux : `sh run.sh`

Au premier lancement, un environnement Python est créé automatiquement et onnxruntime, opencv et numpy sont installés (environ 100 Mo). Ensuite, le service démarre directement.

Pour l'inférence sur un GPU NVIDIA : remplacez `onnxruntime` par `onnxruntime-gpu` dans `requirements.txt`, supprimez le dossier `venv` puis relancez.

## Configuration : config.json

```jsonc
{
  "model": "model.onnx",
  "gpu": true,                 // utilise le GPU si onnxruntime-gpu est installé
  "max_side": 1280,            // réduit le grand côté de chaque image à cette taille avant détection ; plus petit = plus rapide
  "port": 8090,                // port de l'API HTTP
  "interval": 1.0,             // analyse une image toutes les N secondes pour chaque flux
  "streams": [
    {
      "name": "Porte nord",
      "url": "rtsp://admin:motdepasse@192.168.1.64:554/Streaming/Channels/101",
      "region": [[0.1, 0.4], [0.9, 0.4], [0.9, 1.0], [0.1, 1.0]]  // facultatif : zone de détection, coordonnées en proportions 0–1
    }
  ],
  "alert": {
    "labels": ["sans_casque"],  // n'alerter que pour ces classes ; vide = toutes les classes
    "min_score": 0.5,          // seuil de confiance
    "class_scores": {},        // seuil par classe, ex. {"rayure": 0.3} ; les autres classes utilisent min_score
    "cooldown": 60,            // au plus une alerte toutes les N secondes pour un même flux et une même classe
    "hours": "07:00-19:00",    // plage horaire active ; vide = toute la journée ; à cheval sur minuit, écrire "22:00-06:00"
    "save_image": true         // enregistre les captures d'alerte dans alerts/
  },
  "webhook": { "url": "http://votre-plateforme/api/alarm", "with_image": false, "headers": {} },
  "mqtt": { "host": "", "port": 1883, "topic": "xlabel/alarm", "username": "", "password": "" },
  "reflux": { "enabled": true, "low": 0.3, "high": 0.5, "interval": 60 },
  "watch": { "dir": "D:/videos", "out": "watch_results", "every": 1.0, "interval": 10, "settle": 5 }  // facultatif : mode dossier
}
```

- Les sources peuvent être des flux RTSP, HTTP-FLV, HLS (m3u8) ou MP4 via http. Une source peut aussi être le chemin d'un fichier vidéo local, pour les tests (les fichiers locaux et les vidéos à la demande via http sont lus en boucle par défaut). Pour une vidéo locale, indiquez le fichier lui-même, par exemple `D:\\test\\1.mp4` (en JSON, doublez chaque barre oblique inverse ou écrivez `D:/test/1.mp4`), pas seulement le dossier ; si le journal affiche sans arrêt « Échec de la connexion », vérifiez d'abord ce point. Pour reconnaître un dossier d'images, envoyez-les une par une avec `POST /predict`. Pour lire une vidéo locale une seule fois, ajoutez `"loop": false` à ce flux. Pour passer en revue un lot de vidéos une fois chacune, avec des résultats par dossier, utilisez plutôt la « Reconnaissance par lot » de la page Déploiement de Wulabel.
- La zone de détection est évaluée sur le **milieu du bord inférieur** de chaque boîte (là où se tient la personne ou le véhicule).

## Contenu de l'alerte

Message envoyé au Webhook (POST JSON) et via MQTT :

```json
{
  "stream": "Porte nord",
  "time": "2026-11-01T08:30:15",
  "labels": ["sans_casque"],
  "detections": [{ "label": "sans_casque", "score": 0.91, "box": [x, y, w, h] }],
  "width": 1920,
  "height": 1080,
  "image": "alerts/20261101/Porte nord_083015.jpg"
}
```

Si `with_image` est activé pour le Webhook, un champ `image_base64` supplémentaire est ajouté (JPEG avec les boîtes dessinées).

## Résultats selon le type de modèle

Les champs de chaque élément de `detections` dépendent du modèle. Le champ `task` renvoyé par `GET /health` et `POST /predict` indique son type :

| Modèle | task | Champs de chaque élément |
|---|---|---|
| Détection d'objets | `detect` | `label`, `score`, `box` [x, y, w, h] |
| Segmentation d'instances | `segment` | Idem, plus `polygon` [x1, y1, x2, y2, …] (contour de l'objet, pixels de l'image d'origine) |
| Points clés | `pose` | Idem, plus `keypoints` [x, y, v, …] (v = 2 : trouvé, 0 : non trouvé) ; les captures d'alerte tracent le squelette selon `skeleton` dans `config.json` |
| Classification d'images | `classify` | `label`, `score`, sans `box` : un élément par classe dont la probabilité atteint `min_score` (5 au plus ; sans limite pour les modèles multi-étiquettes). La classification porte sur l'image entière et ne filtre pas par `region` |

## API HTTP

| Point d'accès | Description |
|---|---|
| `GET /health` | État du service et disponibilité de chaque flux |
| `GET /alerts?limit=50` | Alertes récentes |
| `GET /alerts/20261101/xxx.jpg` | Capture d'une alerte |
| `POST /predict?conf=0.4` | Envoie une image (fichier image brut dans le corps de la requête, ou JSON `{"image": "base64"}`) et renvoie les détections |

```bash
curl -X POST --data-binary @test.jpg "http://127.0.0.1:8090/predict?conf=0.4"
```

## Mode dossier

Il s'active dès que `watch.dir` est renseigné dans la configuration : le service parcourt ce dossier (sous-dossiers compris) toutes les `interval` secondes et reconnaît automatiquement les nouvelles images et vidéos qu'on y dépose, sans caméra.

- Un fichier n'est reconnu qu'après être resté en place `settle` secondes sans que sa taille change : un fichier en cours de copie n'est pas reconnu à moitié.
- Pour les vidéos, une image est analysée toutes les `every` secondes ; pour chaque classe reconnue, une capture avec les boîtes est enregistrée (l'image au score le plus élevé), rangée selon le sous-dossier du fichier (par exemple le numéro de commande) dans `out` (par défaut `watch_results/` dans le package).
- Chaque fichier ajoute une ligne à `watch_results/resultats.csv` : heure, fichier, dossier, résultat, score max, capture.
- Ce qui passe les règles d'alerte (classes, confiance, plages horaires) est envoyé par Webhook / MQTT comme d'habitude, avec deux champs en plus dans le message : `file` et `folder`.
- Les fichiers reconnus sont notés dans `watch_results/.wulabel-watch.json` : après un redémarrage du service, ils ne sont pas reconnus une seconde fois ; un fichier modifié est reconnu à nouveau. Les dossiers de résultats de la reconnaissance par lot de Wulabel sont ignorés automatiquement.
- `watch` dans `GET /health` indique combien de fichiers ont été reconnus et lequel est le dernier.

## Retour du terrain

Les images sur lesquelles le modèle hésite (confiance entre `low` et `high`) sont enregistrées dans `reflux/` (au plus une par flux toutes les `interval` secondes).
Rapatriez régulièrement le dossier `reflux`, cliquez sur « Importer des images remontées du terrain » dans la page « Déploiement » de Wulabel, annotez-les et réentraînez : plus vous l'utilisez, plus le modèle devient précis.

## Fichiers

```
model.onnx / model.json   modèle et classes
service.py                programme du service
config.json               configuration
alerts/                   captures d'alerte
reflux/                   images du terrain
watch_results/            captures et tableau du mode dossier
logs/service.log          journal
```
