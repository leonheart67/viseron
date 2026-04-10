# Viseron — Patch détection par régions (Frigate-style)

Améliore la détection d'objets de Viseron en n'envoyant à YOLO que les
zones de l'image où du mouvement a été détecté, au lieu de la frame entière.

---

## Pourquoi ce patch ?

Par défaut, Viseron envoie la **frame complète** à YOLO même si le mouvement
ne concerne que 5% de l'image. Frigate, lui, envoie uniquement un **crop carré**
autour de la zone de mouvement — l'objet occupe alors toute l'image analysée,
ce qui améliore la précision de détection.

| | Viseron original | Avec patch | Frigate |
|---|---|---|---|
| Ce qu'envoie YOLO | Frame entière | Crop autour du mouvement | Crop autour du mouvement |
| Résolution effective de l'objet | Faible | Haute | Haute |
| CPU (8 caméras, substreams) | ~16% | ~19% | — |

---

## Fonctionnalités

- **Crops de mouvement** : régions carrées calculées depuis les contours MOG2
- **Scaling correct** : MOG2 travaille à 300×300, les contours sont remis
  à l'échelle de la frame de détection (substream ou full HD)
- **Batch GPU** : toutes les régions d'une frame envoyées en un seul appel YOLO
- **Expand si coupé** : si un objet touche le bord du crop, on agrandit et relance
- **Tracking stationnaire** : quand le mouvement s'arrête, on crope autour du
  dernier bbox connu (évite les inférences full-frame pour objets immobiles)
- **Cooldown mouvement** : 2s de grâce après arrêt du mouvement (anti-flicker HA)
- **Throttle CPU** : le décodage de frame est court-circuité avant même de
  commencer si le fps throttle dit non
- **Filtre bruit** : micro-contours (vent, variation lumineuse) filtrés par aire
- **debug_save_crops** : sauvegarde des crops dans `/config/crops/` pour réglage

---

## Installation

```bash
# 1. Extraire l'archive
tar xzf viseron_region_patch.tar.gz
cd viseron_patch

# 2. Appliquer (remplace "viseron" par le nom de ton container si différent)
chmod +x apply_patch.sh
./apply_patch.sh viseron
```

Le script est **idempotent** : on peut le relancer sans risque sur un container
déjà patché — il détectera que le patch est en place et ne fera rien.

---

## Configuration

Ajoute ces options dans `config.yaml` pour chaque caméra concernée.
**Seule `region_detection: true` est obligatoire**, tout le reste a des valeurs
par défaut raisonnables.

```yaml
yolo:
  object_detector:
    model_path: /config/mon_modele.pt
    device: cuda
    cameras:

      camera_entree:
        fps: 2
        scan_on_motion_only: true    # requis : MOG2 doit être configuré
        region_detection: true       # active le patch
        region_margin: 0.20          # marge autour du bbox de mouvement (20%)
        region_min_size: 0.20        # taille min crop = 20% du plus court côté
        region_expand_cutoff: true   # re-détecte si objet coupé au bord
        debug_save_crops: false      # true = sauvegarde crops dans /config/crops/
        labels:
          - label: person
            confidence: 0.65

      camera_foret:
        fps: 2
        scan_on_motion_only: true
        region_detection: false      # désactivé (végétation = bruit permanent)
        labels:
          - label: person
            confidence: 0.65

mog2:
  motion_detector:
    cameras:
      camera_entree:
        fps: 2
        threshold: 13
        area: 0.003
```

### Paramètres détaillés

| Paramètre | Défaut | Description |
|---|---|---|
| `region_detection` | `false` | Active le patch pour cette caméra |
| `region_margin` | `0.15` | Marge autour du bbox de mouvement (0.0–0.5) |
| `region_min_size` | `0.10` | Taille min du crop = fraction du plus court côté |
| `region_expand_cutoff` | `true` | Agrandit le crop si objet coupé au bord |
| `debug_save_crops` | `false` | Sauvegarde les crops JPEG pour débogage |

---

## Réglage de region_margin et region_min_size

Active temporairement `debug_save_crops: true` sur une caméra et déclenche
du mouvement. Les crops sont sauvegardés dans `/config/crops/<camera_id>/`.

```bash
# Voir les crops depuis l'hôte
docker cp viseron:/config/crops/camera_entree/ /tmp/crops/

# Vider les crops après débogage
docker exec viseron rm -rf /config/crops/
```

Le nom du fichier indique tout :
```
1712345678_450-200-750-500_300x300_full640x360.jpg
  timestamp  x1-y1-x2-y2  crop_w x crop_h  full_frame
```

- **Objet coupé au bord** → augmente `region_margin` (0.25, 0.30)
- **Crop trop petit, objet raté** → augmente `region_min_size` (0.25, 0.30)
- **Trop de faux positifs** → augmente `region_min_size` ou le seuil MOG2 `area`
- **Crops dans les zones masquées** → le mask MOG2 n'est pas assez strict,
  augmente `threshold` ou `area` dans la config mog2

---

## Optimisations recommandées

### Substreams (fortement recommandé)

Sans substream, FFmpeg décode la frame full HD pour MOG2/YOLO → CPU élevé.
Avec substream, MOG2 et YOLO utilisent le flux basse résolution, l'enregistrement
utilise le flux principal.

```yaml
ffmpeg:
  camera:
    camera_entree:
      name: camera_entree
      host: 192.168.1.x
      port: 8554
      path: /camera_entree
      fps: 5
      substream:
        host: 192.168.1.x
        port: 8554
        path: /camera_entree_sub  # flux basse résolution (ex: 640x360)
        fps: 5
```

Impact mesuré sur 8 caméras :

| Config | CPU |
|---|---|
| Sans YOLO | 20% |
| YOLO original (sans patch) | ~16% |
| YOLO + patch + substreams | **~19%** |
| YOLO + patch, sans substreams | ~47% |

### FPS

Inutile de décoder plus vite que MOG2 et YOLO ne consomment :

```yaml
ffmpeg:
  camera:
    camera_entree:
      fps: 2    # FFmpeg décode à 2fps

mog2:
  motion_detector:
    cameras:
      camera_entree:
        fps: 2  # MOG2 analyse à 2fps

yolo:
  object_detector:
    cameras:
      camera_entree:
        fps: 2  # YOLO tourne à 2fps max
```

### Ne pas activer region_detection partout

Les caméras avec beaucoup de mouvement parasite (végétation, vent) généreront
beaucoup de contours → beaucoup de crops → surcharge. Garde `region_detection: false`
sur ces caméras et ajuste les masques MOG2.

---

## Fichiers du patch

| Fichier | Rôle |
|---|---|
| `apply_patch.sh` | Script maître — lance tout en une commande |
| `patch_yolo_detector.py` | Patche `yolo/object_detector.py` |
| `patch_camera_schema.py` | Patche `domains/object_detector/__init__.py` |
| `region_utils.py` | Nouveau module — logique crop/remap/dédup |

---

## Revenir en arrière

Chaque fichier patché a un backup automatique `.py.bak_YYYYMMDD_HHMMSS` :

```bash
# Restaurer les backups
docker exec viseron ls /src/viseron/components/yolo/object_detector.py.bak_*
docker exec viseron cp \
  /src/viseron/components/yolo/object_detector.py.bak_XXXXXXXX \
  /src/viseron/components/yolo/object_detector.py

docker exec viseron ls /src/viseron/domains/object_detector/__init__.py.bak_*
docker exec viseron cp \
  /src/viseron/domains/object_detector/__init__.py.bak_XXXXXXXX \
  /src/viseron/domains/object_detector/__init__.py

docker exec viseron rm /src/viseron/domains/object_detector/region_utils.py
docker restart viseron
```

Ou plus simplement, mettre `region_detection: false` sur toutes les caméras
dans `config.yaml` — le code patché se comportera exactement comme l'original.

---

## Ce que fait le patch vs Frigate

| Fonctionnalité | Frigate | Ce patch |
|---|---|---|
| Crops autour du mouvement | ✅ | ✅ |
| Régions carrées avec marge | ✅ | ✅ |
| Remap coords crop → full frame | ✅ | ✅ |
| Expand si objet coupé | ✅ | ✅ |
| Batch GPU | ✅ | ✅ |
| Déduplication IoU | ✅ | ✅ |
| Throttle fps | ✅ | ✅ |
| Substream détection | ✅ | ✅ |
| Tracking objet stationnaire | ✅ | ✅ |
| Cooldown motion anti-flicker | ✅ | ✅ |
| Tracking multi-objets entre frames | ✅ | ❌ |
| Timeline événements par objet | ✅ | ❌ |
