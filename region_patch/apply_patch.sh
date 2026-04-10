#!/usr/bin/env bash
# =============================================================================
# apply_patch.sh
# =============================================================================
# Script maître pour appliquer le patch de détection par régions (Frigate-style)
# sur une installation Viseron en Docker.
#
# Ce script :
#   1. Extrait les fichiers source du container
#   2. Applique patch_yolo_detector.py    → yolo/object_detector.py
#   3. Applique patch_camera_schema.py    → domains/object_detector/__init__.py
#   4. Copie region_utils.py              → domains/object_detector/
#   5. Reinjecte les fichiers dans le container
#   6. Redémarre le container
#
# Usage :
#   chmod +x apply_patch.sh
#   ./apply_patch.sh [NOM_CONTAINER]    # défaut : "viseron"
#
# Idempotent : peut être relancé sans danger sur un container déjà patché.
# =============================================================================
set -e

CONTAINER="${1:-viseron}"
DIR="$(cd "$(dirname "$0")" && pwd)"

# Chemins dans le container
YOLO_SRC="/src/viseron/components/yolo/object_detector.py"
OD_BASE_SRC="/src/viseron/domains/object_detector/__init__.py"
REG_DST="/src/viseron/domains/object_detector/region_utils.py"

# Copies locales temporaires
YOLO_LOCAL="${DIR}/yolo_object_detector.py"
OD_BASE_LOCAL="${DIR}/object_detector_base__init__.py"

echo "======================================================"
echo " Viseron — Patch détection par régions (Frigate-style)"
echo " Container : ${CONTAINER}"
echo "======================================================"

# ---- Vérification container ----
if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
    echo "❌  Container '${CONTAINER}' non démarré."
    echo "    Lance-le d'abord : docker start ${CONTAINER}"
    exit 1
fi
echo "✅  Container en cours d'exécution."

# ---- Extraction ----
echo ""
echo "📥  Extraction des fichiers source…"
docker cp "${CONTAINER}:${YOLO_SRC}"    "${YOLO_LOCAL}"
docker cp "${CONTAINER}:${OD_BASE_SRC}" "${OD_BASE_LOCAL}"

# ---- Application des patches ----
echo ""
echo "🔧  Patch 1/2 — yolo/object_detector.py (logique région + batch GPU)…"
python3 "${DIR}/patch_yolo_detector.py" "${YOLO_LOCAL}"

echo ""
echo "🔧  Patch 2/2 — domains/object_detector/__init__.py (schéma + throttle)…"
python3 "${DIR}/patch_camera_schema.py" "${OD_BASE_LOCAL}"

# ---- Réinjection ----
echo ""
echo "📤  Réinjection dans le container…"
docker cp "${DIR}/region_utils.py" "${CONTAINER}:${REG_DST}"
docker cp "${YOLO_LOCAL}"          "${CONTAINER}:${YOLO_SRC}"
docker cp "${OD_BASE_LOCAL}"       "${CONTAINER}:${OD_BASE_SRC}"
echo "   ✅  region_utils.py              → ${REG_DST}"
echo "   ✅  yolo/object_detector.py      → ${YOLO_SRC}"
echo "   ✅  object_detector/__init__.py  → ${OD_BASE_SRC}"

# ---- Redémarrage ----
echo ""
echo "🔄  Redémarrage du container '${CONTAINER}'…"
docker restart "${CONTAINER}"

echo ""
echo "======================================================"
echo " Patch appliqué avec succès !"
echo ""
echo " Ajoute ces options dans config.yaml pour chaque"
echo " caméra où tu veux activer la détection par régions :"
echo ""
echo "   yolo:"
echo "     object_detector:"
echo "       cameras:"
echo "         ma_camera:"
echo "           scan_on_motion_only: true    # requis"
echo "           region_detection: true       # active le patch"
echo "           region_margin: 0.20          # marge autour du mouvement"
echo "           region_min_size: 0.20        # taille min du crop"
echo "           region_expand_cutoff: true   # agrandit si objet coupé"
echo "           debug_save_crops: false      # true pour déboguer"
echo "======================================================"
