#!/usr/bin/env python3
"""
patch_camera_schema.py
======================
Patche viseron/domains/object_detector/__init__.py

Ajoute 5 nouvelles clés dans CAMERA_SCHEMA (schéma voluptuous par caméra)
pour que Viseron accepte les options de détection par régions dans config.yaml.

Sans ce patch, Viseron refuse de démarrer avec l'erreur :
  "extra keys not allowed @ data['yolo']['object_detector']['cameras']['...']['region_detection']"

Ajoute aussi le throttle dans _detect() pour court-circuiter le décodage
de frame AVANT qu'il ne se produise, quand le fps throttle dit non.
C'est le fix critique pour le CPU : sans ça, le thread décode chaque frame
même si on va juste retourner le cache.

Usage :
    python3 patch_camera_schema.py <domains/object_detector/__init__.py>
"""
import re
import sys
import shutil
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Clés à ajouter dans CAMERA_SCHEMA
# ---------------------------------------------------------------------------
SCHEMA_KEYS = """
        vol.Optional("region_detection",     default=False): vol.Boolean(),
        vol.Optional("region_margin",        default=0.15):  vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=0.5)),
        vol.Optional("region_min_size",      default=0.10):  vol.All(
            vol.Coerce(float), vol.Range(min=0.01, max=0.5)),
        vol.Optional("region_expand_cutoff", default=True):  vol.Boolean(),
        vol.Optional("debug_save_crops",     default=False): vol.Boolean(),
"""

# ---------------------------------------------------------------------------
# Throttle à injecter dans _detect() — court-circuite AVANT le décodage
# ---------------------------------------------------------------------------
DETECT_THROTTLE = '''    def _detect(self, shared_frame: SharedFrame, frame_time: float):
        """Perform object detection and publish data."""
        import time as _time
        # Court-circuite le décodage si region_detection + throttle fps actif.
        # Sans ça, Python décode chaque frame même pour retourner le cache.
        cam_cfg = self._config.get("cameras", {}).get(
            shared_frame.camera_identifier, {}
        )
        if cam_cfg.get("region_detection", False):
            fps = cam_cfg.get("fps", 2)
            min_interval = 1.0 / max(fps, 0.1)
            last = getattr(self, "_detect_last_time", {})
            cam_id = shared_frame.camera_identifier
            now = _time.time()
            if now - last.get(cam_id, 0) < min_interval:
                return  # skip — économise le décodage de frame
            last[cam_id] = now
            self._detect_last_time = last
        decoded_frame = self._camera.shared_frames.get_decoded_frame_rgb(shared_frame)'''


def already_patched_schema(content: str) -> bool:
    return "region_detection" in content


def already_patched_detect(content: str) -> bool:
    return "_detect_last_time" in content


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 patch_camera_schema.py <object_detector/__init__.py>")
        sys.exit(1)

    p = Path(sys.argv[1])
    if not p.exists():
        print(f"Erreur : {p} introuvable")
        sys.exit(1)

    content = p.read_text()
    needs_schema = not already_patched_schema(content)
    needs_detect = not already_patched_detect(content)

    if not needs_schema and not needs_detect:
        print("✅ Déjà patché — rien à faire.")
        sys.exit(0)

    shutil.copy2(p, p.with_suffix(f".py.bak_{datetime.now():%Y%m%d_%H%M%S}"))
    print("Application des patches…")

    # Patch 1 : CAMERA_SCHEMA — insérer les clés avant la fermeture du dict
    if needs_schema:
        new, n = re.subn(
            r'(CAMERA_SCHEMA\s*=\s*vol\.Schema\(\s*\{)(.*?)(\n    \},\s*\n\))',
            lambda m: m.group(1) + m.group(2) + SCHEMA_KEYS + m.group(3),
            content, count=1, flags=re.DOTALL,
        )
        if n:
            content = new
            print("  ✅  CAMERA_SCHEMA : clés region ajoutées")
        else:
            print("  ⚠️   CAMERA_SCHEMA : pattern non trouvé")

    # Patch 2 : _detect() — injecter le throttle avant le décodage
    if needs_detect:
        old = '''    def _detect(self, shared_frame: SharedFrame, frame_time: float):
        """Perform object detection and publish data."""
        decoded_frame = self._camera.shared_frames.get_decoded_frame_rgb(shared_frame)'''
        if old in content:
            content = content.replace(old, DETECT_THROTTLE, 1)
            print("  ✅  _detect() : throttle CPU ajouté")
        else:
            print("  ⚠️   _detect() : pattern non trouvé")

    p.write_text(content)
    print(f"\n🎉 {p} patché avec succès.")


if __name__ == "__main__":
    main()
