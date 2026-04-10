#!/usr/bin/env python3
"""
patch_yolo_detector.py
======================
Patche viseron/components/yolo/object_detector.py pour ajouter
la détection par régions de type Frigate.

Modifications apportées :
  1. __init__  : souscription à EVENT_MOTION_DETECTED
  2. return_objects() : dispatch région / tracking / full-frame
  3. _return_objects_full() : détection full-frame originale (inchangée)
  4. _on_motion_detected() : reçoit les contours MOG2, filtre le bruit,
                             gère le cooldown de 2s après arrêt du mouvement
  5. _is_motion_active() : teste si le mouvement est actif ou en cooldown
  6. _predict() : appel YOLO réutilisable (full frame OU liste de crops)
  7. _postprocess_crop() : remet les coords YOLO (crop-relatives) dans
                           le repère de la frame complète
  8. _save_crop_debug() : sauvegarde les crops pour débogage (throttlé 1/s)
  9. _return_objects_regions() : pipeline crop → BATCH GPU → remap
 10. _return_objects_tracked() : crop autour du dernier bbox connu
                                 (objet stationnaire, plus de mouvement)

Usage :
    python3 patch_yolo_detector.py <yolo/object_detector.py>

Idempotent : vérifie si le patch est déjà appliqué avant de toucher au fichier.
"""
import ast
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Code injecté dans __init__ après super().__init__(...)
# ---------------------------------------------------------------------------
INIT_INJECT = '''
        # --- Détection par régions (Frigate-style) ---
        from viseron.domains.motion_detector.const import (  # noqa
            EVENT_MOTION_DETECTED,
        )
        self._motion_contours = []      # contours rel [0,1] de MOG2
        self._tracked_bboxes  = []      # bboxes du dernier objet détecté
        self._last_results    = []      # cache résultats (anti-flicker)
        self._last_infer_time = 0.0     # timestamp dernière inférence
        self._motion_clear_at = None    # timestamp fin cooldown mouvement
        self._listeners.append(
            vis.listen_event(
                EVENT_MOTION_DETECTED.format(camera_identifier=camera_identifier),
                self._on_motion_detected,
            )
        )
'''

# ---------------------------------------------------------------------------
# Remplacement de return_objects() + toutes les nouvelles méthodes
# (insérées depuis "def return_objects" jusqu'à "def unload")
# ---------------------------------------------------------------------------
NEW_TAIL = '''    def return_objects(self, frame: np.ndarray) -> list[DetectedObject]:
        """
        Point d'entrée principal de la détection.

        Comportement selon l'état :
          region_detection=False  → full-frame (comportement original)
          Mouvement actif         → crop sur zones de mouvement (batch GPU)
          Objet tracké, pas de mvt→ crop autour du dernier bbox connu
          Rien                    → [] (évite les inférences inutiles)

        Le throttle court-circuite AVANT le décodage de frame (dans _detect
        de la classe de base) pour ne pas gaspiller CPU sur des frames ignorées.
        """
        import time as _time
        cam_cfg = self._config.get("cameras", {}).get(
            self._camera.identifier, {}
        )

        # Si region_detection désactivé → comportement original
        if not cam_cfg.get("region_detection", False):
            return self._return_objects_full(frame)

        # Throttle au fps configuré (évite d'inférer plus vite que nécessaire)
        fps = cam_cfg.get("fps", 2)
        min_interval = 1.0 / max(fps, 0.1)
        now = _time.time()
        if now - self._last_infer_time < min_interval:
            return self._last_results   # retourne le cache
        self._last_infer_time = now

        # --- Cas 1 : mouvement actif (ou cooldown 2s) ---
        if self._is_motion_active() and self._motion_contours:
            results = self._return_objects_regions(frame, cam_cfg)
            # Mémorise les bboxes pour le tracking stationnaire
            self._tracked_bboxes = [
                (d.rel_x1, d.rel_y1, d.rel_x2, d.rel_y2) for d in results
            ]
            self._last_results = results
            return results

        # --- Cas 2 : plus de mouvement mais objet encore présent ---
        if self._tracked_bboxes:
            results = self._return_objects_tracked(frame, cam_cfg)
            self._tracked_bboxes = [
                (d.rel_x1, d.rel_y1, d.rel_x2, d.rel_y2) for d in results
            ]
            self._last_results = results
            return results

        # --- Cas 3 : rien → pas d'inférence ---
        self._last_results = []
        return []

    def _return_objects_full(self, frame: np.ndarray) -> list[DetectedObject]:
        """Détection full-frame originale — comportement Viseron par défaut."""
        try:
            results = self._detector.predict(
                frame,
                conf=self._config[CONFIG_MIN_CONFIDENCE],
                iou=self._config[CONFIG_IOU],
                half=self._config[CONFIG_HALF_PRECISION],
                device=self._config[CONFIG_DEVICE],
                verbose=False,
            )
        except ValueError as error:
            LOGGER.error("Error calling yolo prediction: %s", error)
            return []
        return self.postprocess(results)

    def _on_motion_detected(self, event_data) -> None:
        """
        Reçoit les événements de mouvement MOG2.

        - Quand mouvement=True  : stocke les rel_contours filtrés par taille
        - Quand mouvement=False : démarre un cooldown de 2s avant de vider
                                  les contours (évite le flickering des états HA)

        Les rel_contours sont en [0,1] (relatifs à la frame 300x300 de MOG2).
        region_utils.py les rescale vers la résolution de la frame de détection.

        Filtre les contours trop petits (bruit de vent, variation lumineuse)
        en ne gardant que ceux représentant > 0.02% de la frame MOG2.
        """
        import time as _time
        try:
            motion = event_data.data
            if not motion.motion_detected:
                # Cooldown : garde les contours encore 2s après arrêt du mouvement
                self._motion_clear_at = _time.time() + 2.0
                return
            # Mouvement actif : annule le cooldown
            self._motion_clear_at = None
            contours_obj = motion.motion_contours
            if not contours_obj:
                self._motion_contours = []
                return
            # Filtre les micro-contours (bruit) : garde > 0.02% de la frame MOG2
            min_area = 0.0002
            self._motion_contours = [
                c for c, a in zip(
                    contours_obj.rel_contours,
                    contours_obj.contour_areas,
                )
                if a > min_area
            ]
        except Exception:
            self._motion_contours = []

    def _is_motion_active(self) -> bool:
        """
        Retourne True si le mouvement est actif ou dans le cooldown de 2s.
        Vide les contours si le cooldown est expiré.
        """
        import time as _time
        clear_at = self._motion_clear_at
        if clear_at is None:
            return bool(self._motion_contours)
        if _time.time() < clear_at:
            return True
        # Cooldown expiré
        self._motion_contours = []
        self._motion_clear_at = None
        return False

    def _predict(self, frame_or_batch) -> list:
        """
        Appel YOLO unifié — accepte une image unique ou une liste d'images.
        YOLO supporte nativement le batch : passer une liste réduit le nombre
        d'appels GPU (toutes les régions en une seule inférence).
        """
        return self._detector.predict(
            frame_or_batch,
            conf=self._config[CONFIG_MIN_CONFIDENCE],
            iou=self._config[CONFIG_IOU],
            half=self._config[CONFIG_HALF_PRECISION],
            device=self._config[CONFIG_DEVICE],
            verbose=False,
        )

    def _postprocess_crop(self, results, region, full_w: int, full_h: int) -> list:
        """
        Convertit les résultats YOLO (coords relatives au crop) en
        DetectedObject avec des coords relatives à la frame complète.

        Problème à résoudre :
          YOLO retourne xyxy en pixels DANS LE CROP (ex: x1=50 sur un crop 200px).
          postprocess() original utilise frame_res=self._camera.resolution (full frame).
          Si on lui passe un crop, la normalisation serait fausse.

        Solution :
          On ajoute region.x1/y1 pour revenir en pixels full-frame,
          puis from_absolute() normalise avec la bonne résolution.
        """
        objects = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            names = result.names
            for i in range(len(result.boxes)):
                cls = int(result.boxes[i].cls[0])
                # Coords pixels dans le crop
                cx1, cy1, cx2, cy2 = [int(v) for v in result.boxes[i].xyxy[0]]
                # Remap → pixels dans la frame complète
                fx1 = max(0, min(cx1 + region.x1, full_w))
                fy1 = max(0, min(cy1 + region.y1, full_h))
                fx2 = max(0, min(cx2 + region.x1, full_w))
                fy2 = max(0, min(cy2 + region.y1, full_h))
                objects.append(DetectedObject.from_absolute(
                    label=names[cls],
                    confidence=float(result.boxes[i].conf),
                    x1=fx1, y1=fy1, x2=fx2, y2=fy2,
                    frame_res=(full_w, full_h),  # résolution frame complète ✓
                    model_res=(full_w, full_h),
                ))
        return objects

    def _save_crop_debug(self, crop, camera_id, region, full_w, full_h) -> None:
        """
        Sauvegarde un crop JPEG dans /config/crops/<camera_id>/ pour débogage.

        Throttlé à 1 sauvegarde par caméra par seconde pour éviter de remplir
        le disque. Activé via debug_save_crops: true dans la config.

        Nom du fichier : <timestamp>_<x1>-<y1>-<x2>-<y2>_<wc>x<hc>_full<wf>x<hf>.jpg
        """
        try:
            import cv2 as _cv2
            import os as _os
            import time as _time
            now = _time.time()
            last = getattr(self, "_last_crop_save", {})
            if now - last.get(camera_id, 0) < 1.0:
                return  # throttle 1/s
            last[camera_id] = now
            self._last_crop_save = last
            out = f"/config/crops/{camera_id}"
            _os.makedirs(out, exist_ok=True)
            ts = int(now * 1000)
            _cv2.imwrite(
                f"{out}/{ts}_{region.x1}-{region.y1}-{region.x2}-{region.y2}"
                f"_{crop.shape[1]}x{crop.shape[0]}_full{full_w}x{full_h}.jpg",
                crop,
            )
        except Exception as exc:
            LOGGER.debug("debug_save_crops failed: %s", exc)

    def _return_objects_regions(self, frame: "np.ndarray", cam_cfg: dict) -> list:
        """
        Détection sur crops de mouvement avec BATCH GPU.

        Pipeline :
          1. Calcule les régions carrées depuis les contours MOG2
          2. Crope tous les crops en une passe
          3. Envoie tous les crops en UN SEUL appel YOLO (batch GPU)
          4. Remap les résultats vers la frame complète
          5. Si un objet est coupé au bord → agrandit et re-détecte (2e batch)
          6. Déduplique les détections issues de régions chevauchantes

        Le batch GPU est le principal gain vs une détection par région :
          3 régions → 1 inférence au lieu de 3.
        """
        from viseron.domains.object_detector.region_utils import (  # noqa
            compute_regions_from_contours,
            expand_region,
            deduplicate_detections,
        )

        full_h, full_w = frame.shape[:2]
        margin     = cam_cfg.get("region_margin",        0.15)
        min_size   = cam_cfg.get("region_min_size",      0.10)
        expand     = cam_cfg.get("region_expand_cutoff", True)
        save_crops = cam_cfg.get("debug_save_crops",     False)

        regions = compute_regions_from_contours(
            self._motion_contours, full_w, full_h,
            margin=margin, min_size_ratio=min_size,
        )

        if not regions:
            # Aucune région exploitable → fallback full-frame
            return self._return_objects_full(frame)

        # --- Prépare tous les crops ---
        crops         = []
        valid_regions = []
        for region in regions:
            crop = region.crop(frame)
            if crop.size == 0:
                continue
            crops.append(crop)
            valid_regions.append(region)

        if not crops:
            return self._return_objects_full(frame)

        # --- BATCH GPU : un seul appel YOLO pour tous les crops ---
        try:
            batch_results = self._predict(crops)
        except Exception as exc:
            LOGGER.warning("Batch predict failed: %s — fallback full-frame", exc)
            return self._return_objects_full(frame)

        all_objects   = []
        expand_crops  = []
        expand_regions = []

        for crop, region, result in zip(crops, valid_regions, batch_results):
            detections = self._postprocess_crop([result], region, full_w, full_h)
            final_crop, final_region = crop, region

            # Vérifie si un objet est coupé → collecte pour 2e batch
            if expand and detections:
                for det in detections:
                    cr_x1 = (det.rel_x1 * full_w  - region.x1) / max(region.width,  1)
                    cr_y1 = (det.rel_y1 * full_h - region.y1) / max(region.height, 1)
                    cr_x2 = (det.rel_x2 * full_w  - region.x1) / max(region.width,  1)
                    cr_y2 = (det.rel_y2 * full_h - region.y1) / max(region.height, 1)
                    if region.is_detection_cut_off(cr_x1, cr_y1, cr_x2, cr_y2):
                        exp = expand_region(region, 1.5, full_w, full_h)
                        ec  = exp.crop(frame)
                        if ec.size > 0:
                            expand_crops.append(ec)
                            expand_regions.append(exp)
                            detections = []   # sera remplacé par le 2e batch
                            final_crop, final_region = ec, exp
                        break

            if detections:
                if save_crops:
                    self._save_crop_debug(
                        final_crop, self._camera.identifier,
                        final_region, full_w, full_h,
                    )
                all_objects.extend(detections)

        # --- 2e batch GPU pour les objets coupés ---
        if expand_crops:
            try:
                exp_results = self._predict(expand_crops)
                for ec, er, result in zip(expand_crops, expand_regions, exp_results):
                    detections = self._postprocess_crop([result], er, full_w, full_h)
                    if save_crops:
                        self._save_crop_debug(
                            ec, self._camera.identifier, er, full_w, full_h,
                        )
                    all_objects.extend(detections)
            except Exception as exc:
                LOGGER.warning("Expand batch predict failed: %s", exc)

        return deduplicate_detections(all_objects)

    def _return_objects_tracked(self, frame: "np.ndarray", cam_cfg: dict) -> list:
        """
        Détection sur les bboxes du dernier objet connu (objet stationnaire).

        Quand un objet s'immobilise (ex: voiture garée), MOG2 s'arrête de
        détecter du mouvement mais l'objet est toujours là. Au lieu de
        faire une inférence full-frame coûteuse, on crope autour du dernier
        bbox connu avec une marge.

        Si l'objet a disparu, _tracked_bboxes sera vidé et les inférences
        s'arrêtent automatiquement.
        """
        from viseron.domains.object_detector.region_utils import (  # noqa
            Region,
            deduplicate_detections,
        )

        full_h, full_w = frame.shape[:2]
        margin = cam_cfg.get("region_margin", 0.15)

        all_objects = []
        for (rx1, ry1, rx2, ry2) in self._tracked_bboxes:
            # Reconstruit un crop carré centré sur le bbox précédent
            cx = int((rx1 + rx2) / 2 * full_w)
            cy = int((ry1 + ry2) / 2 * full_h)
            size = max(
                int((rx2 - rx1) * full_w * (1 + margin)),
                int((ry2 - ry1) * full_h * (1 + margin)),
            )
            x1 = max(0, cx - size // 2)
            y1 = max(0, cy - size // 2)
            x2 = min(full_w, x1 + size)
            y2 = min(full_h, y1 + size)

            region = Region(x1, y1, x2, y2)
            crop   = region.crop(frame)
            if crop.size == 0:
                continue
            try:
                all_objects.extend(
                    self._postprocess_crop(
                        self._predict(crop), region, full_w, full_h,
                    )
                )
            except Exception as exc:
                LOGGER.warning("Tracked region predict failed: %s", exc)

        return deduplicate_detections(all_objects)

    def unload(self) -> None:
        """Unload the object detector."""
        super().unload()
        del self._detector
'''


# ---------------------------------------------------------------------------
# Patcher
# ---------------------------------------------------------------------------

def already_patched(content: str) -> bool:
    return "_on_motion_detected" in content


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 patch_yolo_detector.py <yolo/object_detector.py>")
        sys.exit(1)

    p = Path(sys.argv[1])
    if not p.exists():
        print(f"Erreur : {p} introuvable")
        sys.exit(1)

    content = p.read_text()

    if already_patched(content):
        print("✅ Déjà patché — rien à faire.")
        sys.exit(0)

    bak = p.with_suffix(f".py.bak_{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(p, bak)
    print(f"📦 Backup : {bak}\n")
    print("Application des patches…")

    # Patch 1 : injecter le code d'init après super().__init__(...)
    new, n = re.subn(
        r'(super\(\)\.__init__\(\s*vis,\s*COMPONENT,\s*config\[CONFIG_OBJECT_DETECTOR\],\s*camera_identifier\s*\))',
        r'\1' + INIT_INJECT,
        content, count=1,
    )
    if n:
        content = new
        print("  ✅  __init__ inject")
    else:
        print("  ⚠️   __init__ inject : pattern non trouvé")

    # Patch 2 : remplacer tout depuis return_objects jusqu'à la fin
    idx = content.find("    def return_objects(self, frame: np.ndarray)")
    if idx == -1:
        print("  ⚠️   return_objects non trouvé")
    else:
        content = content[:idx] + NEW_TAIL
        print("  ✅  return_objects + nouvelles méthodes")

    # Vérification syntaxique
    try:
        ast.parse(content)
        print("  ✅  Syntaxe OK")
    except SyntaxError as e:
        print(f"  ❌  Erreur de syntaxe : {e}")
        sys.exit(1)

    p.write_text(content)
    print(f"\n🎉 {p} patché avec succès.")


if __name__ == "__main__":
    main()
