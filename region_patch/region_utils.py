"""
viseron/domains/object_detector/region_utils.py
================================================
Frigate-style region-based detection for Viseron.

Principe :
  Au lieu d'envoyer la frame entière à YOLO, on :
    1. Récupère les contours de mouvement MOG2
    2. Construit des régions carrées autour des zones de mouvement
    3. Envoie uniquement ces crops à YOLO (batch GPU)
    4. Remet les coordonnées détectées dans le repère de la frame complète

Gestion de la résolution MOG2 :
  MOG2 travaille à 300x300 (par défaut dans Viseron).
  Il retourne des `rel_contours` en coordonnées relatives [0,1].
  On rescale ces contours vers la résolution de la frame de détection
  (ex: substream 640x360) avant de calculer les régions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

# (x1, y1, x2, y2) en pixels absolus
BBox = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Region dataclass
# ---------------------------------------------------------------------------

@dataclass
class Region:
    """Région carrée d'intérêt dans la frame."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """Retourne le sous-tableau numpy correspondant à cette région."""
        return frame[self.y1:self.y2, self.x1:self.x2]

    def remap_detection(
        self,
        rel_x1: float, rel_y1: float,
        rel_x2: float, rel_y2: float,
        frame_width: int, frame_height: int,
    ) -> Tuple[float, float, float, float]:
        """
        Convertit des coordonnées relatives au crop en coordonnées relatives
        à la frame complète.

        YOLO retourne des coords relatives au crop (0..1).
        On ajoute region.x1/y1 pour revenir dans l'espace de la frame entière,
        puis on normalise par la résolution de la frame.
        """
        abs_x1 = rel_x1 * self.width  + self.x1
        abs_y1 = rel_y1 * self.height + self.y1
        abs_x2 = rel_x2 * self.width  + self.x1
        abs_y2 = rel_y2 * self.height + self.y1

        # Clamp dans les limites de la frame
        abs_x1 = max(0.0, min(abs_x1, frame_width))
        abs_y1 = max(0.0, min(abs_y1, frame_height))
        abs_x2 = max(0.0, min(abs_x2, frame_width))
        abs_y2 = max(0.0, min(abs_y2, frame_height))

        return (
            abs_x1 / frame_width,
            abs_y1 / frame_height,
            abs_x2 / frame_width,
            abs_y2 / frame_height,
        )

    def is_detection_cut_off(
        self,
        rel_x1: float, rel_y1: float,
        rel_x2: float, rel_y2: float,
        edge_threshold: float = 0.05,
    ) -> bool:
        """
        Retourne True si la détection touche le bord du crop.
        Dans ce cas, on agrandit le crop et on relance la détection
        pour ne pas rater la partie de l'objet qui dépasse.
        """
        return (
            rel_x1 < edge_threshold
            or rel_y1 < edge_threshold
            or rel_x2 > 1.0 - edge_threshold
            or rel_y2 > 1.0 - edge_threshold
        )


# ---------------------------------------------------------------------------
# Helpers : contours → bboxes
# ---------------------------------------------------------------------------

def contours_to_bboxes(contours: list) -> List[BBox]:
    """Convertit des contours OpenCV en liste de (x1,y1,x2,y2)."""
    bboxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        bboxes.append((x, y, x + w, y + h))
    return bboxes


def merge_overlapping_bboxes(bboxes: List[BBox]) -> List[BBox]:
    """
    Fusionne les bboxes qui se chevauchent ou se touchent.
    Évite de créer deux régions séparées pour le même objet physique
    (ex: tête + jambes détectées comme deux zones de mouvement distinctes).
    """
    if not bboxes:
        return []
    merged = list(bboxes)
    changed = True
    while changed:
        changed = False
        result: List[BBox] = []
        used = [False] * len(merged)
        for i, (ax1, ay1, ax2, ay2) in enumerate(merged):
            if used[i]:
                continue
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                bx1, by1, bx2, by2 = merged[j]
                if ax1 <= bx2 and ax2 >= bx1 and ay1 <= by2 and ay2 >= by1:
                    ax1, ay1 = min(ax1, bx1), min(ay1, by1)
                    ax2, ay2 = max(ax2, bx2), max(ay2, by2)
                    used[j] = True
                    changed = True
            result.append((ax1, ay1, ax2, ay2))
            used[i] = True
        merged = result
    return merged


def bbox_to_square_region(
    bbox: BBox,
    frame_width: int,
    frame_height: int,
    margin: float = 0.15,
    min_size_ratio: float = 0.10,
) -> Region:
    """
    Transforme un bbox de mouvement en une région carrée avec marge,
    centrée sur le centroïde du mouvement, clampée dans la frame.

    Args:
        bbox           : (x1,y1,x2,y2) bbox de mouvement en pixels
        frame_width/height : dimensions de la frame de détection
        margin         : marge ajoutée de chaque côté (défaut 15%)
        min_size_ratio : taille minimale du crop = fraction du plus court côté
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    # Taille = max(width, height) + marge
    size = max(x2 - x1, y2 - y1)
    size = int(size * (1.0 + 2 * margin))

    # Taille minimale pour éviter les micro-crops sur du bruit
    min_px = int(min(frame_width, frame_height) * min_size_ratio)
    size = max(size, min_px)

    rx1 = cx - size // 2
    ry1 = cy - size // 2
    rx2 = rx1 + size
    ry2 = ry1 + size

    # Shift avant clamp pour conserver la forme carrée
    if rx1 < 0:            rx2 -= rx1;                rx1 = 0
    if ry1 < 0:            ry2 -= ry1;                ry1 = 0
    if rx2 > frame_width:  rx1 -= rx2 - frame_width;  rx2 = frame_width
    if ry2 > frame_height: ry1 -= ry2 - frame_height; ry2 = frame_height

    return Region(max(0, rx1), max(0, ry1), min(frame_width, rx2), min(frame_height, ry2))


def expand_region(region: Region, factor: float, frame_width: int, frame_height: int) -> Region:
    """
    Agrandit une région d'un facteur donné (ex: 1.5 = +50%), même centroïde.
    Utilisé quand un objet est coupé au bord du crop.
    """
    cx = (region.x1 + region.x2) // 2
    cy = (region.y1 + region.y2) // 2
    half = int(max(region.width, region.height) * factor / 2)
    return bbox_to_square_region(
        (cx - half, cy - half, cx + half, cy + half),
        frame_width, frame_height, margin=0.0,
    )


def _scale_rel_contours(contours: list, frame_width: int, frame_height: int) -> list:
    """
    Rescale les contours relatifs [0,1] vers des pixels absolus.

    MOG2 travaille à 300x300, Contours.rel_contours divise par (300, 300).
    On multiplie par la résolution de la frame de détection (substream ou main)
    pour avoir des coordonnées exploitables par cv2.boundingRect.
    """
    scaled = []
    for contour in contours:
        sc = contour.copy().astype(float)
        sc[:, :, 0] *= frame_width   # x * largeur frame
        sc[:, :, 1] *= frame_height  # y * hauteur frame
        scaled.append(sc.astype(np.int32))
    return scaled


def compute_regions_from_contours(
    contours: list,
    frame_width: int,
    frame_height: int,
    margin: float = 0.15,
    min_size_ratio: float = 0.10,
) -> List[Region]:
    """
    Pipeline complet : contours → bboxes fusionnés → régions carrées.

    Accepte automatiquement :
      - Contours relatifs [0,1]  (Contours.rel_contours de Viseron)
      - Contours absolus en pixels (cas moins courant)

    Limite à 3 régions maximum pour éviter la surcharge CPU/GPU.

    Args:
        contours       : liste de contours (relatifs ou absolus)
        frame_width/height : résolution de la frame passée à YOLO
        margin         : marge autour de chaque bbox fusionné
        min_size_ratio : taille minimale du crop

    Returns:
        Liste de Region prêtes à être croppées et envoyées au détecteur.
    """
    if not contours:
        return []

    # Détection automatique : relatif [0,1] ou absolu ?
    first = contours[0]
    if first.size > 0 and float(first.max()) <= 1.0:
        # Contours relatifs → rescale vers pixels absolus de la frame
        contours = _scale_rel_contours(contours, frame_width, frame_height)

    raw    = contours_to_bboxes(contours)
    merged = merge_overlapping_bboxes(raw)

    regions = [
        bbox_to_square_region(b, frame_width, frame_height, margin, min_size_ratio)
        for b in merged
    ]

    # Limite à 3 régions (les plus grandes = plus susceptibles de contenir un objet)
    if len(regions) > 3:
        regions = sorted(regions, key=lambda r: r.width * r.height, reverse=True)[:3]

    LOGGER.debug(
        "Contours %d → merged %d → regions %d",
        len(contours), len(merged), len(regions),
    )
    return regions


# ---------------------------------------------------------------------------
# Déduplication IoU
# ---------------------------------------------------------------------------

def _iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> float:
    """Intersection over Union entre deux bboxes."""
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def deduplicate_detections(objects: list, iou_threshold: float = 0.5) -> list:
    """
    Supprime les détections en double issues de régions qui se chevauchent.
    Garde la détection avec la meilleure confiance pour chaque (label, position).

    Requiert des objets avec : .label .confidence .rel_x1 .rel_y1 .rel_x2 .rel_y2
    """
    if len(objects) <= 1:
        return objects
    objects = sorted(objects, key=lambda o: o.confidence, reverse=True)
    kept = []
    for obj in objects:
        if not any(
            k.label == obj.label and
            _iou(obj.rel_x1, obj.rel_y1, obj.rel_x2, obj.rel_y2,
                 k.rel_x1,   k.rel_y1,   k.rel_x2,   k.rel_y2) > iou_threshold
            for k in kept
        ):
            kept.append(obj)
    return kept
