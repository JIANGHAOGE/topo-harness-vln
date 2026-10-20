from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.cluster.hierarchy import fclusterdata


SOURCE = Path(
    "/home/jianghaoge/CEE-AgentOS/logs/hierarchical/20260817_104439/"
    "observations/0073/topdown.png"
)
OUTPUT = Path(
    "/home/jianghaoge/.codex/workframes/2026-08-06_07-48-03_"
    "先明确定位：Agent_框架应参考_ABot-AgentOS，而不是_ABot-N1。"
    "ABot-N1_更适合作为未来替…_6dcb9600b124/visuals/topology_skeleton_prototype"
)
MAP_ID = "1st_floor_exh"
ORIGIN_XZ_M = (-14.679935455322266, -25.174999237060547)
METERS_PER_PIXEL = 0.09018882364034653


def transitions(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image, 1)
    neighbors = [
        padded[:-2, 1:-1],
        padded[:-2, 2:],
        padded[1:-1, 2:],
        padded[2:, 2:],
        padded[2:, 1:-1],
        padded[2:, :-2],
        padded[1:-1, :-2],
        padded[:-2, :-2],
    ]
    return sum(
        ((neighbors[index] == 0) & (neighbors[(index + 1) % 8] == 1)).astype(np.uint8)
        for index in range(8)
    )


def thin(image: np.ndarray) -> np.ndarray:
    result = image.copy().astype(np.uint8)
    while True:
        changed = 0
        for step in (0, 1):
            padded = np.pad(result, 1)
            neighbors = [
                padded[:-2, 1:-1],
                padded[:-2, 2:],
                padded[1:-1, 2:],
                padded[2:, 2:],
                padded[2:, 1:-1],
                padded[2:, :-2],
                padded[1:-1, :-2],
                padded[:-2, :-2],
            ]
            neighbor_count = sum(neighbors)
            crossing_count = transitions(result)
            north, _, east, _, south, _, west, _ = neighbors
            if step == 0:
                connectivity = (north * east * south == 0) & (east * south * west == 0)
            else:
                connectivity = (north * east * west == 0) & (north * south * west == 0)
            removable = (
                (result == 1)
                & (neighbor_count >= 2)
                & (neighbor_count <= 6)
                & (crossing_count == 1)
                & connectivity
            )
            changed += int(removable.sum())
            result[removable] = 0
        if changed == 0:
            return result


def navigable_mask(image: np.ndarray) -> np.ndarray:
    mask = np.all(image == np.array([92, 96, 92], dtype=np.uint8), axis=2).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    retained = [
        component
        for component in range(1, count)
        if stats[component, cv2.CC_STAT_AREA] >= 500
    ]
    return np.isin(labels, retained).astype(np.uint8)


def radial_branch_count(skeleton: np.ndarray, *, x_px: int, y_px: int, radius_px: int) -> int:
    y_grid, x_grid = np.indices(skeleton.shape)
    distance = np.hypot(x_grid - x_px, y_grid - y_px)
    ring = ((distance >= radius_px - 1) & (distance <= radius_px + 1) & (skeleton > 0)).astype(
        np.uint8
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(ring, 8)
    return int(sum(stats[component, cv2.CC_STAT_AREA] > 0 for component in range(1, count)))


def node_candidates(mask: np.ndarray, skeleton: np.ndarray) -> list[dict[str, object]]:
    clearance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    crossings = transitions(skeleton)
    branch_pixels = np.column_stack(np.where((skeleton > 0) & (crossings >= 3)))
    nodes: list[dict[str, object]] = []

    if len(branch_pixels):
        clusters = fclusterdata(branch_pixels, t=25, criterion="distance", method="single")
        for cluster_id in sorted(set(clusters)):
            pixels = branch_pixels[clusters == cluster_id]
            values = clearance[pixels[:, 0], pixels[:, 1]]
            y_px, x_px = pixels[int(np.argmax(values))]
            clearance_m = float(values.max() * METERS_PER_PIXEL)
            node_type = "open_space_center" if clearance_m >= 1.55 else "junction"
            branch_count = radial_branch_count(
                skeleton, x_px=int(x_px), y_px=int(y_px), radius_px=12
            )
            if node_type == "junction" and branch_count < 3:
                continue
            nodes.append(
                {
                    "node_type": node_type,
                    "x_px": int(x_px),
                    "y_px": int(y_px),
                    "clearance_m": round(clearance_m, 2),
                    "radial_branch_count": branch_count,
                    "branch_support": int(len(pixels)),
                }
            )

    local_maxima = (
        (clearance == cv2.dilate(clearance, np.ones((21, 21), np.uint8)))
        & (clearance >= 15)
    ).astype(np.uint8)
    count, labels, _, _ = cv2.connectedComponentsWithStats(local_maxima, 8)
    for component in range(1, count):
        ys, xs = np.where(labels == component)
        values = clearance[ys, xs]
        index = int(np.argmax(values))
        y_px, x_px = int(ys[index]), int(xs[index])
        if any(
            np.hypot(x_px - int(node["x_px"]), y_px - int(node["y_px"])) < 22
            for node in nodes
        ):
            continue
        nodes.append(
            {
                "node_type": "open_space_center",
                "x_px": x_px,
                "y_px": y_px,
                "clearance_m": round(float(values[index] * METERS_PER_PIXEL), 2),
                "radial_branch_count": radial_branch_count(
                    skeleton, x_px=x_px, y_px=y_px, radius_px=12
                ),
                "branch_support": 0,
            }
        )

    nodes.sort(key=lambda node: (int(node["y_px"]), int(node["x_px"])))
    counters = {"junction": 0, "open_space_center": 0}
    for node in nodes:
        node_type = str(node["node_type"])
        counters[node_type] += 1
        prefix = "J" if node_type == "junction" else "O"
        node["node_id"] = f"{prefix}{counters[node_type]:02d}"
        node["world_x_m"] = round(ORIGIN_XZ_M[0] + int(node["x_px"]) * METERS_PER_PIXEL, 3)
        node["world_z_m"] = round(ORIGIN_XZ_M[1] + int(node["y_px"]) * METERS_PER_PIXEL, 3)
    return nodes


def render(source: np.ndarray, mask: np.ndarray, skeleton: np.ndarray, nodes: list[dict[str, object]]) -> None:
    output = source.copy()
    output[mask > 0] = (58, 65, 64)
    wide_skeleton = cv2.dilate(skeleton, np.ones((2, 2), np.uint8))
    output[wide_skeleton > 0] = (238, 224, 55)
    for node in nodes:
        point = (int(node["x_px"]), int(node["y_px"]))
        color = (61, 196, 255) if node["node_type"] == "junction" else (214, 102, 238)
        cv2.circle(output, point, 7, color, -1, cv2.LINE_AA)
        cv2.circle(output, point, 8, (15, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(
            output,
            str(node["node_id"]),
            (point[0] + 9, point[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )

    ys, xs = np.where(mask > 0)
    x0, x1 = max(0, int(xs.min()) - 18), min(mask.shape[1], int(xs.max()) + 19)
    y0, y1 = max(0, int(ys.min()) - 18), min(mask.shape[0], int(ys.max()) + 19)
    raw_crop = cv2.cvtColor(source[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    overlay_crop = cv2.cvtColor(output[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    target_height = 930
    target_width = round(raw_crop.shape[1] * target_height / raw_crop.shape[0])
    raw_image = Image.fromarray(raw_crop).resize((target_width, target_height), Image.Resampling.NEAREST)
    overlay_image = Image.fromarray(overlay_crop).resize(
        (target_width, target_height), Image.Resampling.NEAREST
    )

    margin, gap, header = 52, 42, 150
    canvas = Image.new(
        "RGB",
        (margin * 2 + target_width * 2 + gap, header + target_height + margin),
        (9, 15, 16),
    )
    canvas.paste(raw_image, (margin, header))
    canvas.paste(overlay_image, (margin + target_width + gap, header))
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    title = ImageFont.truetype(bold_path, 28)
    label = ImageFont.truetype(bold_path, 18)
    body = ImageFont.truetype(font_path, 15)
    draw.text((margin, 28), "Static Navmesh to Backbone Topology", font=title, fill=(235, 242, 239))
    summary = (
        f"map_id={MAP_ID}   scale={METERS_PER_PIXEL:.3f} m/px   "
        f"junctions={sum(n['node_type'] == 'junction' for n in nodes)}   "
        f"open centers={sum(n['node_type'] == 'open_space_center' for n in nodes)}"
    )
    draw.text((margin, 70), summary, font=body, fill=(154, 174, 170))
    draw.text((margin, 112), "RAW NAVMESH", font=label, fill=(190, 205, 202))
    draw.text((margin + target_width + gap, 112), "EXTRACTED BACKBONE", font=label, fill=(190, 205, 202))
    legend_x = canvas.width - 450
    draw.line((legend_x, 81, legend_x + 38, 81), fill=(55, 224, 238), width=4)
    draw.text((legend_x + 48, 70), "medial skeleton", font=body, fill=(200, 215, 212))
    draw.ellipse((legend_x + 190, 73, legend_x + 206, 89), fill=(255, 196, 61))
    draw.text((legend_x + 214, 70), "junction", font=body, fill=(200, 215, 212))
    draw.ellipse((legend_x + 314, 73, legend_x + 330, 89), fill=(238, 102, 214))
    draw.text((legend_x + 338, 70), "open center", font=body, fill=(200, 215, 212))
    canvas.save(OUTPUT / "backbone_topology_comparison.png")
    Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)).save(OUTPUT / "backbone_topology_full.png")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = cv2.imread(str(SOURCE))
    if source is None:
        raise FileNotFoundError(SOURCE)
    mask = navigable_mask(source)
    skeleton = thin(mask)
    nodes = node_candidates(mask, skeleton)
    render(source, mask, skeleton, nodes)
    payload = {
        "prototype": True,
        "map_id": MAP_ID,
        "source": str(SOURCE),
        "origin_world_xz_m": ORIGIN_XZ_M,
        "meters_per_pixel": METERS_PER_PIXEL,
        "navigable_component_min_area_px": 500,
        "junction_merge_radius_px": 25,
        "open_center_min_clearance_px": 15,
        "skeleton_pixel_count": int(skeleton.sum()),
        "nodes": nodes,
    }
    (OUTPUT / "backbone_topology_nodes.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"node_count": len(nodes), "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
