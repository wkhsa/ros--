import cv2
import numpy as np

# ================== 预设 & 参数 ==================
CAP_W = 240
CAP_H = 240

PRESETS = {
    "white": {"H_low": 0, "H_high": 179, "S_low": 0, "S_high": 70, "V_low": 175, "V_high": 255},
}
DRAW_COLORS = {"white": (0, 255, 0)}  # 仅保留以防后续可视化扩展

INIT_MODE = "white"
INIT_MIN_AREA = 300
WARP_SIZE = 200
CENTER_EXCL = 0
GAUSS_K = 7
KERNEL_SZ = 3
CLOSE_ITERS = 2
DILATE_ITERS = 1

# 用黑色占比判定（保持你原来的逻辑）
TARGET = "black"

def make_mask_hsv(hsv, h_low, h_high, s_low, s_high, v_low, v_high):
    """支持 H 跨界"""
    lower1 = np.array([min(h_low, h_high), s_low, v_low], dtype=np.uint8)
    upper1 = np.array([max(h_low, h_high), s_high, v_high], dtype=np.uint8)
    if h_low <= h_high:
        return cv2.inRange(hsv, lower1, upper1)
    else:
        lowerA = np.array([0,      s_low, v_low], dtype=np.uint8)
        upperA = np.array([h_high, s_high, v_high], dtype=np.uint8)
        lowerB = np.array([h_low,  s_low, v_low], dtype=np.uint8)
        upperB = np.array([179,    s_high, v_high], dtype=np.uint8)
        return cv2.bitwise_or(cv2.inRange(hsv, lowerA, upperA),
                              cv2.inRange(hsv, lowerB, upperB))

def order_quad_pts(pts4):
    """把4点按 tl,tr,br,bl 排序（用于透视）"""
    pts = pts4.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

# ================== 固定参数（替代滑块） ==================
cfg = PRESETS[INIT_MODE]
H_low, H_high = cfg["H_low"], cfg["H_high"]
S_low, S_high = cfg["S_low"], cfg["S_high"]
V_low, V_high = cfg["V_low"], cfg["V_high"]
min_area = INIT_MIN_AREA
ratio_thr_pct = 70         # 按你的要求：ratio% = 70
ratio_thr = 0.70

# ================== 摄像头 ==================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头"); raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

print("运行中（无窗口/无滑块）。按 Ctrl+C 结束。当前参数：")
print(f"H:[{H_low},{H_high}] S:[{S_low},{S_high}] V:[{V_low},{V_high}] "
      f"min_area:{min_area} ratio_thr:{ratio_thr_pct}% TARGET:{TARGET} "
      f"GAUSS={GAUSS_K}, KERNEL={KERNEL_SZ}, CLOSE={CLOSE_ITERS}, DILATE={DILATE_ITERS} "
      f"RES={CAP_W}x{CAP_H}")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧"); break

        # 统一分辨率
        frame = cv2.resize(frame, (CAP_W, CAP_H), interpolation=cv2.INTER_AREA)

        # ===== HSV & 观感增强 =====
        hsv0 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h0, s0, v0 = cv2.split(hsv0)
        v_clahe = clahe.apply(v0)
        hsv = cv2.merge((h0, s0, v_clahe))

        # 阈值 & 二值
        mask = make_mask_hsv(hsv, H_low, H_high, S_low, S_high, V_low, V_high)
        mask_blur = cv2.GaussianBlur(mask, (GAUSS_K, GAUSS_K), 0)
        _, binary = cv2.threshold(mask_blur, 128, 255, cv2.THRESH_BINARY)
        kernel = np.ones((KERNEL_SZ, KERNEL_SZ), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITERS)
        binary = cv2.dilate(binary, kernel, iterations=DILATE_ITERS)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_poly = None
        best_bbox = None
        max_rect_area = 0

        # 旋转矩形优先
        for c in contours:
            if cv2.contourArea(c) <= max(min_area, 1):
                continue
            rect = cv2.minAreaRect(c)
            (cx, cy), (rw, rh), ang = rect
            if rw < 20 or rh < 20:
                continue
            box = cv2.boxPoints(rect).astype(np.int32)
            rect_area = rw * rh
            if rect_area > max_rect_area:
                max_rect_area = rect_area
                best_poly = box.reshape(4, 1, 2)
                x, y, w, h = cv2.boundingRect(best_poly)
                best_bbox = (x, y, w, h)

        # 兜底：最大连通域外接矩形
        if best_poly is None and len(contours) > 0:
            c_big = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c_big) > max(min_area, 1):
                x, y, w, h = cv2.boundingRect(c_big)
                if w >= 20 and h >= 20:
                    best_bbox = (x, y, w, h)
                    best_poly = np.array(
                        [[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]],
                        dtype=np.int32
                    )

        decision_text = "B"
        white_ratio = 0.0

        if best_poly is not None:
            # 多边形 mask（ROI）
            poly_mask_full = np.zeros(mask.shape, dtype=np.uint8)
            cv2.fillPoly(poly_mask_full, [best_poly.reshape(-1, 2)], 255)

            # 使用形态学后的 binary 统计
            inner_white = cv2.bitwise_and(binary, poly_mask_full)

            # 透视展开（严格二值）
            src = order_quad_pts(best_poly)
            dst = np.array([[0,0],[WARP_SIZE-1,0],[WARP_SIZE-1,WARP_SIZE-1],[0,WARP_SIZE-1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(inner_white, M, (WARP_SIZE, WARP_SIZE), flags=cv2.INTER_NEAREST)
            _, warped = cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY)

            poly_warped = cv2.warpPerspective(poly_mask_full, M, (WARP_SIZE, WARP_SIZE), flags=cv2.INTER_NEAREST)
            _, poly_warped = cv2.threshold(poly_warped, 127, 255, cv2.THRESH_BINARY)

            if CENTER_EXCL > 0:
                cx0 = WARP_SIZE//2 - CENTER_EXCL//2
                cx1 = WARP_SIZE//2 + CENTER_EXCL//2
                warped[:, cx0:cx1] = 0
                poly_warped[:, cx0:cx1] = 0

            white_pixels = int(cv2.countNonZero(warped))
            roi_pixels   = int(cv2.countNonZero(poly_warped))
            total_pixels = max(roi_pixels, 1)
            white_ratio  = white_pixels / float(total_pixels)
            black_ratio  = 1.0 - white_ratio

            fill_ratio = white_ratio if TARGET == "white" else black_ratio
            decision_text = "A" if fill_ratio >= ratio_thr else "B"

            # 控制台日志（每帧打印可能太多，可按需限频）
            print(f"FINAL: {decision_text}  white={white_ratio*100:.1f}%  "
                  f"black={black_ratio*100:.1f}%  thr={ratio_thr_pct}% ({TARGET})")

        else:
            # 没检测到候选
            print("未检测到有效候选区域；维持 B")

except KeyboardInterrupt:
    print("\n手动结束。")

finally:
    cap.release()
    cv2.destroyAllWindows()