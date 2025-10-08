import cv2
import numpy as np

# ================== 预设 & 参数 ==================
# 处理分辨率
CAP_W = 240
CAP_H = 240

# 白色：低S、高V；H不限（可在面板微调）
PRESETS = {
    "white": {"H_low": 0, "H_high": 179, "S_low": 0, "S_high": 70, "V_low": 175, "V_high": 255},
}
DRAW_COLORS = {"white": (0, 255, 0)}  # 叠加层线条颜色

INIT_MODE = "white"
INIT_MIN_AREA = 300      # 轮廓最小面积
WARP_SIZE = 200          # 透视展开尺寸（正方形边长）
CENTER_EXCL = 0          # 忽略展开图中线附近像素带(0 关闭；8~12 可减抖)
GAUSS_K = 7              # 高斯核
KERNEL_SZ = 3            # 形态学核大小
CLOSE_ITERS = 2          # 闭运算次数
DILATE_ITERS = 1         # 膨胀次数

# ================== 工具函数 ==================
def nothing(x): pass

def set_trackbar_values(win, cfg, min_area):
    cv2.setTrackbarPos("H low",  win, cfg["H_low"])
    cv2.setTrackbarPos("H high", win, cfg["H_high"])
    cv2.setTrackbarPos("S low",  win, cfg["S_low"])
    cv2.setTrackbarPos("S high", win, cfg["S_high"])
    cv2.setTrackbarPos("V low",  win, cfg["V_low"])
    cv2.setTrackbarPos("V high", win, cfg["V_high"])
    cv2.setTrackbarPos("min area", win, min_area)

def get_trackbar_values(win):
    h_low  = cv2.getTrackbarPos("H low",  win)
    h_high = cv2.getTrackbarPos("H high", win)
    s_low  = cv2.getTrackbarPos("S low",  win)
    s_high = cv2.getTrackbarPos("S high", win)
    v_low  = cv2.getTrackbarPos("V low",  win)
    v_high = cv2.getTrackbarPos("V high", win)
    min_area = cv2.getTrackbarPos("min area", win)
    # 合法化
    s_low, v_low = min(s_low, s_high), min(v_low, v_high)
    return (h_low, h_high, s_low, s_high, v_low, v_high, max(0, min_area))

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

# ================== 摄像头 & 面板 ==================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头"); raise SystemExit

# 尝试设置硬分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

ctrl_win = "Controls"
cv2.namedWindow(ctrl_win)
cv2.createTrackbar("H low",   ctrl_win, 0,   179, nothing)
cv2.createTrackbar("H high",  ctrl_win, 179, 179, nothing)
cv2.createTrackbar("S low",   ctrl_win, 0,   255, nothing)
cv2.createTrackbar("S high",  ctrl_win, 70,  255, nothing)
cv2.createTrackbar("V low",   ctrl_win, 175, 255, nothing)
cv2.createTrackbar("V high",  ctrl_win, 255, 255, nothing)
cv2.createTrackbar("min area",ctrl_win, INIT_MIN_AREA, 10000, nothing)

mode = INIT_MODE
set_trackbar_values(ctrl_win, PRESETS[mode], INIT_MIN_AREA)

print("热键：1=white，S=打印阈值，Q=退出（仅显示：Overlay & Binary）")

# ================== 主循环 ==================
while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取帧"); break

    # —— 保证处理分辨率为 240x240 ——
    frame = cv2.resize(frame, (CAP_W, CAP_H), interpolation=cv2.INTER_AREA)

    # HSV & 观感增强
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_clahe = clahe.apply(v)
    frame_enhanced = cv2.cvtColor(cv2.merge((h, s, v_clahe)), cv2.COLOR_HSV2BGR)

    # 阈值
    h_low, h_high, s_low, s_high, v_low, v_high, min_area = get_trackbar_values(ctrl_win)
    mask = make_mask_hsv(hsv, h_low, h_high, s_low, s_high, v_low, v_high)

    # —— 形态学（对空心框友好）——
    mask_blur = cv2.GaussianBlur(mask, (GAUSS_K, GAUSS_K), 0)
    _, binary = cv2.threshold(mask_blur, 128, 255, cv2.THRESH_BINARY)
    kernel = np.ones((KERNEL_SZ, KERNEL_SZ), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITERS)
    binary = cv2.dilate(binary, kernel, iterations=DILATE_ITERS)

    # 轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detection_layer = np.zeros_like(frame)

    # —— 寻找候选矩形（先旋转矩形，后兜底轴对齐）——
    best_poly = None      # 多边形(4点)用于内部mask
    best_bbox = None      # (x,y,w,h) 用于可视化
    max_rect_area = 0

    # 1) 旋转矩形（更贴合真实方向）
    for c in contours:
        if cv2.contourArea(c) <= max(min_area, 1):
            continue
        rect = cv2.minAreaRect(c)             # ((cx,cy),(w,h),angle)
        (cx, cy), (rw, rh), ang = rect
        if rw < 20 or rh < 20:
            continue
        box = cv2.boxPoints(rect).astype(np.int32)  # 4x2
        rect_area = rw * rh
        if rect_area > max_rect_area:
            max_rect_area = rect_area
            best_poly = box.reshape(4, 1, 2)
            x, y, w, h = cv2.boundingRect(best_poly)
            best_bbox = (x, y, w, h)

    # 2) 兜底：最大连通域的外接矩形
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

    # —— 在 ROI 内部做左右判定 ——
    direction_text = "LEFT"     # 默认值（避免为空）
    conf = 0.0

    if best_poly is not None:
        x, y, w, h = best_bbox

        # 可视化：外接框 + 多边形
        cv2.rectangle(detection_layer, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.polylines(detection_layer, [best_poly.reshape(-1, 2)], True, (0, 200, 0), 2)

        # 内部白色：多边形填充 ∩ 白色掩码
        poly_mask_full = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillPoly(poly_mask_full, [best_poly.reshape(-1, 2)], 255)
        inner_white = cv2.bitwise_and(mask, poly_mask_full)

        # 透视展开
        src = order_quad_pts(best_poly)
        dst = np.array([[0,0],[WARP_SIZE-1,0],[WARP_SIZE-1,WARP_SIZE-1],[0,WARP_SIZE-1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(inner_white, M, (WARP_SIZE, WARP_SIZE))

        # 可选：忽略中线附近像素
        if CENTER_EXCL > 0:
            cx0 = WARP_SIZE//2 - CENTER_EXCL//2
            cx1 = WARP_SIZE//2 + CENTER_EXCL//2
            warped[:, cx0:cx1] = 0

        # 左右计数（必选左右：平局偏 LEFT；想偏 RIGHT 就把 >= 改成 >）
        L = int(cv2.countNonZero(warped[:, :WARP_SIZE//2]))
        R = int(cv2.countNonZero(warped[:, WARP_SIZE//2:]))
        direction_text = "LEFT" if L >= R else "RIGHT"

        total = max(L + R, 1)
        conf = abs(L - R) / float(total)  # 相对差当作“置信度”
        absdiff = abs(L - R)

        # 可视化计数与中线
        midx = x + w // 2
        cv2.line(detection_layer, (midx, y), (midx, y+h), (255, 255, 0), 2)
        cv2.putText(detection_layer, f"L:{L}", (x+5, y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
        cv2.putText(detection_layer, f"R:{R}", (x+w-120, y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
        cv2.putText(detection_layer, f"rel:{conf:.3f} abs:{absdiff}",
                    (x+5, y+45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,255), 2, cv2.LINE_AA)

    # 叠加显示（只显示 Overlay 和 Binary）
    overlay = cv2.addWeighted(frame, 1.0, detection_layer, 1.0, 0)
    cv2.putText(overlay, f"FINAL: {direction_text} ({conf*100:.1f}%)", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)

    cv2.imshow("Overlay", overlay)
    cv2.imshow("Binary", binary)

    # 按键
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        print(f"[{mode}] H:[{h_low},{h_high}] S:[{s_low},{s_high}] V:[{v_low},{v_high}] min_area:{min_area}  "
              f"GAUSS={GAUSS_K}, KERNEL={KERNEL_SZ}, CLOSE={CLOSE_ITERS}, DILATE={DILATE_ITERS}  "
              f"RES={CAP_W}x{CAP_H}")
    elif key == ord('1'):
        mode = "white"
        set_trackbar_values(ctrl_win, PRESETS[mode], min_area)

cap.release()
cv2.destroyAllWindows()
