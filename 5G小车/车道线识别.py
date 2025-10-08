import cv2
import numpy as np
import math

# ---------- 基础工具（原样保留） ----------
def auto_canny(gray, sigma=0.33):
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)

def color_lane_mask(bgr):
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    white = cv2.inRange(hls, (0, 190, 0), (179, 255, 255))
    yellow = cv2.inRange(hls, (15, 110, 70), (45, 255, 255))
    return cv2.bitwise_or(white, yellow)

def angle_ok(x1, y1, x2, y2, min_deg=10, max_deg=88):
    dx, dy = (x2 - x1), (y2 - y1)
    if dx == 0 and dy == 0:
        return False
    ang = abs(math.degrees(math.atan2(dy, dx)))
    ang = 180 - ang if ang > 90 else ang
    return (min_deg <= ang <= max_deg)

def line_len(x1, y1, x2, y2):
    return math.hypot(x2-x1, y2-y1)

def fit_line(points, weights=None):
    if len(points) < 2:
        return None
    x = points[:,0]
    y = points[:,1]
    if weights is None:
        a, b = np.polyfit(x, y, 1)
    else:
        a, b = np.polyfit(x, y, 1, w=weights)
    return a, b

def intersect_with_y(a, b, Y):
    if abs(a) < 1e-6:
        return None
    return (Y - b) / a

# ---------- 单张图处理：返回 (edges, roi_edges, 叠加结果) ----------
def process_image(path, use_color_prior=True):
    img = cv2.imread(path)
    assert img is not None, f"读取图像失败: {path}"
    h, w = img.shape[:2]

    # 预处理
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    edges = auto_canny(gray_eq, sigma=0.40)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

    # ROI（梯形）
    mask = np.zeros((h, w), dtype=np.uint8)
    top_y = int(h * 0.28)
    bottom_y = h - 1
    top_left  = (int(w * 0.10), top_y)
    top_right = (int(w * 0.90), top_y)
    bot_left  = (int(w * 0.02), bottom_y)
    bot_right = (int(w * 0.98), bottom_y)
    poly = np.array([top_left, top_right, bot_right, bot_left], dtype=np.int32)
    cv2.fillPoly(mask, [poly], 255)

    if use_color_prior:
        cmask = color_lane_mask(img)
        masked_edges = cv2.bitwise_and(edges, cmask)
    else:
        masked_edges = edges

    roi_edges = cv2.bitwise_and(masked_edges, mask)

    # Hough
    def hough_once(ed):
        return cv2.HoughLinesP(
            ed, 1, np.pi/180,
            threshold=14,
            minLineLength=45,
            maxLineGap=40
        )

    lines_all = []
    L1 = hough_once(roi_edges)
    if L1 is not None: lines_all.append(L1)

    scale = 0.75
    small = cv2.resize(roi_edges, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    L2 = hough_once(small)
    if L2 is not None:
        L2 = (L2 / scale).astype(np.int32)
        lines_all.append(L2)

    if len(lines_all):
        lines = np.vstack(lines_all)
    else:
        lines = None

    # 只保留右车道点（如需恢复主线拟合可使用 right_pts/right_w）
    right_pts, right_w = [], []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:,0]:
            if not angle_ok(x1,y1,x2,y2, 10, 88):
                continue
            dx = (x2-x1)
            dy = (y2-y1)
            slope = dy/dx if abs(dx) > 1e-6 else 1e6
            seg_len = line_len(x1,y1,x2,y2)
            if slope > 0:
                right_pts.extend([(x1,y1),(x2,y2)])
                right_w.extend([seg_len, seg_len])

    img_draw = img.copy()
    if lines is not None:
        for x1, y1, x2, y2 in lines[:,0]:
            if angle_ok(x1,y1,x2,y2, 10, 88):
                cv2.line(img_draw, (x1,y1), (x2,y2), (255, 0, 0), 1)

    return img, edges, roi_edges, img_draw

# ---------- 可视化拼接 ----------
def label(img, text):
    out = img.copy()
    # 统一转 BGR 显示（单通道转三通道）
    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0,0,0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    return out

def make_panel(orig, edges, roi_edges, overlay, title):
    # 统一缩放（便于拼接）
    target_w = 640
    scale = target_w / orig.shape[1]
    def rs(x):
        return cv2.resize(x, (int(x.shape[1]*scale), int(x.shape[0]*scale)))

    a = label(rs(orig),      f"{title} - 原图")
    b = label(rs(edges),     f"{title} - edges")
    c = label(rs(roi_edges), f"{title} - ROI edges")
    d = label(rs(overlay),   f"{title} - 叠加结果")

    top  = cv2.hconcat([a, b])
    bottom = cv2.hconcat([c, d])

    # 对齐高度（防止因奇偶造成 1px 差异）
    h_min = min(top.shape[0], bottom.shape[0])
    top = top[:h_min]
    bottom = bottom[:h_min]

    panel = cv2.vconcat([top, bottom])
    return panel

if __name__ == "__main__":
    paths = [("2.png", "图A"), ("5_2.jpg", "图B")]

    panels = []
    for p, t in paths:
        orig, edges, roi_edges, overlay = process_image(p, use_color_prior=True)
        panel = make_panel(orig, edges, roi_edges, overlay, t)
        panels.append(panel)

    # 让两张图的面板等高后并排
    h = min(p.shape[0] for p in panels)
    panels = [p[:h] for p in panels]
    big = cv2.hconcat(panels)

    cv2.imshow("两张图的车道线识别（并排）", big)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
