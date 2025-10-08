#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
import sys

# ================== 参数区域 ==================
# 摄像头尺寸
CAP_W, CAP_H = 240, 240

# HSV 预设（检测白色区域）
PRESETS = {
    "white": {"H_low": 0, "H_high": 179, "S_low": 0, "S_high": 70, "V_low": 175, "V_high": 255},
}
INIT_MODE = "white"

# 形态学/阈值
GAUSS_K = 7          # 高斯核，必须奇数
KERNEL_SZ = 3        # 形态学核，必须奇数
CLOSE_ITERS = 2
DILATE_ITERS = 1

# 候选与统计
INIT_MIN_AREA = 300
WARP_SIZE = 200
CENTER_EXCL = 0
TARGET = "black"     # 用黑色占比判定（保留你的逻辑）
ratio_thr_pct = 70
ratio_thr = 0.60

# 日志与决策平滑
LOG_INTERVAL = 0.2   # s
EMA_ALPHA = 0.3      # 滑动指数平均，抑制抖动

# ================== 工具函数 ==================
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

def odd(x):
    return x if x % 2 else x + 1

# ================== 主控制类（纯视觉，无任何 GPIO） ==================
class ABFollower:
    def __init__(self, cam_index=0):
        # --- OpenCV 预处理器 ---
        self.cfg = PRESETS[INIT_MODE]
        self.H_low = self.cfg["H_low"]; self.H_high = self.cfg["H_high"]
        self.S_low = self.cfg["S_low"]; self.S_high = self.cfg["S_high"]
        self.V_low = self.cfg["V_low"]; self.V_high = self.cfg["V_high"]

        self.min_area = INIT_MIN_AREA
        self.gauss_k = odd(GAUSS_K)
        self.kernel_sz = odd(KERNEL_SZ)

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cv2.setUseOptimized(True)
        try:
            cv2.ocl.setUseOpenCL(True)
        except:
            pass

        # --- 摄像头 ---
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            print("无法打开摄像头"); sys.exit(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except:
            pass

        # --- 运行态 ---
        self.last_log = 0
        self.ema = None

        print("运行中（A=左，B=右，未识别=直行；已去除舵机/电机控制，只做视觉与打印）。按 Ctrl+C 或按 q 结束。")
        print(f"H:[{self.H_low},{self.H_high}] S:[{self.S_low},{self.S_high}] V:[{self.V_low},{self.V_high}] "
              f"min_area:{self.min_area} thr:{ratio_thr_pct}% TARGET:{TARGET} "
              f"GAUSS={self.gauss_k}, KERNEL={self.kernel_sz}, CLOSE={CLOSE_ITERS}, DILATE={DILATE_ITERS} "
              f"RES={CAP_W}x{CAP_H}")

    # --- 视觉处理 & 决策 ---
    def process_frame(self, frame):
        """返回决策(A/B/None)、白/黑占比"""
        frame = cv2.resize(frame, (CAP_W, CAP_H), interpolation=cv2.INTER_AREA)

        hsv0 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h0, s0, v0 = cv2.split(hsv0)
        v_clahe = self.clahe.apply(v0)
        hsv = cv2.merge((h0, s0, v_clahe))

        mask = make_mask_hsv(hsv, self.H_low, self.H_high, self.S_low, self.S_high, self.V_low, self.V_high)
        mask_blur = cv2.GaussianBlur(mask, (self.gauss_k, self.gauss_k), 0)
        _, binary = cv2.threshold(mask_blur, 128, 255, cv2.THRESH_BINARY)
        kernel = np.ones((self.kernel_sz, self.kernel_sz), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITERS)
        binary = cv2.dilate(binary, kernel, iterations=DILATE_ITERS)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_poly = None
        max_rect_area = 0

        for c in contours:
            if cv2.contourArea(c) <= max(self.min_area, 1):
                continue
            rect = cv2.minAreaRect(c)
            (cx, cy), (rw, rh), ang = rect
            if rw < 20 or rh < 20:
                continue
            box = cv2.boxPoints(rect).astype(np.int32)
            # 额外过滤：填充率/长宽比
            x, y, w, h = cv2.boundingRect(box)
            if w*h == 0:
                continue
            fill = cv2.contourArea(c) / float(w*h)
            ar = max(w, h) / max(1.0, min(w, h))
            if fill < 0.3 or ar > 4.0:
                continue
            rect_area = rw * rh
            if rect_area > max_rect_area:
                max_rect_area = rect_area
                best_poly = box.reshape(4, 1, 2)

        # 兜底：最大连通域外接矩形
        if best_poly is None and len(contours) > 0:
            c_big = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c_big) > max(self.min_area, 1):
                x, y, w, h = cv2.boundingRect(c_big)
                if w >= 20 and h >= 20:
                    best_poly = np.array(
                        [[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]],
                        dtype=np.int32
                    )

        if best_poly is None:
            return None, 0.0, 1.0, frame, binary  # 未识别

        poly_mask_full = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillPoly(poly_mask_full, [best_poly.reshape(-1, 2)], 255)

        inner_white = cv2.bitwise_and(binary, poly_mask_full)

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

        # 平滑（EMA）
        fill_ratio = white_ratio if TARGET == "white" else black_ratio
        if self.ema is None:
            self.ema = fill_ratio
        else:
            self.ema = EMA_ALPHA*fill_ratio + (1-EMA_ALPHA)*self.ema

        decision = "A" if self.ema >= ratio_thr else "B"

        # 画辅助线（可注释）
        box_draw = best_poly.reshape(-1,2)
        cv2.polylines(frame, [box_draw], True, (0,255,0), 2)
        cv2.putText(frame, f"white={white_ratio*100:.1f}% black={black_ratio*100:.1f}% ema={self.ema*100:.1f}%",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        return decision, white_ratio, black_ratio, frame, binary

    # --- 主循环（不做任何硬件控制） ---
    def run(self, show=True):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("无法读取帧"); break

                decision, wr, br, vis_frame, binary = self.process_frame(frame)

                # 动作含义：A=左，B=右，None=直行（仅打印，不控制舵机）
                if decision == "A":
                    action = "LEFT"
                elif decision == "B":
                    action = "RIGHT"
                else:
                    action = "STRAIGHT"

                # 限频日志
                now = time.time()
                if now - self.last_log >= LOG_INTERVAL:
                    print(f"DECISION={decision or 'None'}  action={action}  "
                          f"white={wr*100:.1f}%  black={(1-wr)*100:.1f}%  thr={ratio_thr_pct}% ({TARGET})")
                    self.last_log = now

                if show:
                    cv2.imshow("Frame", vis_frame)
                    cv2.imshow("Binary", binary)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except KeyboardInterrupt:
            print("\n手动结束")
        finally:
            self.cleanup()

    def cleanup(self):
        # 仅释放资源（无 GPIO）
        self.cap.release()
        cv2.destroyAllWindows()

# ================== 入口 ==================
if __name__ == "__main__":
    follower = ABFollower(cam_index=0)
    follower.run(show=True)

