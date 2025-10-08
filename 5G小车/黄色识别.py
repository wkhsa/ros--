#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math

class LaneDetector:
    def __init__(self, cam_index=0, width=240, height=240):
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width, self.height = width, height

        # 状态与去抖（黄色）
        self.yellow_present_streak = 0
        self.yellow_absent_streak  = 0
        self.yellow_state = False  # True 时才认为“有黄色”，才会画黄线与用中点

        # 窗口
        cv2.namedWindow("Result")
        cv2.namedWindow("White HSV")
        cv2.namedWindow("Yellow HSV")
        cv2.namedWindow("Mask White")
        cv2.namedWindow("Mask Yellow")
        cv2.namedWindow("Params")
        self._create_trackbars()

        # 右白线 ROI：下半部分
        self.y_start = int(self.height * 0.5)
        self.y_end   = self.height

    def _create_trackbars(self):
        def nothing(x): pass
        # 白色
        cv2.createTrackbar("H Low",  "White HSV", 0,   179, nothing)
        cv2.createTrackbar("H High", "White HSV", 179, 179, nothing)
        cv2.createTrackbar("S Low",  "White HSV", 0,   255, nothing)
        cv2.createTrackbar("S High", "White HSV", 30,  255, nothing)
        cv2.createTrackbar("V Low",  "White HSV", 200, 255, nothing)
        cv2.createTrackbar("V High", "White HSV", 255, 255, nothing)

        # 黄色（OpenCV HSV: H≈20~35 较常见）
        cv2.createTrackbar("H Low",  "Yellow HSV", 20, 179, nothing)
        cv2.createTrackbar("H High", "Yellow HSV", 35, 179, nothing)
        cv2.createTrackbar("S Low",  "Yellow HSV", 60, 255, nothing)
        cv2.createTrackbar("S High", "Yellow HSV", 255,255, nothing)
        cv2.createTrackbar("V Low",  "Yellow HSV", 60, 255, nothing)
        cv2.createTrackbar("V High", "Yellow HSV", 255,255, nothing)

        # 参数
        cv2.createTrackbar("Right Offset(px)", "Params", max(10, self.width//6), self.width//2, nothing)
        cv2.createTrackbar("Yellow MinArea",   "Params", 80,  3000, nothing)  # 黄色连通域最小面积
        cv2.createTrackbar("Yellow BottomY(%)","Params", 60,  100,  nothing)  # 黄色必须出现在画面底部百分比以下
        cv2.createTrackbar("Debounce Frames",  "Params", 3,   10,   nothing)  # 去抖帧数
        cv2.createTrackbar("Polyline Step(px)","Params", 2,   10,   nothing)  # 逐行抽样步长

    def _get_threshold(self, win):
        h_low  = cv2.getTrackbarPos("H Low",  win)
        h_high = cv2.getTrackbarPos("H High", win)
        s_low  = cv2.getTrackbarPos("S Low",  win)
        s_high = cv2.getTrackbarPos("S High", win)
        v_low  = cv2.getTrackbarPos("V Low",  win)
        v_high = cv2.getTrackbarPos("V High", win)
        return np.array([h_low, s_low, v_low]), np.array([h_high, s_high, v_high])

    # 过滤出满足面积&出现位置的黄色区域
    def _filtered_yellow_mask(self, mask_yellow, h):
        min_area   = max(1, cv2.getTrackbarPos("Yellow MinArea", "Params"))
        bottom_pct = cv2.getTrackbarPos("Yellow BottomY(%)", "Params")
        bottom_y_th = int(h * (bottom_pct / 100.0))

        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_yellow, connectivity=8)

        filtered = np.zeros_like(mask_yellow)  # 0/255
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            y    = stats[i, cv2.CC_STAT_TOP]
            hh   = stats[i, cv2.CC_STAT_HEIGHT]
            y_bottom = y + hh - 1
            if area >= min_area and y_bottom >= bottom_y_th:
                filtered[labels == i] = 255
        return filtered

    # 返回 1) 最底行的最右黄色点（用于中点计算） 2) “逐行最右点”的折线点集（自 bottom_y_th->h）
    def _yellow_rightmost_polyline(self, filtered_mask, h):
        pts = cv2.findNonZero(filtered_mask)
        if pts is None:
            return (None, None), []

        ys = pts[:, 0, 1]
        y_max = int(np.max(ys))
        row_pts = pts[ys == y_max][:, 0, :]
        x_max = int(np.max(row_pts[:, 0]))
        bottom_rightmost = (x_max, y_max)

        step = max(1, cv2.getTrackbarPos("Polyline Step(px)", "Params"))
        # 自下阈值开始逐行扫描
        bottom_pct = cv2.getTrackbarPos("Yellow BottomY(%)", "Params")
        bottom_y_th = int(h * (bottom_pct / 100.0))

        poly_pts = []
        # 为了效率，用每行的非零索引
        # 构建每行的右端索引
        for y in range(bottom_y_th, h, step):
            row = filtered_mask[y, :]
            xs = np.where(row > 0)[0]
            if xs.size > 0:
                x_right = int(xs.max())
                poly_pts.append((x_right, y))

        # 如果没有逐行点，至少返回底部点（可选）
        if not poly_pts and bottom_rightmost[0] is not None:
            poly_pts = [bottom_rightmost]

        return bottom_rightmost, poly_pts

    # 右侧（ROI）找白线底部行的均值 x
    def _bottom_x_from_mask(self, side_mask, y_offset, x_offset=0):
        pts = cv2.findNonZero(side_mask)
        if pts is None:
            return None, None
        max_y = int(np.max(pts[:, 0, 1]))
        row_pts = pts[pts[:, 0, 1] == max_y][:, 0, :]
        x_mean = int(np.mean(row_pts[:, 0])) + x_offset
        y_abs  = max_y + y_offset
        return x_mean, y_abs

    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️ 无法读取摄像头帧"); break

                h, w = frame.shape[:2]
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                lower_w, upper_w = self._get_threshold("White HSV")
                lower_y, upper_y = self._get_threshold("Yellow HSV")

                mask_white = cv2.inRange(hsv, lower_w, upper_w)
                mask_yellow  = cv2.inRange(hsv, lower_y, upper_y)

                # 形态学去噪
                kernel = np.ones((3,3), np.uint8)
                mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN,  kernel, iterations=1)
                mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel, iterations=1)
                mask_yellow  = cv2.morphologyEx(mask_yellow,  cv2.MORPH_OPEN,  kernel, iterations=1)
                mask_yellow  = cv2.morphologyEx(mask_yellow,  cv2.MORPH_CLOSE, kernel, iterations=1)

                cv2.imshow("Mask White", mask_white)
                cv2.imshow("Mask Yellow",  mask_yellow)

                # ---------- 右白线（ROI 下半部，右半幅） ----------
                roi_w = mask_white[self.y_start:self.y_end, :]
                right_mask = roi_w[:, w // 2:]
                right_x, right_y = self._bottom_x_from_mask(right_mask, self.y_start, x_offset=w // 2)

                # ---------- 稳健黄色边缘折线（全图） ----------
                filtered_yellow = self._filtered_yellow_mask(mask_yellow, h)
                (yellow_x, yellow_y), yellow_poly = self._yellow_rightmost_polyline(filtered_yellow, h)

                # ---------- 去抖状态机（黄色） ----------
                debounce = max(1, cv2.getTrackbarPos("Debounce Frames", "Params"))
                if yellow_x is not None:
                    self.yellow_present_streak += 1
                    self.yellow_absent_streak = 0
                else:
                    self.yellow_absent_streak += 1
                    self.yellow_present_streak = 0

                if not self.yellow_state and self.yellow_present_streak >= debounce:
                    self.yellow_state = True
                if self.yellow_state and self.yellow_absent_streak >= debounce:
                    self.yellow_state = False

                # ---------- 计算绿色线 ----------
                image_center = w // 2
                green_x = None
                if self.yellow_state and (yellow_x is not None) and (right_x is not None):
                    green_x = (yellow_x + right_x) // 2
                    mode_text = "MODE: YELLOW+RIGHT midpoint (stable)"
                elif (not self.yellow_state) and (right_x is not None):
                    offset = cv2.getTrackbarPos("Right Offset(px)", "Params")
                    green_x = max(0, right_x - offset)
                    mode_text = f"MODE: RIGHT only, offset {offset}px left"
                else:
                    mode_text = "MODE: insufficient edges"

                # 偏差
                if green_x is not None:
                    deviation = green_x - image_center
                    deviation_text = str(deviation)
                else:
                    deviation = math.nan
                    deviation_text = "N/A"

                # ---------- 可视化 ----------
                vis = frame.copy()

                # 右白线
                if right_x is not None and right_y is not None:
                    cv2.circle(vis, (right_x, right_y), 6, (255, 255, 255), -1)
                    cv2.line(vis, (right_x, 0), (right_x, h), (200, 200, 200), 1)

                # 黄色折线（逐行最右点相连）
                if self.yellow_state and len(yellow_poly) >= 2:
                    for i in range(1, len(yellow_poly)):
                        p1 = yellow_poly[i-1]
                        p2 = yellow_poly[i]
                        # 画多条线连接最右黄色像素点
                        cv2.line(vis, p1, p2, (0, 255, 255), 2)  # BGR: Yellow
                # 标注代表点（最底行的最右点）
                if self.yellow_state and yellow_x is not None and yellow_y is not None:
                    cv2.circle(vis, (yellow_x, yellow_y), 6, (0, 255, 255), -1)
                    cv2.line(vis, (yellow_x, 0), (yellow_x, h), (0, 255, 255), 2)

                # 图像中心线
                cv2.line(vis, (image_center, 0), (image_center, h), (255, 0, 0), 2)

                # 绿色线
                if green_x is not None:
                    cv2.line(vis, (green_x, 0), (green_x, h), (0, 255, 0), 2)

                # 文本
                cv2.putText(vis, f"Deviation: {deviation_text}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(vis, mode_text, (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
                if green_x is None:
                    note = "No right white" if right_x is None else "Need YELLOW & RIGHT for midpoint"
                    cv2.putText(vis, note, (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)

                # 去抖状态提示
                dbg = f"YELLOW_STATE={self.yellow_state}  present_streak={self.yellow_present_streak}  absent_streak={self.yellow_absent_streak}"
                cv2.putText(vis, dbg, (10, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1)

                cv2.imshow("Result", vis)
                print(deviation)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            if self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = LaneDetector(cam_index=0, width=240, height=240)
    detector.run()
