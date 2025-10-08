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

        cv2.namedWindow("Result")
        cv2.namedWindow("White HSV")   # 白色范围（右侧 & 左侧回退）
        cv2.namedWindow("Blue HSV")    # 蓝色范围（左侧优先）
        cv2.namedWindow("Mask White")
        cv2.namedWindow("Mask Blue")
        self._create_trackbars()

        # 只在图片下半部分做检测
        self.y_start = int(self.height * 0.5)
        self.y_end   = self.height

        # ===== 新增：用于单边外推的历史信息 =====
        self.last_lane_width = None    # 上一帧左右边界的像素宽度
        self.last_lane_center = None   # 上一帧车道中心

    def _create_trackbars(self):
        def nothing(x): pass
        # 白色（亮、饱和度低）
        cv2.createTrackbar("H Low",  "White HSV", 0,   179, nothing)
        cv2.createTrackbar("H High", "White HSV", 179, 179, nothing)
        cv2.createTrackbar("S Low",  "White HSV", 0,   255, nothing)
        cv2.createTrackbar("S High", "White HSV", 30,  255, nothing)
        cv2.createTrackbar("V Low",  "White HSV", 200, 255, nothing)
        cv2.createTrackbar("V High", "White HSV", 255, 255, nothing)
        # 蓝色（典型 100~130）
        cv2.createTrackbar("H Low",  "Blue HSV", 100, 179, nothing)
        cv2.createTrackbar("H High", "Blue HSV", 130, 179, nothing)
        cv2.createTrackbar("S Low",  "Blue HSV", 60,  255, nothing)
        cv2.createTrackbar("S High", "Blue HSV", 255, 255, nothing)
        cv2.createTrackbar("V Low",  "Blue HSV", 60,  255, nothing)
        cv2.createTrackbar("V High", "Blue HSV", 255, 255, nothing)

    def _get_threshold(self, win):
        h_low  = cv2.getTrackbarPos("H Low",  win)
        h_high = cv2.getTrackbarPos("H High", win)
        s_low  = cv2.getTrackbarPos("S Low",  win)
        s_high = cv2.getTrackbarPos("S High", win)
        v_low  = cv2.getTrackbarPos("V Low",  win)
        v_high = cv2.getTrackbarPos("V High", win)
        return np.array([h_low, s_low, v_low]), np.array([h_high, s_high, v_high])

    def _bottom_x_from_mask(self, side_mask, y_offset, x_offset=0):
        """原有策略：先取最下面那一行，再取该行所有点的 x 均值。"""
        pts = cv2.findNonZero(side_mask)
        if pts is None:
            return None, None
        max_y = int(np.max(pts[:, 0, 1]))
        row_pts = pts[pts[:, 0, 1] == max_y][:, 0, :]  # (x,y)
        x_mean = int(np.mean(row_pts[:, 0])) + x_offset
        y_abs  = max_y + y_offset
        return x_mean, y_abs

    def _right_then_bottom_from_mask(self, side_mask, y_offset, x_offset=0):
        """蓝色点专用策略：先选最右的列，再在该列中选最下面的点。"""
        pts = cv2.findNonZero(side_mask)
        if pts is None:
            return None, None
        xy = pts[:, 0, :]                   # (N, 2) -> (x, y)
        max_x = int(np.max(xy[:, 0]))       # 先取最右
        candidates = xy[xy[:, 0] == max_x]  # 所有最右候选
        max_y = int(np.max(candidates[:, 1]))  # 在最右里再取最下面
        x_abs = max_x + x_offset
        y_abs = max_y + y_offset
        return x_abs, y_abs

    def _clip_int(self, v, lo, hi):
        return int(max(lo, min(hi, v)))

    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️ 无法读取摄像头帧"); break

                h, w = frame.shape[:2]
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                lower_w, upper_w = self._get_threshold("White HSV")
                lower_b, upper_b = self._get_threshold("Blue HSV")

                mask_white = cv2.inRange(hsv, lower_w, upper_w)
                mask_blue  = cv2.inRange(hsv, lower_b, upper_b)

                # 形态学去噪
                kernel = np.ones((3,3), np.uint8)
                mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN,  kernel, iterations=1)
                mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel, iterations=1)
                mask_blue  = cv2.morphologyEx(mask_blue,  cv2.MORPH_OPEN,  kernel, iterations=1)
                mask_blue  = cv2.morphologyEx(mask_blue,  cv2.MORPH_CLOSE, kernel, iterations=1)

                cv2.imshow("Mask White", mask_white)
                cv2.imshow("Mask Blue",  mask_blue)

                # ROI：下半部分
                roi_w = mask_white[self.y_start:self.y_end, :]
                roi_b = mask_blue [self.y_start:self.y_end, :]

                # 左侧：优先蓝色（先最右，再最下面）-> 回退白色（原策略）
                left_x = left_y = None
                left_color = (255, 0, 0)  # 蓝点的可视化颜色：BGR
                left_blue_mask = roi_b[:, :w // 2]
                left_x, left_y = self._right_then_bottom_from_mask(
                    left_blue_mask, self.y_start, x_offset=0
                )

                if left_x is None:
                    left_white_mask = roi_w[:, :w // 2]
                    left_x, left_y = self._bottom_x_from_mask(
                        left_white_mask, self.y_start, x_offset=0
                    )
                    left_color = (0, 255, 255)  # 回退白色：黄色点

                # 右侧：白色（原策略）
                right_mask = roi_w[:, w // 2:]
                right_x, right_y = self._bottom_x_from_mask(
                    right_mask, self.y_start, x_offset=w // 2
                )

                # ========= 改动核心：允许在单边时外推 lane_center =========
                image_center = w // 2
                lane_center = None
                inference_note = ""  # 用于 UI 提示是否是外推

                if left_x is not None and right_x is not None:
                    # 双边都有：直接求中心，并更新历史宽度
                    lane_center = (left_x + right_x) // 2
                    lane_width  = right_x - left_x
                    if lane_width > 0:
                        self.last_lane_width = lane_width
                    self.last_lane_center = lane_center

                elif left_x is not None and right_x is None:
                    # 只有左边：用历史宽度外推；没有历史则用固定比例
                    if self.last_lane_width is not None:
                        lane_center = left_x + self.last_lane_width // 2
                        inference_note = " (inferred from left)"
                    else:
                        lane_center = left_x + int(0.25 * w)  # 初始估计：1/4 宽
                        inference_note = " (bootstrapped left)"
                    self.last_lane_center = lane_center

                elif left_x is None and right_x is not None:
                    # 只有右边：用历史宽度外推；没有历史则用固定比例
                    if self.last_lane_width is not None:
                        lane_center = right_x - self.last_lane_width // 2
                        inference_note = " (inferred from right)"
                    else:
                        lane_center = right_x - int(0.25 * w)  # 初始估计：1/4 宽
                        inference_note = " (bootstrapped right)"
                    self.last_lane_center = lane_center

                # 合理裁剪，避免越界
                if lane_center is not None:
                    lane_center = self._clip_int(lane_center, 0, w - 1)

                # 偏差显示
                if lane_center is not None:
                    deviation = lane_center - image_center
                    deviation_text = f"{deviation}"
                else:
                    deviation = math.nan
                    deviation_text = "N/A"
                # =========================================================

                # 可视化
                vis = frame.copy()
                if left_x is not None and left_y is not None:
                    cv2.circle(vis, (left_x, left_y), 6, left_color, -1)
                if right_x is not None and right_y is not None:
                    cv2.circle(vis, (right_x, right_y), 6, (255, 255, 255), -1)

                # 画图像中心线和车道中心线
                cv2.line(vis, (image_center, 0), (image_center, h), (255, 0, 0), 2)
                if lane_center is not None:
                    cv2.line(vis, (lane_center, 0), (lane_center, h), (0, 255, 0), 2)

                # 显示文本
                cv2.putText(vis, f"Deviation: {deviation_text}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                mode = "LEFT: BLUE (right-then-bottom)" if left_color == (255, 0, 0) else "LEFT: WHITE (fallback)"
                cv2.putText(vis, mode, (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                if lane_center is None:
                    if left_x is None and right_x is None:
                        note = "No edges on both sides"
                    elif left_x is None:
                        note = "Left edge missing"
                    else:
                        note = "Right edge missing"
                    cv2.putText(vis, note, (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                else:
                    if inference_note:
                        cv2.putText(vis, f"Center{inference_note}", (10, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                cv2.imshow("Result", vis)

                # 终端输出：左右都在时输出数值；单边时输出外推偏差；都缺失时 NaN
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
