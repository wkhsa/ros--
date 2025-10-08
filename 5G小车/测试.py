#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import pigpio
import time
import math
from collections import deque


# ========================== 可调参数 ==========================
SERVO_PIN = 12
MOTOR_PIN = 13
CAM_INDEX = 0
FRAME_W, FRAME_H = 240, 240

# 舵机/电机脉宽（按你原车调）
SERVO_CENTER_PW = 1500                 # 舵机中位
SERVO_MIN_PW, SERVO_MAX_PW = 1300, 1700
START_SPEED_PW = 1250                  # 上电起步
CRUISE_SPEED_PW = 1250                 # 正常巡航速度（避障结束后恢复）
AVOID_SPEED_PW = 1200                  # 避障时减速（数值>1000 取决于你电调前进/倒退定义）

# PID
KP = 3.0
KD = 0.8

# 视觉ROI（只看下半区）
ROI_Y_START_RATIO = 0.5                # 从图像高度的 50% 开始
MORPH_KERNEL = (3, 3)

# 避障逻辑
BLUE_DET_BAND_RATIO = (0.35, 0.85)     # 垂直方向的“前方检测带”占 ROI 的比例 (top, bottom)
BLUE_AREA_PX = 350                     # 蓝色面积阈值（像素）
BLUE_PERSIST_FRAMES = 3                # 达到阈值需连续帧数（抗抖动）
BLUE_CLEAR_FRAMES = 5                  # 消失判定所需连续帧数
AVOID_STEER_OFFSET = 150               # 避障时在 SERVO_CENTER_PW 基础上的额外偏转 (us)
AVOID_TIME = 0.6                       # 避障偏转保持时间 (sec)
# ============================================================


class LaneFollower:
    def __init__(self,
                 servo_pin=SERVO_PIN,
                 motor_pin=MOTOR_PIN,
                 cam_index=CAM_INDEX,
                 width=FRAME_W,
                 height=FRAME_H):

        # ---- pigpio ----
        self.servo_pin = servo_pin
        self.motor_pin = motor_pin
        self.pi = pigpio.pi()
        self.pi.set_mode(self.servo_pin, pigpio.OUTPUT)
        self.pi.set_mode(self.motor_pin, pigpio.OUTPUT)

        # ---- PID ----
        self.base_pw = SERVO_CENTER_PW
        self.kp = KP
        self.kd = KD
        self.prev_error = 0

        # ---- Camera ----
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width, self.height = width, height

        # ---- Windows & Trackbars ----
        cv2.namedWindow("Result")
        cv2.namedWindow("White HSV")
        cv2.namedWindow("Blue HSV")
        cv2.namedWindow("Mask White")
        cv2.namedWindow("Mask Blue")
        self._create_trackbars()

        # ---- ROI ----
        self.y_start = int(self.height * ROI_Y_START_RATIO)
        self.y_end = self.height

        # ---- 单边外推历史 ----
        self.last_lane_width = None
        self.last_lane_center = None

        # ---- 避障状态 ----
        self.blue_hits = 0
        self.blue_clears = 0
        self.avoiding = False
        self.avoid_until = 0.0
        self.avoid_dir = 0  # -1 向左避，+1 向右避

        # 窗口内信息平滑（显示用）
        self.deviation_history = deque(maxlen=10)

    # ----------------- UI & HSV -----------------
    def _create_trackbars(self):
        def nothing(x): pass
        # White
        cv2.createTrackbar("H Low",  "White HSV", 0,   179, nothing)
        cv2.createTrackbar("H High", "White HSV", 179, 179, nothing)
        cv2.createTrackbar("S Low",  "White HSV", 0,   255, nothing)
        cv2.createTrackbar("S High", "White HSV", 30,  255, nothing)
        cv2.createTrackbar("V Low",  "White HSV", 200, 255, nothing)
        cv2.createTrackbar("V High", "White HSV", 255, 255, nothing)
        # Blue
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

    # ----------------- 低层控制 -----------------
    def control_servo(self, error, dt=0.1):
        derivative = (error - self.prev_error) / max(1e-3, dt)
        pw = int(self.base_pw + self.kp * error + self.kd * derivative)
        pw = max(SERVO_MIN_PW, min(SERVO_MAX_PW, pw))
        self.pi.set_servo_pulsewidth(self.servo_pin, pw)
        self.prev_error = error

    def set_servo_direct(self, pw):
        pw = max(SERVO_MIN_PW, min(SERVO_MAX_PW, int(pw)))
        self.pi.set_servo_pulsewidth(self.servo_pin, pw)

    def motor_run(self, speed_pw):
        self.pi.set_servo_pulsewidth(self.motor_pin, int(speed_pw))

    # ----------------- 视觉工具 -----------------
    @staticmethod
    def _morph(mask, ksize=MORPH_KERNEL, it=1):
        kernel = np.ones(ksize, np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=it)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=it)
        return mask

    @staticmethod
    def _bottom_x_from_mask(side_mask, y_offset, x_offset=0):
        # 先找最下面一行，再用该行所有点的 x 均值
        pts = cv2.findNonZero(side_mask)
        if pts is None:
            return None, None
        max_y = int(np.max(pts[:, 0, 1]))
        row_pts = pts[pts[:, 0, 1] == max_y][:, 0, :]
        x_mean = int(np.mean(row_pts[:, 0])) + x_offset
        y_abs = max_y + y_offset
        return x_mean, y_abs

    def _infer_lane_center(self, left_x, right_x, w):
        """允许单边外推：优先用历史宽度，没有则用 1/4 宽的启发式"""
        lane_center = None
        if left_x is not None and right_x is not None:
            lane_center = (left_x + right_x) // 2
            lane_width = right_x - left_x
            if lane_width > 0:
                self.last_lane_width = lane_width
            self.last_lane_center = lane_center
        elif left_x is not None and right_x is None:
            if self.last_lane_width is not None:
                lane_center = left_x + self.last_lane_width // 2
            else:
                lane_center = left_x + int(0.25 * w)
            self.last_lane_center = lane_center
        elif left_x is None and right_x is not None:
            if self.last_lane_width is not None:
                lane_center = right_x - self.last_lane_width // 2
            else:
                lane_center = right_x - int(0.25 * w)
            self.last_lane_center = lane_center

        if lane_center is not None:
            lane_center = int(max(0, min(w - 1, lane_center)))
        return lane_center

    # ----------------- 蓝色障碍检测 -----------------
    def _detect_blue_obstacle(self, blue_mask):
        """
        在 ROI 内的“前方检测带”统计蓝色面积与重心。
        返回 (should_avoid, dir)  ，dir: -1(向左避), +1(向右避), 0(未知)
        """
        h, w = blue_mask.shape[:2]
        top = int(h * BLUE_DET_BAND_RATIO[0])
        bot = int(h * BLUE_DET_BAND_RATIO[1])
        band = blue_mask[top:bot, :]

        # 面积
        area = cv2.countNonZero(band)

        # 连续帧计数（带滞回）
        if area >= BLUE_AREA_PX:
            self.blue_hits += 1
            self.blue_clears = 0
        else:
            self.blue_clears += 1
            if self.blue_clears >= BLUE_CLEAR_FRAMES:
                self.blue_hits = 0  # 彻底清空

        should_avoid = self.blue_hits >= BLUE_PERSIST_FRAMES

        # 方向：基于质心左右
        dir_sign = 0
        if should_avoid:
            M = cv2.moments(band, binaryImage=True)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                dir_sign = -1 if cx >= w // 2 else +1
            else:
                # 没有质心，用左右半区面积决定
                left_area = cv2.countNonZero(band[:, :w // 2])
                right_area = cv2.countNonZero(band[:, w // 2:])
                dir_sign = -1 if right_area >= left_area else +1

        return should_avoid, dir_sign, area, (top, bot)

    # ----------------- 主循环 -----------------
    def run(self):
        try:
            # 启动
            self.motor_run(START_SPEED_PW)
            t_prev = time.time()

            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️ 无法读取摄像头帧")
                    break

                h, w = frame.shape[:2]
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # 阈值
                lower_w, upper_w = self._get_threshold("White HSV")
                lower_b, upper_b = self._get_threshold("Blue HSV")

                mask_white = cv2.inRange(hsv, lower_w, upper_w)
                mask_blue  = cv2.inRange(hsv, lower_b, upper_b)

                mask_white = self._morph(mask_white)
                mask_blue  = self._morph(mask_blue)

                cv2.imshow("Mask White", mask_white)
                cv2.imshow("Mask Blue",  mask_blue)

                # ROI
                roi_w = mask_white[self.y_start:self.y_end, :]
                roi_b = mask_blue [self.y_start:self.y_end, :]

                # 左右白线（右侧用白色，左侧也用白色；如你的左边是蓝色可以替换策略）
                left_mask  = roi_w[:, :w // 2]
                right_mask = roi_w[:, w // 2:]

                left_x,  left_y  = self._bottom_x_from_mask(left_mask,  self.y_start, x_offset=0)
                right_x, right_y = self._bottom_x_from_mask(right_mask, self.y_start, x_offset=w // 2)

                # 车道中心（允许单边外推）
                lane_center = self._infer_lane_center(left_x, right_x, w)
                image_center = w // 2

                # 避障检测（对 ROI 的“前向检测带”统计蓝色）
                should_avoid, dir_sign, blue_area, (btop, bbot) = self._detect_blue_obstacle(roi_b)

                # 可视化
                vis = frame.copy()
                if left_x is not None and left_y is not None:
                    cv2.circle(vis, (left_x, left_y), 5, (0, 255, 255), -1)
                if right_x is not None and right_y is not None:
                    cv2.circle(vis, (right_x, right_y), 5, (255, 255, 255), -1)
                # 画ROI与蓝色检测带
                cv2.rectangle(vis, (0, self.y_start), (w-1, self.y_end-1), (100, 100, 100), 1)
                cv2.rectangle(vis, (0, self.y_start + btop), (w-1, self.y_start + bbot), (255, 0, 0), 1)

                # 主控制：避障优先
                now = time.time()
                if self.avoiding:
                    # 在避障窗口内强制打角并减速
                    self.motor_run(AVOID_SPEED_PW)
                    steer_pw = self.base_pw + (AVOID_STEER_OFFSET * self.avoid_dir)
                    self.set_servo_direct(steer_pw)

                    if now >= self.avoid_until:
                        # 结束避障，恢复巡航+PID接管
                        self.avoiding = False
                        self.blue_hits = 0
                        self.blue_clears = 0
                        self.prev_error = 0
                        self.motor_run(CRUISE_SPEED_PW)

                else:
                    if should_avoid and dir_sign != 0:
                        # 进入避障：朝与蓝色相反方向打角
                        self.avoiding = True
                        self.avoid_dir = dir_sign   # -1 向左避, +1 向右避
                        self.avoid_until = now + AVOID_TIME
                        # 立即生效一帧（更果断）
                        self.motor_run(AVOID_SPEED_PW)
                        steer_pw = self.base_pw + (AVOID_STEER_OFFSET * self.avoid_dir)
                        self.set_servo_direct(steer_pw)
                    else:
                        # 正常循迹：PID 让车自动回正
                        if lane_center is not None:
                            error = lane_center - image_center
                            t_now = time.time()
                            dt = t_now - t_prev
                            t_prev = t_now
                            self.control_servo(error, dt=max(1e-3, dt))
                            self.motor_run(CRUISE_SPEED_PW)
                            self.deviation_history.append(error)
                        else:
                            # 看不见线：保守做法 —— 舵机回中，低速直行
                            self.set_servo_direct(self.base_pw)
                            self.motor_run(AVOID_SPEED_PW)

                # 画中心线
                cv2.line(vis, (image_center, 0), (image_center, h), (255, 0, 0), 2)
                if lane_center is not None:
                    cv2.line(vis, (lane_center, 0), (lane_center, h), (0, 255, 0), 2)

                # 叠字
                dev_disp = (sum(self.deviation_history)/len(self.deviation_history)) if self.deviation_history else 0
                state_text = "AVOIDING" if self.avoiding else "FOLLOWING"
                cv2.putText(vis, f"State: {state_text}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
                cv2.putText(vis, f"Blue area: {blue_area}", (10, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                if lane_center is not None:
                    cv2.putText(vis, f"Deviation: {int(dev_disp)}", (10, 71),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    cv2.putText(vis, "Deviation: N/A", (10, 71),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.imshow("Result", vis)

                # 退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nCtrl+C detected! Stopping...")
        finally:
            self.cleanup()

    # ----------------- 收尾 -----------------
    def cleanup(self):
        # 停车&释放
        self.motor_run(0)
        self.pi.set_servo_pulsewidth(self.servo_pin, 0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.pi.stop()


# ----------------- 主程序 -----------------
if __name__ == "__main__":
    follower = LaneFollower(
        servo_pin=SERVO_PIN,
        motor_pin=MOTOR_PIN,
        cam_index=CAM_INDEX,
        width=FRAME_W,
        height=FRAME_H
    )
    follower.run()
