def run(self):
    try:
        # —— 校准/空挡：先给中位脉宽几秒（按你 ESC 需要来）——
        self.pi.set_servo_pulsewidth(self.motor_pin, 1500)
        time.sleep(2.0)

        # 启动速度（略慢，便于调参）
        drive_pw = 1470  # 1500为空挡，<1500倒车/刹，>1500前进（依 ESC 而定）
        self.motor_run(drive_pw)

        last_time = time.monotonic()
        last_lane_center = self.width // 2
        lost_frames = 0

        # ROI 参数（取底部 40%）
        y_start = int(self.height * 0.6)
        y_end = self.height - 1

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Camera read failed.")
                break

            # 动态 dt
            now = time.monotonic()
            dt = max(1e-3, now - last_time)
            last_time = now

            # 预处理
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower, upper = self.get_hsv_threshold()

            mask = cv2.inRange(hsv, lower, upper)
            # 形态学去噪
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            roi_mask = mask[y_start:min(y_end, self.height-1), :]
            h, w = frame.shape[:2]
            left_mask = roi_mask[:, :w // 2]
            right_mask = roi_mask[:, w // 2:]

            def bottom_mean_x(binary):
                pts = cv2.findNonZero(binary)
                if pts is None:
                    return None, None
                ys = pts[:, 0, 1]
                yb = ys.max()
                xs_bottom = pts[ys == yb][:, 0, 0]
                xb = int(xs_bottom.mean())
                return xb, int(yb)

            lx, ly = bottom_mean_x(left_mask)
            rx, ry = bottom_mean_x(right_mask)
            if rx is not None:
                rx += w // 2  # 右半区需要偏移

            have_left = lx is not None
            have_right = rx is not None

            # 计算 lane_center（含丢线策略）
            if have_left and have_right:
                lane_center = (lx + rx) // 2
                lost_frames = 0
            elif have_left and not have_right:
                # 估计右线，或者将车道中心向左线右侧偏移固定量
                lane_center = int(0.5 * (lx + (w * 0.75)))
                lost_frames += 1
            elif have_right and not have_left:
                lane_center = int(0.5 * (rx + (w * 0.25)))
                lost_frames += 1
            else:
                # 双丢：使用上一次的中心，但进入安全模式
                lane_center = last_lane_center
                lost_frames += 1

            image_center = w // 2
            error = lane_center - image_center

            # 误差低通（抑制微分噪声）
            alpha = 0.7
            filt_error = int(alpha * error + (1 - alpha) * (self.prev_error))

            # —— PD 控制（微分限幅）——
            deriv = (filt_error - self.prev_error) / dt
            deriv = max(-2000, min(2000, deriv))  # 限幅，避免尖峰
            pw = int(self.base_pw + self.kp * filt_error + self.kd * deriv)
            pw = max(1300, min(1700, pw))
            self.pi.set_servo_pulsewidth(self.servo_pin, pw)
            self.prev_error = filt_error
            last_lane_center = lane_center

            # 丢线安全：连续丢线时减速/空挡
            if lost_frames >= 5:
                self.motor_run(1500)  # 空挡/刹停（按 ESC 定义）
            else:
                self.motor_run(drive_pw)

            # 叠加可视化（可选）
            cv2.line(frame, (image_center, 0), (image_center, h), (255, 0, 0), 2)
            cv2.line(frame, (lane_center, 0), (lane_center, h), (0, 255, 0), 2)
            cv2.putText(frame, f"err:{error} dt:{dt:.3f} lost:{lost_frames}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Mask", mask)
            cv2.imshow("Result", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nCtrl+C detected! Stopping...")
    finally:
        self.cleanup()
