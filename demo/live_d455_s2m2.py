
import os
import time
import cv2
import numpy as np
import torch
import rclpy
import message_filters

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from s2m2.core.utils.model_utils import (
    load_model,
    run_stereo_matching,
)

REPO = "/data/slam_algorithms/s2m2"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LEFT_TOPIC = "/camera/camera/infra1/image_rect_raw"
RIGHT_TOPIC = "/camera/camera/infra2/image_rect_raw"
INFO_TOPIC = "/camera/camera/infra1/camera_info"
D455_DEPTH_TOPIC = "/camera/camera/depth/image_rect_raw"
S2M2_DEPTH_TOPIC = "/s2m2/depth"

# Approximate D455 stereo baseline. Verify using actual calibration.
BASELINE_M = 0.095


class LiveS2M2(Node):
    def __init__(self):
        super().__init__("s2m2_live_d455")

        self.bridge = CvBridge()
        self.busy = False
        self.last_run = 0.0
        self.count = 0

        self.fx = None
        self.s2m2_depth = None
        self.d455_depth = None
        self.cursor_x = None
        self.cursor_y = None

        self.get_logger().info(f"Loading S²M²-S on {DEVICE}...")

        self.model = load_model(
            os.path.join(REPO, "weights/pretrain_weights"),
            "S",
            True,
            3,
            DEVICE,
        )

        if self.model is None:
            raise RuntimeError("Could not load S²M² model")

        self.depth_pub = self.create_publisher(
            Image, S2M2_DEPTH_TOPIC, 10
        )

        self.info_sub = self.create_subscription(
            CameraInfo,
            INFO_TOPIC,
            self.camera_info_callback,
            qos_profile_sensor_data,
        )

        self.depth_sub = self.create_subscription(
            Image,
            D455_DEPTH_TOPIC,
            self.depth_callback,
            qos_profile_sensor_data,
        )

        self.left_sub = message_filters.Subscriber(
            self,
            Image,
            LEFT_TOPIC,
            qos_profile=qos_profile_sensor_data,
        )

        self.right_sub = message_filters.Subscriber(
            self,
            Image,
            RIGHT_TOPIC,
            qos_profile=qos_profile_sensor_data,
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub],
            queue_size=5,
            slop=0.03,
        )
        self.sync.registerCallback(self.callback)

        self.window = "D455 live S2M2"
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self.mouse_callback)

        self.get_logger().info(
            "S²M² ready. Hover over the disparity image; press q to quit."
        )

    def camera_info_callback(self, msg):
        if msg.k[0] > 0:
            self.fx = float(msg.k[0])

    def depth_callback(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough"
            )
            depth = depth.astype(np.float32)

            # Standard RealSense 16UC1 depth values are millimetres.
            if msg.encoding == "16UC1":
                depth /= 1000.0

            self.d455_depth = depth

        except Exception as exc:
            self.get_logger().warning(f"D455 depth error: {exc}")

    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_MOUSEMOVE:
            return

        if self.s2m2_depth is None:
            return

        h, w = self.s2m2_depth.shape

        # Disparity is displayed on the right half of the window.
        if w <= x < 2 * w and 0 <= y < h:
            self.cursor_x = x - w
            self.cursor_y = y
        else:
            self.cursor_x = None
            self.cursor_y = None

    def callback(self, left_msg, right_msg):
        if self.busy:
            return

        now = time.monotonic()
        if now - self.last_run < 0.30:
            return

        self.busy = True
        self.last_run = now

        try:
            left_gray = self.bridge.imgmsg_to_cv2(
                left_msg, desired_encoding="mono8"
            )
            right_gray = self.bridge.imgmsg_to_cv2(
                right_msg, desired_encoding="mono8"
            )

            if left_gray.shape != right_gray.shape:
                self.get_logger().warning(
                    "Left/right image sizes differ"
                )
                return

            left_rgb = cv2.cvtColor(
                left_gray, cv2.COLOR_GRAY2RGB
            )
            right_rgb = cv2.cvtColor(
                right_gray, cv2.COLOR_GRAY2RGB
            )

            left_t = (
                torch.from_numpy(left_rgb)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(DEVICE)
            )
            right_t = (
                torch.from_numpy(right_rgb)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(DEVICE)
            )

            start = time.perf_counter()

            disp, occ, conf, avg_conf, _ = run_stereo_matching(
                self.model, left_t, right_t, DEVICE
            )

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start

            disp = disp.detach().cpu().numpy().squeeze()
            occ = occ.detach().cpu().numpy().squeeze()
            conf = conf.detach().cpu().numpy().squeeze()

            if disp.ndim != 2:
                self.get_logger().warning(
                    f"Unexpected disparity shape: {disp.shape}"
                )
                return

            finite = np.isfinite(disp)
            if not finite.any():
                self.get_logger().warning(
                    "Model returned no finite disparity"
                )
                return

            # Provisional metric depth. This assumes disparity is in pixels.
            # Validate baseline, disparity scale and camera calibration first.
            depth_m = np.full(disp.shape, np.nan, dtype=np.float32)

            if self.fx is not None:
                valid = finite & (disp > 0)
                depth_m[valid] = (
                    self.fx * BASELINE_M / disp[valid]
                )
                depth_m[
                    (~np.isfinite(depth_m))
                    | (depth_m < 0.15)
                    | (depth_m > 20.0)
                ] = np.nan

                self.s2m2_depth = depth_m

                depth_msg = self.bridge.cv2_to_imgmsg(
                    depth_m, encoding="32FC1"
                )
                depth_msg.header = left_msg.header
                self.depth_pub.publish(depth_msg)

            # Relative-colour disparity view.
            values = disp[finite]
            lo, hi = np.percentile(values, [2, 98])

            if hi <= lo:
                hi = lo + 1.0

            disp_vis = np.clip(
                (disp - lo) / (hi - lo), 0, 1
            )
            disp_vis[~finite] = 0
            disp_vis = (disp_vis * 255).astype(np.uint8)
            disp_color = cv2.applyColorMap(
                disp_vis, cv2.COLORMAP_TURBO
            )

            left_bgr = cv2.cvtColor(
                left_gray, cv2.COLOR_GRAY2BGR
            )
            combined = np.hstack((left_bgr, disp_color))

            cv2.putText(
                combined, "Left IR", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2
            )
            cv2.putText(
                combined, "S2M2 disparity", (left_gray.shape[1] + 10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2
            )

            # Display distances at the selected pixel.
            if (
                self.s2m2_depth is not None
                and self.cursor_x is not None
                and self.cursor_y is not None
            ):
                x = self.cursor_x
                y = self.cursor_y
                h, w = self.s2m2_depth.shape

                if 0 <= x < w and 0 <= y < h:
                    z = float(self.s2m2_depth[y, x])

                    if np.isfinite(z):
                        s2m2_text = f"S2M2 estimate: {z:.2f} m"
                    else:
                        s2m2_text = "S2M2 estimate: invalid"

                    cv2.putText(
                        combined, s2m2_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2
                    )

                    # Approximate same-pixel comparison only.
                    # The depth image is not guaranteed to align to IR1.
                    if self.d455_depth is not None:
                        dh, dw = self.d455_depth.shape
                        if y < dh and x < dw:
                            dz = float(self.d455_depth[y, x])

                            if np.isfinite(dz) and 0.15 <= dz <= 20.0:
                                d455_text = f"D455 reference: {dz:.2f} m"
                            else:
                                d455_text = "D455 reference: invalid"

                            cv2.putText(
                                combined, d455_text, (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0), 2
                            )

            cv2.imshow(self.window, combined)
            cv2.waitKey(1)

            self.count += 1
            self.get_logger().info(
                f"pair={self.count} | "
                f"image={left_gray.shape[1]}x{left_gray.shape[0]} | "
                f"inference={elapsed * 1000:.1f} ms | "
                f"FPS={1.0 / elapsed:.2f} | "
                f"model_conf={avg_conf:.3f} | "
                f"median_disparity={np.median(values):.2f} px"
            )

        except Exception as exc:
            self.get_logger().error(f"Inference failed: {exc}")

        finally:
            self.busy = False

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main():
    rclpy.init()
    node = None

    try:
        node = LiveS2M2()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
'''




import os
import time
import cv2
import numpy as np
import torch
import rclpy
import message_filters

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from s2m2.core.utils.model_utils import (
    load_model,
    run_stereo_matching,
)

REPO = "/data/slam_algorithms/s2m2"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LEFT_TOPIC = "/camera/camera/infra1/image_rect_raw"
RIGHT_TOPIC = "/camera/camera/infra2/image_rect_raw"


class LiveS2M2(Node):
    def __init__(self):
        super().__init__("s2m2_live_d455")

        self.bridge = CvBridge()
        self.busy = False
        self.last_run = 0.0
        self.count = 0

        self.current_disp = None
        self.cursor_x = None
        self.cursor_y = None

        self.get_logger().info(f"Loading S²M²-S on {DEVICE}...")

        self.model = load_model(
            os.path.join(REPO, "weights/pretrain_weights"),
            "S",
            True,
            3,
            DEVICE,
        )

        if self.model is None:
            raise RuntimeError("Could not load S²M² model")

        self.left_sub = message_filters.Subscriber(
            self,
            Image,
            LEFT_TOPIC,
            qos_profile=qos_profile_sensor_data,
        )

        self.right_sub = message_filters.Subscriber(
            self,
            Image,
            RIGHT_TOPIC,
            qos_profile=qos_profile_sensor_data,
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub],
            queue_size=5,
            slop=0.03,
        )
        self.sync.registerCallback(self.callback)

        self.window = "S2M2 - Live D455 Infrared"
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self.mouse_callback)

        self.get_logger().info(
            "S²M² ready. Move the mouse over the right-hand disparity image. "
            "Press q to quit."
        )

    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_MOUSEMOVE:
            return

        if self.current_disp is None:
            return

        h, w = self.current_disp.shape

        # The disparity image occupies the right half of the combined window.
        if w <= x < 2 * w and 0 <= y < h:
            self.cursor_x = x - w
            self.cursor_y = y
        else:
            self.cursor_x = None
            self.cursor_y = None

    def callback(self, left_msg, right_msg):
        if self.busy:
            return

        now = time.monotonic()

        # Limit inference rate to about 3 stereo pairs per second.
        if now - self.last_run < 0.30:
            return

        self.busy = True
        self.last_run = now

        try:
            left_gray = self.bridge.imgmsg_to_cv2(
                left_msg, desired_encoding="mono8"
            )
            right_gray = self.bridge.imgmsg_to_cv2(
                right_msg, desired_encoding="mono8"
            )

            if left_gray.shape != right_gray.shape:
                self.get_logger().warning(
                    "Left/right image sizes differ"
                )
                return

            # Convert mono infrared images to three-channel model inputs.
            left_rgb = cv2.cvtColor(
                left_gray, cv2.COLOR_GRAY2RGB
            )
            right_rgb = cv2.cvtColor(
                right_gray, cv2.COLOR_GRAY2RGB
            )

            left_t = (
                torch.from_numpy(left_rgb)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(DEVICE)
            )

            right_t = (
                torch.from_numpy(right_rgb)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(DEVICE)
            )

            start = time.perf_counter()

            disp, occ, conf, avg_conf, _ = run_stereo_matching(
                self.model,
                left_t,
                right_t,
                DEVICE,
            )

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start

            disp = disp.detach().cpu().numpy().squeeze()
            occ = occ.detach().cpu().numpy().squeeze()
            conf = conf.detach().cpu().numpy().squeeze()

            if disp.ndim != 2:
                self.get_logger().warning(
                    f"Unexpected disparity shape: {disp.shape}"
                )
                return

            finite = np.isfinite(disp)

            if not finite.any():
                self.get_logger().warning(
                    "Model returned no finite disparity"
                )
                return

            # Save the disparity map for mouse-based inspection.
            self.current_disp = disp.copy()

            # Normalize colours for visualization only.
            values = disp[finite]
            lo, hi = np.percentile(values, [2, 98])

            if hi <= lo:
                hi = lo + 1.0

            disp_vis = np.zeros(disp.shape, dtype=np.uint8)
            disp_vis[finite] = (
                np.clip(
                    (disp[finite] - lo) / (hi - lo),
                    0.0,
                    1.0,
                ) * 255
            ).astype(np.uint8)

            disp_color = cv2.applyColorMap(
                disp_vis, cv2.COLORMAP_TURBO
            )

            left_bgr = cv2.cvtColor(
                left_gray, cv2.COLOR_GRAY2BGR
            )

            combined = np.hstack((left_bgr, disp_color))

            cv2.putText(
                combined,
                "Left Infrared",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                combined,
                "S2M2 Disparity",
                (left_gray.shape[1] + 10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            # Show the model's raw disparity value under the cursor.
            if (
                self.cursor_x is not None
                and self.cursor_y is not None
            ):
                x = self.cursor_x
                y = self.cursor_y

                if (
                    0 <= y < disp.shape[0]
                    and 0 <= x < disp.shape[1]
                ):
                    value = float(disp[y, x])

                    if np.isfinite(value):
                        text = (
                            f"Disparity: {value:.2f} px"
                        )
                    else:
                        text = "Disparity: invalid"

                    cv2.drawMarker(
                        combined,
                        (x + left_gray.shape[1], y),
                        (255, 255, 255),
                        cv2.MARKER_CROSS,
                        16,
                        2,
                    )

                    cv2.putText(
                        combined,
                        text,
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 255),
                        2,
                    )

            cv2.imshow(self.window, combined)
            cv2.waitKey(1)

            self.count += 1

            self.get_logger().info(
                f"pair={self.count} | "
                f"image={left_gray.shape[1]}x{left_gray.shape[0]} | "
                f"inference={elapsed * 1000:.1f} ms | "
                f"FPS={1.0 / elapsed:.2f} | "
                f"model_conf={avg_conf:.3f} | "
                f"median_disparity={np.median(values):.2f}"
            )

        except Exception as exc:
            self.get_logger().error(
                f"Inference failed: {exc}"
            )

        finally:
            self.busy = False

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main():
    rclpy.init()
    node = None

    try:
        node = LiveS2M2()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
'''
