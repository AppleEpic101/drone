import airsim
import time
import numpy as np


class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0.0
        self.integral = 0.0

    def calculate(self, setpoint, current, dt):
        error = setpoint - current
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        self.previous_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return output


class Drone:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.pid_x = PIDController(kp=5.0, ki=0.2, kd=0.3)
        self.pid_y = PIDController(kp=5.0, ki=0.2, kd=0.3)
        self.pid_z = PIDController(kp=20.0, ki=0.5, kd=0.7)
        self.pid_roll = PIDController(kp=10.0, ki=0.5, kd=0.5)
        self.pid_pitch = PIDController(kp=10.0, ki=0.5, kd=0.5)
        self.pid_yaw = PIDController(kp=10.0, ki=0.5, kd=0.5)

    def disarm(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)

    def move_to(self, target_position, target_orientation, dt=1, tolerance=0.05):
        target_x, target_y, target_z = target_position
        target_roll, target_pitch, target_yaw = target_orientation

        while True:
            state = self.client.getMultirotorState()
            barometer_data = self.client.getBarometerData()
            imu_data = self.client.getImuData()

            current_x = state.kinematics_estimated.position.x_val
            current_y = state.kinematics_estimated.position.y_val
            current_z = state.kinematics_estimated.position.z_val

            orientation = imu_data.orientation

            current_roll, current_pitch, current_yaw = self.convert_quaternion(
                orientation.w_val,
                orientation.x_val,
                orientation.y_val,
                orientation.z_val,
            )

            angular_velocity = imu_data.angular_velocity
            linear_acceleration = imu_data.linear_acceleration

            print("========================================================")
            print(
                f"Position - x: {current_x:.2f}, y: {current_y:.2f}, z: {current_z:.2f}"
            )
            print(
                f"Barometer - Altitude: {barometer_data.altitude:.2f}, Pressure: {barometer_data.pressure:.2f}"
            )
            print(
                f"IMU - Orientation: ({orientation.w_val:.2f}, {orientation.x_val:.2f}, {orientation.y_val:.2f}, {orientation.z_val:.2f})"
            )

            print(
                f"IMU - Orientation (Roll, Pitch, Yaw): ({current_roll:.2f}, {current_pitch:.2f}, {current_yaw:.2f})"
            )
            print(
                f"IMU - Angular Velocity Vector: x: {angular_velocity.x_val:.2f}, y: {angular_velocity.y_val:.2f}, z: {angular_velocity.z_val:.2f}"
            )
            print(
                f"IMU - Linear Acceleration Vector: x: {linear_acceleration.x_val:.2f}, y: {linear_acceleration.y_val:.2f}, z: {linear_acceleration.z_val:.2f}"
            )

            velocity_x = self.pid_x.calculate(target_x, current_x, dt)
            velocity_x = np.clip(velocity_x, -5.0, 5.0)  # limit velocity

            velocity_y = self.pid_y.calculate(target_y, current_y, dt)
            velocity_y = np.clip(velocity_y, -5.0, 5.0)

            velocity_z = self.pid_z.calculate(target_z, current_z, dt)
            velocity_z = np.clip(velocity_z, -2.0, 2.0)

            angular_velocity_roll = self.pid_roll.calculate(
                target_roll, current_roll, dt
            )
            angular_velocity_roll = np.clip(angular_velocity_roll, -5.0, 5.0)

            angular_velocity_pitch = self.pid_pitch.calculate(
                target_pitch, current_pitch, dt
            )
            angular_velocity_pitch = np.clip(angular_velocity_pitch, -5.0, 5.0)

            angular_velocity_yaw = self.pid_yaw.calculate(target_yaw, current_yaw, dt)
            angular_velocity_yaw = np.clip(angular_velocity_yaw, -5.0, 5.0)

            self.client.moveByVelocityAsync(
                velocity_x,
                velocity_y,
                velocity_z,
                dt,
                airsim.DrivetrainType.MaxDegreeOfFreedom,
                airsim.YawMode(False, angular_velocity_yaw),
            ).join()

            if (
                abs(current_x - target_x) < tolerance
                and abs(current_y - target_y) < tolerance
                and abs(current_z - target_z) < tolerance
                and abs(current_roll - target_roll) < tolerance
                and abs(current_pitch - target_pitch) < tolerance
                and abs(current_yaw - target_yaw) < tolerance
            ):
                break

            time.sleep(dt)

    def convert_quaternion(self, w, x, y, z):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)

        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


if __name__ == "__main__":
    drone = Drone()

    try:
        drone.move_to((0.0, 0.0, -10.0), (0.0, 0.0, 0.0))

        drone.move_to((20.0, 0.0, -10.0), (0.0, 0.0, 45.0))
        drone.move_to((20.0, 20.0, -10.0), (0.0, 0.0, 90.0))
        drone.move_to((0.0, 20.0, -10.0), (0.0, 0.0, 135.0))

        drone.move_to((0.0, 0.0, 0.0))

    except KeyboardInterrupt:
        print("Flight interrupted by user.")

    finally:
        drone.disarm()
