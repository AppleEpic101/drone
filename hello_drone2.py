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
        self.pid_altitude = PIDController(kp=20.0, ki=0.5, kd=0.7)

    def disarm(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)

    def move_to(self, target_position, dt=0.01, tolerance=0.05):
        target_x, target_y, target_z = target_position

        while True:
            state = self.client.getMultirotorState()
            current_x = state.kinematics_estimated.position.x_val
            current_y = state.kinematics_estimated.position.y_val
            current_z = state.kinematics_estimated.position.z_val

            velocity_x = self.pid_x.calculate(target_x, current_x, dt)
            velocity_x = np.clip(velocity_x, -5.0, 5.0)  # limit velocity

            velocity_y = self.pid_y.calculate(target_y, current_y, dt)
            velocity_y = np.clip(velocity_y, -5.0, 5.0)

            velocity_z = self.pid_altitude.calculate(target_z, current_z, dt)
            velocity_z = np.clip(velocity_z, -2.0, 2.0)

            self.client.moveByVelocityAsync(
                velocity_x, velocity_y, velocity_z, dt
            ).join()

            if (
                abs(current_x - target_x) < tolerance
                and abs(current_y - target_y) < tolerance
                and abs(current_z - target_z) < tolerance
            ):
                break

            time.sleep(dt)


if __name__ == "__main__":
    drone = Drone()

    try:
        drone.move_to((0.0, 0.0, -10.0), dt=0.01)

        drone.move_to((20.0, 0.0, -10.0), dt=0.01)
        drone.move_to((20.0, 20.0, -10.0), dt=0.01)
        drone.move_to((0.0, 20.0, -10.0), dt=0.01)
        drone.move_to((0.0, 0.0, -10.0), dt=0.01)

        drone.move_to((0.0, 0.0, 0.0), dt=0.01)

    except KeyboardInterrupt:
        print("Flight interrupted by user.")

    finally:
        drone.disarm()
