import airsim  # to call AirSim APIs
import time  # for time.time()
import csv  # for logging drone posit
import os  # for deleting log files


class PIDController:  # here is the PID class
    def __init__(self, kp, ki, kd):  # constructor method with "self" instance
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0

    def calculate(
        self, setpoint, current, steptimet
    ):  # method to calculate PID outputs
        error = setpoint - current
        self.integral += error * steptimet
        derivative = (error - self.previous_error) / steptimet
        self.previous_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return output


# connection setup for AirSim multirotor and car
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# this part is for logging the drone positions you can ignore it
# log_file = "drone_positions.csv"
# if os.path.exists(log_file):
#   os.remove(log_file)  # Remove old log file if it exists

# with open(log_file, mode='w', newline='') as file:
#   writer = csv.writer(file)
#  writer.writerow(["Time", "X", "Y", "Z"])

# Target positions are set here. altitude is negative because upwards is negative in AirSim (everything is in SI units)
target_altitude = -20
target_x = 20
target_y = 20
reverse_target_x = 0
reverse_target_y = 0


pid_altitude = PIDController(
    kp=20, ki=0.5, kd=0.7
)  # PID objects. we have to make 3 different ones. one for each axis. can create more if we want to control any other 21 states
pid_position_x = PIDController(kp=5, ki=0.2, kd=0.3)
pid_position_y = PIDController(kp=5, ki=0.2, kd=0.3)

steptimet = 0.001  # 1 ms of step time. although unrealistic it gives us a response which mimics realtime systems
tolerance = 0.05  # if the drone reaches within 5 cm of the target positions mentioned above we can break the loop and move to the next phase

try:
    while True:  # altitude control
        current_time = time.time()
        current_altitude = (
            client.getMultirotorState().kinematics_estimated.position.z_val
        )
        current_x = client.getMultirotorState().kinematics_estimated.position.x_val
        current_y = client.getMultirotorState().kinematics_estimated.position.y_val

        velocity_z = pid_altitude.calculate(
            target_altitude, current_altitude, steptimet
        )  # object uses calculate method to calculate pid outputs
        velocity_z = max(
            -2.0, min(2.0, velocity_z)
        )  # very important to control overshooots

        client.moveByVelocityAsync(0, 0, velocity_z, steptimet).join()

        # writer.writerow([current_time, current_x, current_y, current_altitude]) #this is for writing a row in CSV

        if abs(current_altitude - target_altitude) < tolerance:
            print("altitude reahced")
            break

    while True:  # same but for moving in x while maintaining altitude
        current_time = time.time()
        current_x = client.getMultirotorState().kinematics_estimated.position.x_val
        current_altitude = (
            client.getMultirotorState().kinematics_estimated.position.z_val
        )

        velocity_x = pid_position_x.calculate(target_x, current_x, steptimet)
        velocity_x = max(-5.0, min(5.0, velocity_x))

        velocity_z = pid_altitude.calculate(
            target_altitude, current_altitude, steptimet
        )
        velocity_z = max(-2.0, min(2.0, velocity_z))

        client.moveByVelocityAsync(velocity_x, 0, velocity_z, steptimet).join()

        # writer.writerow([current_time, current_x, 0, current_altitude])

        if abs(current_x - target_x) < tolerance:
            print("Target reached")
            break

    while True:  # same but for moving in y while maintaining altitude
        current_time = time.time()
        current_y = client.getMultirotorState().kinematics_estimated.position.y_val
        current_altitude = (
            client.getMultirotorState().kinematics_estimated.position.z_val
        )

        velocity_y = pid_position_y.calculate(target_y, current_y, steptimet)
        velocity_y = max(-5.0, min(5.0, velocity_y))

        velocity_z = pid_altitude.calculate(
            target_altitude, current_altitude, steptimet
        )
        velocity_z = max(-2.0, min(2.0, velocity_z))

        client.moveByVelocityAsync(0, velocity_y, velocity_z, steptimet).join()

        # writer.writerow([current_time, target_x, current_y, current_altitude])

        if abs(current_y - target_y) < tolerance:
            print("Target reached")
            break

    while True:  # same but for moving in x while maintaining altitude
        current_time = time.time()
        current_x = client.getMultirotorState().kinematics_estimated.position.x_val
        current_altitude = (
            client.getMultirotorState().kinematics_estimated.position.z_val
        )

        velocity_x = pid_position_x.calculate(reverse_target_x, current_x, steptimet)
        velocity_x = max(-5.0, min(5.0, velocity_x))

        velocity_z = pid_altitude.calculate(
            target_altitude, current_altitude, steptimet
        )
        velocity_z = max(-2.0, min(2.0, velocity_z))

        client.moveByVelocityAsync(velocity_x, 0, velocity_z, steptimet).join()

        # writer.writerow([current_time, current_x, target_y, current_altitude])

        if abs(current_x - reverse_target_x) < tolerance:
            print("Target reached")
            break

    while True:  # same but for moving in y while maintaining altitude
        current_time = time.time()
        current_y = client.getMultirotorState().kinematics_estimated.position.y_val
        current_altitude = (
            client.getMultirotorState().kinematics_estimated.position.z_val
        )

        velocity_y = pid_position_y.calculate(reverse_target_y, current_y, steptimet)
        velocity_y = max(-5.0, min(5.0, velocity_y))

        velocity_z = pid_altitude.calculate(
            target_altitude, current_altitude, steptimet
        )
        velocity_z = max(-2.0, min(2.0, velocity_z))

        client.moveByVelocityAsync(0, velocity_y, velocity_z, steptimet).join()

        # writer.writerow([current_time, reverse_target_x, current_y, current_altitude])

        if abs(current_y - reverse_target_y) < tolerance:
            print("origin reached")
            break

    time.sleep(2)

except KeyboardInterrupt:
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)

# print(f"Drone positions logged to {log_file}")
