"""
Self-balancing cart-pole built from a single LEGO Education Double Motor.

Build: the Double Motor unit is mounted upright with one wheel on each
motor output. The unit's own tilt (read from its built-in IMU) is the
pole angle; wheel rotation (read from the motor encoders) is the cart
position. A wheeled inverted pendulum is dynamically equivalent to a
classic cart-pole, so it's balanced here with an LQR state-feedback
controller on state x = [position, velocity, pitch, pitch_rate].
"""

import time
import numpy as np
from scipy.linalg import solve_continuous_are

import legoeducation as le

# --- Physical parameters (measure/tune for your build) ---------------------
# All of these are placeholders. The controller is only as good as these
# numbers, so measure what you can and tune the rest by feel.

WHEEL_RADIUS_M = 0.043          # radius of the driven wheels, meters
BODY_MASS_KG = 0.300             # mass of everything above the wheel axle
                                 # (Double Motor unit + frame + battery)
BODY_COM_HEIGHT_M = 0.12        # height of the body's center of mass
                                 # above the wheel axle, meters
EFFECTIVE_CART_MASS_KG = 0.200   # reflected translational inertia of the
                                 # wheels/drivetrain at the contact patch
GRAVITY = 9.81

# Point-mass approximation: I_pivot = m * l^2. Override this if you have a
# better estimate of the body's moment of inertia about the wheel axle.
BODY_INERTIA_ABOUT_AXLE = BODY_MASS_KG * BODY_COM_HEIGHT_M ** 2

# Newtons of net wheel-contact force produced per 1% commanded duty cycle.
# There's no published torque constant for this motor, so find this by
# testing: command a duty cycle, measure the resulting acceleration, and
# back out F = mass * acceleration. The sign/order of magnitude matters far
# more than precision -- LQR is robust to a reasonable misestimate here.
FORCE_PER_DUTY_PERCENT = 0.02

# --- LQR cost weights (tune to trade off balance vs. position-holding) -----
Q = np.diag([0.5, 50.0, 20.0])   # weights on [position, pitch, pitch_rate]
R = np.array([[0.5]])                # weight on control effort (duty cycle)

# --- Safety / loop parameters ------------------------------------------------
FALL_ANGLE_DEG = 35.0            # past this tilt from calibrated upright, assume it fell
LOOP_HZ = 100.0
NOTIFICATION_DELAY_MS = 20       # matches LOOP_HZ
CALIBRATION_SECONDS = 1.0


def compute_lqr_gain():
    """Build the linearized wheeled-inverted-pendulum model and solve for
    the LQR state-feedback gain K, where u = -K @ x.

    State x = [s, s_dot, theta, theta_dot] (theta measured from upright).
    """
    M = EFFECTIVE_CART_MASS_KG
    m = BODY_MASS_KG
    l = BODY_COM_HEIGHT_M
    I = BODY_INERTIA_ABOUT_AXLE
    g = GRAVITY

    denom = (M + m) * I - (m * l) ** 2

    A = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, (M + m) * m * g * l / denom, 0],
    ])
    B = np.array([
        [WHEEL_RADIUS_M],
        [0],
        [-m * l / denom],
    ])

    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K


def read_state(doublemotor, prev_position_m, prev_pitch_deg, prev_time):
    """Read the current [position, velocity, pitch, pitch_rate] state.
    Velocity and pitch_rate are derived by finite difference since the
    motor/IMU notification units for instantaneous rate aren't documented."""
    now = time.monotonic()
    dt = max(now - prev_time, 1e-3)

    left_deg = doublemotor.motor[le.MOTOR_LEFT].position
    right_deg = doublemotor.motor[le.MOTOR_RIGHT].position
    wheel_deg = (left_deg + right_deg) / 2.0
    position_m = wheel_deg * (np.pi / 180.0) * WHEEL_RADIUS_M

    pitch_deg = doublemotor.imu_device.pitch/10
    pitch_rate_deg_s = (pitch_deg - prev_pitch_deg) / dt

    return position_m, pitch_deg, pitch_rate_deg_s, now


def calibrate(doublemotor):
    """Zero the wheel encoders and measure the resting pitch offset so
    'upright' matches this build's actual balance point."""
    doublemotor.motor_reset_relative_position(motor=le.MOTOR_BOTH)

    samples = []
    end_time = time.monotonic() + CALIBRATION_SECONDS
    while time.monotonic() < end_time:
        samples.append(doublemotor.imu_device.pitch)
        time.sleep(0.02)

    pitch_offset_deg = sum(samples) / len(samples)
    return pitch_offset_deg


def main():
    K = compute_lqr_gain()
    print(f"LQR gain K = {K}")

    doublemotor = le.DoubleMotor()
    print("Scanning for Double Motor...")
    doublemotor.connect(card_serial="5186", device_notification_delay=NOTIFICATION_DELAY_MS)
    if not doublemotor.connected:
        print("Could not connect to Double Motor.")
        return

    doublemotor.motor_set_end_state(le.MOTOR_END_STATE_BRAKE, motor=le.MOTOR_BOTH)

    print("Hold the robot upright and still for calibration...")
    pitch_offset_deg = 3.5
    print(f"Calibrated upright pitch offset: {pitch_offset_deg:.2f} deg")

    print("Balancing. Press Ctrl+C to stop.")
    loop_dt = 1.0 / LOOP_HZ
    position_m = 0.0
    pitch_deg = pitch_offset_deg
    prev_time = time.monotonic()

    try:
        while True:
            position_m, pitch_deg, pitch_rate_deg_s, prev_time = read_state(
                doublemotor, position_m, pitch_deg, prev_time
            )
            pitch_error_deg = pitch_deg - pitch_offset_deg
            if abs(pitch_error_deg) > FALL_ANGLE_DEG:
                print("Fell over -- stopping motors.")
                doublemotor.motor_stop()
                input("press to continue")
                position_m = 0
                time.sleep(0.5)

            theta_rad = np.radians(pitch_error_deg)
            theta_dot_rad_s = np.radians(pitch_rate_deg_s)

            state = np.array([position_m, theta_rad, theta_dot_rad_s])
            u = float(-(K @ state)[0])  # Newtons of commanded wheel-contact force
            duty = float(np.clip(40*u, -100, 100))
            print(u-duty)
            doublemotor.movement_move(speed=duty, blocking=False)

            time.sleep(loop_dt)
    except KeyboardInterrupt:
        print("Stopping.")
    finally:
        doublemotor.motor_stop(motor=le.MOTOR_BOTH)
        doublemotor.disconnect()


if __name__ == "__main__":
    main()
