# Mostly written by Claude, I wrote all those comments to explain stuff to myself
import legoeducation as le
import random
import time

# --- Network ---
# W[i][j] = weight from input j to output i
# two ins and outs --> two neurons in only layer, each with two weights --> 3 x 2 weight matrix and 3 x 1 bias vector (omitted for simplicity)
# Just randomly assigning this for now and will update it as it learns
W = [[random.uniform(-0.1, 0.1) for _ in range(2)] for _ in range(3)]

def forward(x):
    """Compute Q-values for both actions."""
    # No bias vector, so just passting through by multiplying in vector by weight matrix (Q function), predicts reward?
    return [
        W[0][0]*x[0] + W[0][1]*x[1],  # Q(action 0)
        W[1][0]*x[0] + W[1][1]*x[1],  # Q(action 1)
        W[2][0]*x[0] + W[2][1]*x[1],  # Q(action 2)
    ]

def update(x, action, target, lr=0.1):
    """Nudge the weights for the chosen action toward the target."""
    # Predict reward
    q = forward(x)

    # Calculate the error (just difference here)
    error = q[action] - target                  # how wrong we were

    # Need to update the weights
    # Say we chose action 0, then we want to take the weights in the first row (that was the equation we used to get that strong reward), and update them
    # This equation calculates the derivative of the loss equation (MSE) with respect to W[action] times the learning rate
    # We want to subtract that vector so that if the error was negative (our pred reward was too low), increase the weight to make pred reward higher. 
    W[action][0] -= lr * 2 * error * x[0]       # gradient step
    W[action][1] -= lr * 2 * error * x[1]

# --- Environment ---
def get_reward(sensorL, sensorR):
    # maybe change this, for now I am just saying that the reward is inversely proportional to the sum of the sensor readings
    # ie, reward when the sensors see dark (are on the line)
    return (200 - (sensorL + sensorR))

def move(dm, action):
    if action == 0:
        #print("left")
        turn_left(dm, 5)
    elif action == 1:
        #print("right")
        turn_right(dm, 5)
    else:
        #print("forwards")
        forwards(dm)

def turn_right(dm, degrees):
    dm.movement_turn_for_degrees(degrees=degrees, direction=le.MOVEMENT_TURN_DIRECTION_RIGHT, speed=10)
    time.sleep(0.5)

def turn_left(dm, degrees):
    dm.movement_turn_for_degrees(degrees=degrees, direction=le.MOVEMENT_TURN_DIRECTION_LEFT, speed=10)
    time.sleep(0.5)

def forwards(dm):
    dm.motor_run(direction=le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE, motor=le.MOTOR_RIGHT, speed=10)
    dm.motor_run(direction=le.MOTOR_MOVE_DIRECTION_CLOCKWISE, motor=le.MOTOR_LEFT, speed=10)
    time.sleep(0.25)

# --- Training loop ---
# start the exploration high because we know nothing
exploration_rate = 0.5 

# will have left and right sensor readings
state = []

dm = le.DoubleMotor()
cL = le.ColorSensor()
cR = le.ColorSensor()

dm.connect(card_serial="1128", card_color=le.LEGO_COLOR_PURPLE)
cL.connect(card_serial="1128", card_color=le.LEGO_COLOR_PURPLE)
cR.connect(card_serial="1131")

i = 0
while True:
    
    # Get current state
    state = [cL.sensor.reflection, cR.sensor.reflection]

    # Get the q value (reward)
    q = forward(state)

    # choose to explore or exploit
    # higher exploration_rates mean we are more likely to take the exploration path (more numbers are less than it)
    if random.random() < exploration_rate:
        # explore --> choose randomly 
        print("explored")
        action = random.randint(0, 2)
    else:
        print("exploited")
        # exploit --> just take the known good action calculated by Q
        action = q.index(max(q))
    
    # take action:
    move(dm, action)

    # calculate what the reward was from that action
    reward = get_reward(cL.sensor.reflection, cR.sensor.reflection)
    #print("reward was: " + str(reward))
    
    # Our target is just the immediate reward, 
    target = reward

    # Update the values
    update(state, action, target)

    # Decay exploration_rate: explore less over time
    exploration_rate = max(0.01, exploration_rate * 0.99)

    if (i % 10 == 0):
        print(str(W))

    i += 1