# Flight Data Transmission & Anomaly Detection  
## A Simulation Study

## The Starting Question

Aircraft generate a huge amount of data during flight — everything from airspeed and altitude to control surface positions and engine parameters. In reality, sending all this data continuously to the ground is often not possible due to limited bandwidth, satellite costs, and communication delays.

So the natural question is:

**If we can't send everything perfectly, can we still understand what's happening on the aircraft and catch problems early?**

This project tries to answer that question through simulation.

---

## How the System Works

The system is built as a complete simulation with two main parts:

- **In-Air System (Aircraft):**  
  A simplified Boeing 747 longitudinal model with an autopilot and fault injection capability.

- **Ground Station:**  
  Receives limited and imperfect data, reconstructs the flight state, and detects anomalies.

Between them is a communication channel that introduces:
- Packet loss  
- Latency  
- Bandwidth constraints  

The goal is simple but challenging:

Send compressed and imperfect data, and still recover useful flight information and detect issues like wind gusts, sensor failures, or actuator faults.

---

## Key Components

### 1. Aircraft Model and Autopilot

- Based on a longitudinal dynamics model of a B747  
- PID controller for pitch  
- PI/P controllers for velocity and thrust  

One important observation was that gain polarity matters. Due to aircraft physics, negative gains were required for stability in certain loops.

---

### 2. Data Compression

To reduce bandwidth usage:

- 7-bit linear quantization (128 levels per signal)  
- Bit-packing into 32-bit frames  

This reduces the data rate significantly but introduces quantization noise, creating a trade-off between accuracy and bandwidth.

---

### 3. Communication Channel

The channel simulates realistic constraints:

- Random packet loss  
- Variable latency  
- Limited bandwidth  

---

### 4. Ground-Side Processing

On the ground:

- Data is first dequantized  
- An Adaptive Kalman Filter (AKF) reconstructs the flight states  
- Residual analysis is used for anomaly detection  

The adaptive filter adjusts how much it trusts incoming data based on its reliability.

---

## What I Observed

### When it works well

- Moderate compression and manageable packet loss still allow accurate reconstruction  
- Wind gusts and sensor faults are clearly visible in residual signals  

### Where it struggles

- High compression removes important details  
- High packet loss delays detection  
- Strong disturbances can saturate the autopilot  

### Main takeaway

There is a clear trade-off:

Sending less data can still work, but beyond a point, important anomalies start getting missed.

---

## Evaluation

Performance was evaluated using:

- Mean Squared Error (MSE) between true and estimated states  
- Residual magnitude and behavior  
- Kalman gain variation  
- Detection latency  

---

## Hybrid Approach: GRU + Kalman Filter

After implementing the Kalman Filter-based system, a data-driven extension was added using a Gated Recurrent Unit (GRU).

The idea was to improve estimation when the physics-based model is not perfect.

### Core Idea

Instead of predicting the state directly, the GRU learns the residual error of the Kalman Filter.

Final estimate:

---

### Why GRU

- Flight data is sequential  
- Gusts and disturbances have temporal structure  
- Feedforward networks lack memory  
- Standard RNNs suffer from vanishing gradients  

GRU provides a balance by retaining useful temporal information while remaining computationally efficient.

---

### Inputs to the GRU

- Kalman Filter estimated states  
- Control inputs (elevator, thrust)  
- Recent residual (innovation) history  

---

### Results

The hybrid system showed improvements, especially during disturbances:

- Better altitude and velocity estimation  
- Lower estimation error  
- Faster response to sudden changes  

### Insight

The Kalman Filter handles known physics, while the GRU learns unmodeled disturbances. Together, they complement each other effectively.

---

## Limitations and Future Work

### Limitations

- GRU requires good training data  
- Risk of overfitting  
- Higher computational cost than KF alone  
- Tested only in simulation  

### Future Work

- Compare GRU with LSTM  
- Use real flight datasets  
- Explore online/adaptive learning  
- Extend to EKF or UKF for nonlinear systems  

---

## Why This Project Matters

In aviation systems:

- Sending all data is often impractical  
- Over-compression risks losing critical information  

This project explores a middle ground using estimation and learning techniques to extract meaningful insights from limited data.

---

## Final Thought

Even with imperfect and incomplete data, useful information can still be extracted by combining classical models with learning-based approaches.

---

## Author

Saharsh Kumar 
B.Tech. Electronics & Communication Engineering  
IIIT Dharwad  

---

## Note

This project was developed primarily to understand engineering trade-offs in constrained systems rather than to build a production-ready solution.
