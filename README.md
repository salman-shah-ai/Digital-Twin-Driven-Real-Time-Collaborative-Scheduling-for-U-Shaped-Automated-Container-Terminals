# Digital Twin–Driven Real-Time Collaborative Scheduling for U-Shaped Automated Container Terminals

### Practical Proof-of-Concept (U-ACT_POC)

## 📘 Overview

This project is a **practical proof-of-concept (POC)** inspired by the research paper:

> **“Digital twin-driven real-time collaborative scheduling for U-shaped automated container terminals”**

The goal is to demonstrate the **core scheduling logic and reinforcement learning approach** proposed in the paper — in a compact, runnable Python environment suitable for experimentation and teaching.

This implementation simplifies the original **digital twin (DT)** model, which in the paper was developed with **FlexSim and a PPO algorithm**, into a lightweight **Python + PyTorch** environment that captures the essence of:

* Real-time collaborative scheduling among **AGVs**, **YCs**, and **ETs**.
* **Event-driven** dynamics in a U-shaped terminal layout.
* The **six composite scheduling rules** (LRTC/SRTC × EUTA/SPTA/NLRA + FCFSY).
* A compact **Proximal Policy Optimization (PPO)** agent that learns dispatching policies.

---

## 🧩 Project Structure

```
📂 project_root/
├── UACT_poc.ipynb           # Jupyter Notebook (interactive, runnable version)
├── uact_poc.py              # Python script (same logic, runnable via CLI)
├── uact_poc_policy.pth      # Trained demo policy (saved after running)
└── README.md                # This documentation
```

---

## 🚀 Installation

### 1. Clone or copy this project

Download the files into a local folder, for example:

```bash
git clone <your-repo-url>
cd project_root
```

*(If you only have the files, just place `UACT_poc.ipynb`, `uact_poc.py`, and `README.md` in the same directory.)*

### 2. Install dependencies

Make sure Python 3.8+ is installed. Then install required packages:

```bash
pip install torch numpy matplotlib
```

---

## ⚙️ Running the Project

### **Option 1 – Notebook mode**

1. Open `UACT_poc.ipynb` in **Jupyter Notebook**, **VS Code**, or **Google Colab**.
2. Run all cells — it will:

   * Initialize a simplified **U-ACT simulator**
   * Create a **PPO agent**
   * Train it briefly on a small instance
   * Plot training rewards
   * Save a model to `uact_poc_policy.pth`

### **Option 2 – Script mode**

Run directly from terminal or command prompt:

```bash
python uact_poc.py
```

The script runs a short demo training session and prints training logs like:

```
Ep 5/20, reward=0.0123, time=46.23, completed=40
Ep 10/20, reward=0.0198, time=48.90, completed=40
...
Saved demo policy to uact_poc_policy.pth
```

---

## 🧠 Model Components

| Component                            | Description                                                                                    |
| ------------------------------------ | ---------------------------------------------------------------------------------------------- |
| **Environment**                      | Simplified simulator modeling tasks (containers), AGVs, YCs, and ET arrivals                   |
| **State Space (20-dim)**             | Features consistent with Table 4 in the paper — counts, means, stds, and queue/load indicators |
| **Action Space (6 composite rules)** | LRTC/SRTC × EUTA/SPTA/NLRA + FCFSY prioritization                                              |
| **Reward**                           | Approximates Eq.(19) in the paper: weighted combination of completion rate and queue growth    |
| **Learning Algorithm**               | Compact **Proximal Policy Optimization (PPO)** with an actor-critic neural network             |

---

## 📊 Outputs

* **Training curve** — average reward per episode
* **Console logs** — episode rewards, completion stats, runtime
* **Saved model** — trained PPO weights saved as `uact_poc_policy.pth`

---

## ⚠️ Limitations

This implementation is designed for **conceptual demonstration** and educational use.
It does **not** replicate the full fidelity of the paper’s **FlexSim + DT** system, which involved:

* Detailed spatial simulation of crane/vehicle trajectories
* Stochastic disturbances and delays
* High-performance optimization and long PPO training

Instead, this POC models those mechanisms **abstractly**, preserving the **decision-making structure** and **RL learning workflow**.

---

## 🔬 Future Improvements

You can extend this project by:

* Incorporating **multi-YC and multi-QC** configurations
* Modeling **energy consumption** or **AGV path planning**
* Integrating a **discrete-event simulator** (SimPy or FlexSim API)
* Expanding PPO training and adding **rule-based baselines** for comparison

---

## 👨‍💻 Author & Credits

Developed by **[Salman Shah](https://github.com/salman-shah-ai)**.
Based on the academic work by *Li et al., “Digital twin-driven real-time collaborative scheduling for U-shaped automated container terminals.”*

---

## 📁 Local Paths

If running in your environment, you can find:

* Notebook: `UACT_poc.ipynb`
* Script: `uact_poc.py`
* Saved policy: `uact_poc_policy.pth`

*(These are local files in your working directory.)*

---

### ✅ Example next steps

**a.** Add evaluation comparing PPO vs heuristic rules (6 composites).
**b.** Visualize container throughput, AGV utilization, and makespan improvements.

---
