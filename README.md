# Autoscaling: From Modeled to Measured

A Kubernetes autoscaling system driven by multi-objective reinforcement learning. The agent dynamically adjusts the Horizontal Pod Autoscaler (HPA) CPU target to simultaneously optimise latency, replica cost, and energy consumption — replacing the static default (50 % CPU) with a policy that responds to live workload patterns.

The project went through two main stages:
1. **Modeled energy** — power estimated analytically from CPU utilisation using a non-linear server model (`energy_model.py`).
2. **Measured energy** — power measured in real time via Kepler + Prometheus, eliminating the modelling assumptions.

---

## Architecture overview

```
Azure Functions trace (2 weeks, Jan 2021)
        │
        ▼
KubernetesEnv (Gymnasium)
  ├─ replays real invocation counts each minute
  ├─ fires load against the factorizator service using hey
  ├─ reads CPU / RAM / replica metrics from kubectl
  └─ reads pod power from Kepler / Prometheus (measured variant)
        │
        ▼
Multi-Objective PPO  (Transformer feature extractor)
  ├─ reward vector: [r_perf, r_cost, r_energy]
  ├─ scalarised with LinearReward (configurable weight profile)
  └─ action: choose HPA CPU target from {10, 30, 50, 70, 90} %
        │
        ▼
kubectl patch hpa → Kubernetes scales pods
```

### Reward profiles

| Profile | r_perf | r_cost | r_energy |
|---|---|---|---|
| `perf_focused` | 0.8 | 0.1 | 0.1 |
| `cost_focused` | 0.1 | 0.8 | 0.1 |
| `energy_focused` | 0.1 | 0.1 | 0.8 |
| `balanced` | 0.4 | 0.3 | 0.3 |

SLA threshold: **20 ms** average response latency.

---

## Repository layout

```
.
├── app.py                          # CPU-intensive Flask service (the workload)
├── Dockerfile                      # Container image for the factorizator service
├── energy_model.py                 # Analytic node power model (modeled-energy stage)
├── run_simulation_baseline.py      # Main script: MO-PPO + HPA baseline + SO-PPO baseline
├── run_simulation_hw_power.py      # MO-PPO with Kepler-measured power (single node)
├── run_simulation_hw_power_masterk8s.py  # Same, multi-node cluster variant
├── run_simulation_lstm_ppo.py      # LSTM-PPO variant (recurrent policy)
├── visualize_results.py            # Plot latency, power, replicas from test CSVs
├── analyze_traces.py               # Exploratory analysis of the Azure trace
├── launch_experiment.sh            # Helper to launch a named training run
├── requirements.txt
├── InvocationTraces                # Azure Functions trace file (not tracked by git)
├── Results/                        # Sample result CSVs and plots
└── k8s/
    ├── factorizator-namespace.yaml
    ├── factorizator-deployment.yaml
    ├── factorizator-service.yaml
    ├── hpa.yaml
    ├── RBAC.yaml
    └── components.yaml             # Metrics-server / Prometheus stack
```

---

## Prerequisites

- Kubernetes cluster — single-node minikube or a multi-node cluster (the scripts default to `microk8s kubectl`, change `KUBECTL_CMD` for vanilla kubectl)
- `kubectl` / `microk8s kubectl` configured and pointing at the cluster
- Prometheus reachable at `http://localhost:9090` (port-forward or NodePort)
- [Kepler](https://github.com/sustainable-computing-io/kepler) deployed in the cluster (required for the `hw_power` variants; the baseline variant uses the analytic model instead)
- [`hey`](https://github.com/rakyll/hey) load generator on `$PATH`
- Python 3.9+
- The Azure Functions trace file: `AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt` in the repo root (download from the [Azure Public Dataset](https://github.com/Azure/AzurePublicDataset))

---

## Setup

### 1. Python environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Build and push the factorizator image

```bash
docker build -t factorizator:latest .
# push to your registry or load into minikube:
minikube image load factorizator:latest
```

### 3. Deploy to Kubernetes

```bash
kubectl apply -f factorizator-namespace.yaml
kubectl apply -f factorizator-deployment.yaml
kubectl apply -f factorizator-service.yaml
kubectl apply -f hpa.yaml
kubectl apply -f RBAC.yaml
kubectl apply -f components.yaml     # metrics-server, Prometheus
```

Verify everything is running:

```bash
kubectl get pods -n factorizator
kubectl get hpa -n factorizator
```

### 4. Set the service URL

Edit the `url_app_service` constant in the simulation script you intend to run. For a NodePort service:

```python
url_app_service = "http://<node-ip>:<nodeport>/factor"
```

For Prometheus, update `PROMETHEUS_URL` if you are not using a local port-forward.

---

## Running experiments

### Training (multi-objective PPO)

```bash
python run_simulation_baseline.py --mode train --profile perf_focused
```

Available profiles: `perf_focused`, `cost_focused`, `energy_focused`, `balanced`.

The trained model and VecNormalize state are saved to `results/checkpoints/<profile>/`.

**Using the launch script** (adds timestamped metadata and logs):

```bash
bash launch_experiment.sh perf_focused 1 42
# args: <profile> <run_id> <seed>
```

### Testing a trained policy

```bash
python run_simulation_baseline.py --mode test --profile perf_focused
```

Results are written to `results/test/<profile>/<profile>_test_run.csv`.

### Baselines

```bash
# Baseline 1 — static HPA at 50 % CPU (no RL)
python run_simulation_baseline.py --mode test_hpa_baseline

# Baseline 2 — single-objective PPO (train then test)
python run_simulation_baseline.py --mode train_so_ppo
python run_simulation_baseline.py --mode test_so_ppo
```

### Measured-energy variant (Kepler)

Uses live Kepler telemetry from Prometheus instead of the analytic model:

```bash
python run_simulation_hw_power.py --mode train --profile balanced
python run_simulation_hw_power.py --mode test  --profile balanced
```

Multi-node cluster variant (reads per-node metrics from `masterk8s`, `worker11`–`worker14`):

```bash
python run_simulation_hw_power_masterk8s.py --mode train --profile balanced
```

### LSTM-PPO variant

```bash
python run_simulation_lstm_ppo.py --mode train --profile perf_focused
```

---

## Monitoring and visualisation

TensorBoard logs are written to `results/tb/<profile>/` during training and `results/tb_test/<profile>/` during testing:

```bash
tensorboard --logdir results/tb/
```

Plot latency, power, and replica traces from a test CSV:

```bash
python visualize_results.py
# edit RESULTS_FILE at the top of the script to point to the CSV you want
```

---

## Output files

Each run produces:

| File | Contents |
|---|---|
| `results/checkpoints/<profile>/final_model.zip` | Saved PPO policy |
| `results/checkpoints/<profile>/vecnorm.pkl` | Observation normalisation state |
| `results/logs/<profile>_train_<timestamp>.csv` | Per-step training metrics |
| `results/test/<profile>/<profile>_test_run.csv` | Per-step test metrics |
| `results/tb/` | TensorBoard event files |

Test CSV columns: `timestep`, `latency`, `p95_latency`, `success_ratio`, `replicas`, `avg_cpu_percent`, `pod_power`, `hpa_target`, `r_perf`, `r_cost`, `r_energy`, `reward_scalar`.

---

## Workload trace

Invocation counts are replayed from the **Azure Functions Public Dataset (Jan 2021)**, two-week trace. The simulation samples four days at random (three for training, one for testing) and replays them at one control step per minute. Each step fires the replayed number of requests against the live factorizator service using `hey`.

---

## Energy model (baseline variant)

`energy_model.py` implements a non-linear server power model:

```
P_node = P_idle + (P_max - P_idle) × util²
```

Pod power is attributed proportionally to CPU share. The measured-energy variants (`hw_power`) replace this with Kepler's per-container joule counters exposed through Prometheus.

---

## License

See [LICENSE](LICENSE).
