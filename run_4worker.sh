#!/bin/bash
kubectl port-forward svc/kube-prom-stack-kube-prome-prometheus -n observability 9090:9090 > /dev/null 2>&1 &
sleep 5
WORKERS=4
CPU=2vCPU
RAM=16GB
CLUSTER=masterk8s

for PROFILE in perf_focused cost_focused energy_focused balanced; do
  echo ">>> Training $PROFILE"
  python3 run_simulation_hw_power_masterk8s.py --mode train --profile $PROFILE \
    --num-workers $WORKERS --worker-cpu $CPU --worker-ram $RAM --cluster $CLUSTER

  echo ">>> Testing $PROFILE"
  python3 run_simulation_hw_power_masterk8s.py --mode test --profile $PROFILE \
    --num-workers $WORKERS --worker-cpu $CPU --worker-ram $RAM --cluster $CLUSTER
done
