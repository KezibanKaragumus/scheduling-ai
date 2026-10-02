import argparse
import torch
import matplotlib.pyplot as plt
from sim.core import User, Channel, SlotSimulator
from data.generate import poisson_traffic
from schedulers.edf import EDFScheduler
from schedulers.wfq import WFQScheduler
from sim.environment import SchedulingEnv
from schedulers.rl_dqn import QNet

# --------------------------- #
#   KLASİK PLANLAYICILAR     #
# --------------------------- #

def run_once(sched_name="edf", users_n=6, slots=2000, total_rb=12, lam=0.35, seed=7):
    users = [User(id=i, qos_delay_ms=50, weight=1.0) for i in range(users_n)]
    poisson_traffic(users, slots=1, lam=0.0, seed=seed)  # Kuyruğu temizle
    chan = Channel(seed=seed)
    sim = SlotSimulator(users, chan, total_rb=total_rb, slot_ms=1)

    if sched_name == "edf":
        sched = EDFScheduler(default_rb_per_user=2)
    elif sched_name == "wfq":
        weights = {i: 1.0 for i in range(users_n)}
        sched = WFQScheduler(weights=weights)
    else:
        raise ValueError("Unknown scheduler")

    from random import Random
    rng = Random(seed)

    for t in range(slots):
        # Yeni paket gelişleri
        for u in users:
            if rng.random() < lam:
                from sim.core import Packet
                size = int(max(4000, rng.gauss(12000, 12000 * 0.1)))
                deadline_t = t + 50
                u.queue.append(Packet(size_bits=size, arrival_t=t, deadline_t=deadline_t))

        gains = chan.gains(users)
        alloc = sched.select(users, {"total_rb": total_rb, "channel_gains": gains})
        sim.step(alloc)

    m = sim.metrics()
    return m


# --------------------------- #
#       GRAFİK ÇİZDİRME      #
# --------------------------- #

def plot_results(results):
    sched_names = list(results.keys())
    delays = [results[s]["avg_delay_ms"] for s in sched_names]
    throughputs = [results[s]["throughput_mbps"] for s in sched_names]
    fairness = [results[s]["fairness"] for s in sched_names]
    violations = [results[s]["deadline_violations"] for s in sched_names]

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    plt.bar(sched_names, delays, color=['steelblue', 'orange', 'limegreen'][:len(sched_names)])
    plt.title("Ortalama Gecikme (ms)")

    plt.subplot(2, 2, 2)
    plt.bar(sched_names, throughputs, color=['green', 'purple', 'cyan'][:len(sched_names)])
    plt.title("Ortalama Throughput (Mbps)")

    plt.subplot(2, 2, 3)
    plt.bar(sched_names, fairness, color=['gray', 'skyblue', 'lightgreen'][:len(sched_names)])
    plt.title("Adalet (Fairness)")

    plt.subplot(2, 2, 4)
    plt.bar(sched_names, violations, color=['red', 'salmon', 'pink'][:len(sched_names)])
    plt.title("Deadline İhlali Sayısı")

    plt.tight_layout()
    plt.show()


# --------------------------- #
#       RL (DQN) MODELİ      #
# --------------------------- #

def run_rl(users_n=6, slots=2000, total_rb=12, lam=0.35, seed=7, model_path="models/dqn.pt"):
    env = SchedulingEnv(users_n=users_n, slots=slots, total_rb=total_rb, lam=lam, seed=seed)
    obs, _ = env.reset()
    state_dim = obs.shape[0]
    action_dim = env.action_space.n

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    q = QNet(state_dim, action_dim).to(device)
    q.load_state_dict(torch.load(model_path, map_location=device))
    q.eval()

    done = False
    while not done:
        with torch.no_grad():
            s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            a = int(torch.argmax(q(s), dim=1).item())
        obs, r, term, trunc, _ = env.step(a)
        done = term or trunc

    return env.sim.metrics()


# --------------------------- #
#       KARŞILAŞTIRMALAR     #
# --------------------------- #

def compare():
    edf_m = run_once("edf")
    wfq_m = run_once("wfq")
    print("EDF:", edf_m)
    print("WFQ:", wfq_m)
    results = {"EDF": edf_m, "WFQ": wfq_m}
    plot_results(results)


def compare_with_rl():
    edf_m = run_once("edf")
    wfq_m = run_once("wfq")
    try:
        rl_m = run_rl()
        print("RL(DQN):", rl_m)
        results = {"EDF": edf_m, "WFQ": wfq_m, "RL(DQN)": rl_m}
    except Exception as e:
        print("RL modeli yüklenemedi:", e)
        results = {"EDF": edf_m, "WFQ": wfq_m}
    plot_results(results)


# --------------------------- #
#         ANA PROGRAM         #
# --------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="EDF ve WFQ'yu karşılaştır")
    parser.add_argument("--compare_rl", action="store_true", help="Eğitilmiş RL modeliyle karşılaştır")
    parser.add_argument("--sched", type=str, default="edf", help="edf|wfq")
    args = parser.parse_args()

    if args.compare_rl:
        compare_with_rl()
    elif args.compare:
        compare()
    else:
        print(run_once(args.sched))
