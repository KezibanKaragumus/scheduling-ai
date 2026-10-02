import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sim.core import User, Channel, SlotSimulator, Packet

class SchedulingEnv(gym.Env):
    """
    Her step = 1 RB tahsisi. Bir slotta total_rb kadar step atılır.
    RB'ler bitince sim.step() çağrılır, slot metrik farkından ödül hesaplanır.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 users_n=6,
                 slots=2000,
                 total_rb=12,
                 lam=0.35,
                 seed=7,
                 qos_delay_ms=50):
        super().__init__()
        self.users_n = users_n
        self.slots_total = slots
        self.total_rb = total_rb
        self.lam = lam
        self.seed_val = seed
        self.qos_delay_ms = qos_delay_ms

        # Aksiyon: 0..users_n-1 (1 RB hangi kullanıcıya?)
        self.action_space = spaces.Discrete(self.users_n)

        # Gözlem: kullanıcı başına 4 özellik + global 1 = users_n*4 + 1
        # [queue_bits, head_deadline, last_gain, last_alloc] (normalize) * N + [remaining_rb]
        obs_dim = self.users_n * 4 + 1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.rng = np.random.RandomState(self.seed_val)
        self._build_world()

    # ---------- world ----------

    def _build_world(self):
        self.users = [User(id=i, qos_delay_ms=self.qos_delay_ms, weight=1.0) for i in range(self.users_n)]
        self.channel = Channel(seed=self.seed_val)
        self.sim = SlotSimulator(self.users, self.channel, total_rb=self.total_rb, slot_ms=1)
        self.curr_slot = 0
        self.rb_used_in_slot = 0
        self.pending_alloc = {u.id: 0 for u in self.users}
        self.idle_in_slot = 0  # boş kuyruk hedefleme sayacı
        self.last_metrics = self.sim.metrics()            # başlangıç metrikleri
        self.last_gains = self.channel.gains(self.users)  # başlangıç kazançları
        self.last_sent_bits_total = sum(self.sim.sent_bits.values())

    def _spawn_packets_for_slot(self, t):
        for u in self.users:
            if self.rng.rand() < self.lam:
                size = int(max(4000, self.rng.normal(12000, 12000 * 0.1)))
                deadline_t = t + 50
                u.queue.append(Packet(size_bits=size, arrival_t=t, deadline_t=deadline_t))

    # ---------- features ----------

    def _features(self):
        MAX_QUEUE_BITS = 200000.0
        MAX_DEADLINE_SLOTS = 100.0
        MAX_GAIN = 5.0
        MAX_ALLOC = float(self.total_rb)

        feats = []
        for u in self.users:
            queue_bits = float(sum(p.size_bits for p in u.queue))
            head_dl = (u.queue[0].deadline_t - self.sim.t) if u.queue else MAX_DEADLINE_SLOTS
            gain = float(self.last_gains.get(u.id, 1.0))
            alloc = float(self.pending_alloc[u.id])

            feats.extend([
                min(queue_bits / MAX_QUEUE_BITS, 1.0),
                min(max(head_dl, 0) / MAX_DEADLINE_SLOTS, 1.0),
                min(gain / MAX_GAIN, 1.0),
                min(alloc / MAX_ALLOC, 1.0),
            ])
        remaining_rb_norm = (self.total_rb - self.rb_used_in_slot) / float(self.total_rb)
        feats.append(remaining_rb_norm)
        return np.array(feats, dtype=np.float32)

    # ---------- gym api ----------

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self._build_world()
        self._spawn_packets_for_slot(self.curr_slot)
        self.last_gains = self.channel.gains(self.users)
        obs = self._features()
        return obs, {}

    def step(self, action):
        # 1 RB'yi seçilen kullanıcıya ver
        uid = int(action) % self.users_n
        if not self.users[uid].queue_empty():
            self.pending_alloc[uid] += 1
        else:
            # boş kuyruk hedeflendi
            self.idle_in_slot += 1

        self.rb_used_in_slot += 1

        terminated = False
        truncated = False
        reward = 0.0
        info = {}

        # Slot tamamlandıysa sim.step() ve ödül
        if self.rb_used_in_slot >= self.total_rb:
            alloc_list = [(u.id, self.pending_alloc[u.id]) for u in self.users if self.pending_alloc[u.id] > 0]

            # simülasyonu ilerlet
            self.last_gains = self.sim.step(alloc_list)
            new_metrics = self.sim.metrics()

            # ----- reward shaping -----
            # 1) Throughput artışı (slot bazında gönderilen bitler)
            sent_bits_total = sum(self.sim.sent_bits.values())
            sent_bits_delta = max(0, sent_bits_total - self.last_sent_bits_total)
            tp_term = sent_bits_delta / float(self.total_rb * 5000.0)  # ~normalize (gain≈5, 1000bit/RB)

            # 2) Deadline ihlali artışı (ceza)
            viol_delta = new_metrics["deadline_violations"] - self.last_metrics["deadline_violations"]
            viol_term = -max(0.0, float(viol_delta))  # artış kadar eksi

            # 3) Ortalama gecikme artışı (ceza)
            old_delay = self.last_metrics["avg_delay_ms"] or 0.0
            new_delay = new_metrics["avg_delay_ms"] or 0.0
            delay_delta = max(0.0, new_delay - old_delay)
            delay_term = -(delay_delta / 50.0)  # +50ms ≈ -1

            # 4) Kanal kazancı kullanan tahsis (bonus)
            if self.total_rb > 0:
                total_alloc = sum(a for _, a in alloc_list) if alloc_list else 1.0
                gain_used = 0.0
                for uid2, a2 in alloc_list:
                    gain_used += self.last_gains.get(uid2, 1.0) * a2
                gain_term = gain_used / (self.total_rb * 5.0)  # normalize
            else:
                gain_term = 0.0

            # 5) Boş hedef cezası (slot içinde kaç kez boşa atıldı)
            idle_term = -0.2 * float(self.idle_in_slot)

            # Ağırlıklı toplam
            reward = (1.0 * tp_term) + (2.0 * gain_term) + (1.5 * viol_term) + (0.2 * delay_term) + idle_term

            # Durum güncelle
            self.last_metrics = new_metrics
            self.last_sent_bits_total = sent_bits_total
            self.curr_slot += 1
            self.rb_used_in_slot = 0
            self.pending_alloc = {u.id: 0 for u in self.users}
            self.idle_in_slot = 0

            # yeni slot: trafik + kazançlar
            if self.curr_slot < self.slots_total:
                self._spawn_packets_for_slot(self.curr_slot)
                self.last_gains = self.channel.gains(self.users)

        if self.curr_slot >= self.slots_total:
            terminated = True

        obs = self._features()
        return obs, float(reward), terminated, truncated, info

    def render(self):
        return
