import math
import random
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Packet:
    size_bits: int
    arrival_t: int  # slot index
    deadline_t: Optional[int] = None

@dataclass
class User:
    id: int
    qos_delay_ms: int = 50
    weight: float = 1.0
    queue: List[Packet] = field(default_factory=list)

    def queue_empty(self):
        return len(self.queue) == 0

    def next_deadline(self):
        if self.queue:
            return self.queue[0].deadline_t
        return None

@dataclass
class Channel:
    # Basit kanal modeli: her kullanıcıya slot başı kazanç (bit/RB) üret
    seed: int = 0
    def __post_init__(self):
        self.rng = random.Random(self.seed)
    def gains(self, users):
        return {u.id: max(0.5, self.rng.gauss(2.0, 0.5)) for u in users}

class SlotSimulator:
    def __init__(self, users, channel: Channel, total_rb=10, slot_ms=1):
        self.users = users
        self.chan = channel
        self.total_rb = total_rb
        self.slot_ms = slot_ms
        self.t = 0
        self.delays = []
        self.sent_bits = {u.id: 0 for u in users}
        self.deadline_violations = 0

    def step(self, alloc):
        gains = self.chan.gains(self.users)
        # alloc: list[(user_id, rb_count)]
        for uid, rbc in alloc:
            u = next(u for u in self.users if u.id == uid)
            if u.queue_empty():
                continue
            bits_per_rb = gains[uid] * 1000  # örnek: 1000 bit/RB
            budget = int(rbc * bits_per_rb)
            # Kuyruktan paket gönder
            while budget > 0 and not u.queue_empty():
                p = u.queue[0]
                if p.size_bits <= budget:
                    budget -= p.size_bits
                    self.delays.append((self.t - p.arrival_t) * self.slot_ms)
                    self.sent_bits[uid] += p.size_bits
                    if p.deadline_t is not None and self.t > p.deadline_t:
                        self.deadline_violations += 1
                    u.queue.pop(0)
                else:
                    p.size_bits -= budget
                    self.sent_bits[uid] += budget
                    budget = 0
        self.t += 1
        return gains

    def metrics(self):
        import numpy as np

        # Ortalama gecikme
        if self.delays:
            avg_delay = float(np.mean(self.delays))
        else:
            avg_delay = math.inf

        # Eğer hiç slot işlenmediyse (t=0) sıfırla
        if self.t == 0:
            total_tp_mbps = 0.0
        else:
            total_tp_mbps = sum(self.sent_bits.values()) / (self.t * self.slot_ms) / 1e6 * 1000.0

        # Adalet (Jain Fairness Index)
        vals = list(self.sent_bits.values())
        if sum(vals) == 0:
            fairness = 0.0
        else:
            s = sum(vals)
            sq = sum(v * v for v in vals)
            fairness = (s * s) / (len(vals) * sq)

        return {
            "avg_delay_ms": round(avg_delay, 3) if avg_delay != math.inf else None,
            "throughput_mbps": round(total_tp_mbps, 6),
            "fairness": round(fairness, 4),
            "deadline_violations": self.deadline_violations,
            "slots": self.t
        }
