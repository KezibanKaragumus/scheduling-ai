import math
from .base import BaseScheduler

class EDFScheduler(BaseScheduler):
    def __init__(self, default_rb_per_user=1):
        self.default_rb = default_rb_per_user

    def select(self, users, slot_info):
        # En küçük deadline’a sahip akışları öncele
        total_rb = slot_info.get("total_rb", 10)
        chan = slot_info.get("channel_gains", {})  # user_id -> gain

        # Kullanıcıları sıralıyoruz: deadline artan, kanal kazancı azalan
        sorted_users = sorted(users, key=lambda u: (u.next_deadline() or math.inf, -chan.get(u.id, 0.0)))
        alloc = []
        remaining = total_rb
        i = 0

        while remaining > 0 and i < len(sorted_users):
            u = sorted_users[i]
            take = min(self.default_rb, remaining)
            alloc.append((u.id, take))
            remaining -= take
            i += 1

        return alloc
