from .base import BaseScheduler

class WFQScheduler(BaseScheduler):
    def __init__(self, weights=None):
        # weights: user_id -> weight, yoksa hepsi 1
        self.weights = weights or {}

    def select(self, users, slot_info):
        total_rb = slot_info.get("total_rb", 10)
        active = [u for u in users if not u.queue_empty()]
        if not active:
            return []

        # Ağırlıkları topla
        wsum = sum(self.weights.get(u.id, 1.0) for u in active)

        alloc = []
        remaining = total_rb

        for i, u in enumerate(active):
            if i < len(active) - 1:
                share = int(total_rb * (self.weights.get(u.id, 1.0) / wsum))
                share = max(share, 1)
            else:
                share = remaining

            alloc.append((u.id, max(0, share)))
            remaining -= share

            if remaining <= 0:
                break

        return alloc
