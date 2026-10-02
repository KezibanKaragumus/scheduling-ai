def qos_satisfaction(delays_ms, qos_thr_ms=50):
    """
    Gecikmesi verilen QoS eşiğinin altında olan paketlerin oranını hesaplar.
    Örnek: 50 ms altında kalanların yüzdesi.
    """
    if not delays_ms:
        return 0.0
    ok = sum(1 for d in delays_ms if d <= qos_thr_ms)
    return ok / len(delays_ms)
