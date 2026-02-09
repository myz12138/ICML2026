
def neff_from_logits(logits):
    if not logits:
        return float("inf")
    xs = [float(x) for x in logits]
    mn = min(xs)
    if mn <= 0.0:
        ws = [x - mn + 1e-6 for x in xs]
    else:
        ws = xs
    z = sum(ws)
    if z <= 0.0:
        return float("inf")
    ps = [w / z for w in ws]
    return 1.0 / sum(p * p for p in ps)

def is_resolved(neff_value, threshold):
    try:
        return float(neff_value) <= float(threshold)
    except Exception:
        return False
