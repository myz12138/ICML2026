
from difflib import SequenceMatcher

def str_sim(a, b):
    return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()

def has_token_overlap(a, b):
    ta = [t for t in (a or "").lower().split() if t]
    tb = [t for t in (b or "").lower().split() if t]
    return len(set(ta) & set(tb)) > 0
