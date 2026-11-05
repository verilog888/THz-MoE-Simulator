def contributions(model, owner, experts):
    """Logical expert IDs; shared IDs follow routed IDs."""
    result = {}
    for e in experts:
        result.setdefault(int(e) % 16, []).append(int(e))
    if model['shared_experts']:
        for branch in range(2):
            die = owner if owner % 2 == branch else owner ^ 1
            result.setdefault(die, []).append(model['routed_experts'] + branch)
    return result
