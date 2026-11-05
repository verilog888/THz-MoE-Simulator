"""Independent energy reconstruction from event counters; no timing reruns."""
import math

def energy(ledger,c):
    h,m,e,co=c['hardware'],c['memory'],c['energy'],c['compute']
    seconds=ledger['duration_ps']*1e-12
    capacity=h['edram_bytes']*8*h['vdies']
    tx=(ledger['tx_ps']+ledger['control_tx_ps'])*1e-12
    rx=(ledger['rx_ps']+ledger['control_rx_ps'])*1e-12
    parts={
        'compute':ledger['matrix_ops']*e['compute_pj_op']*1e-12*co['energy_scale'],
        'vector':ledger['vector_ops']*e['compute_pj_op']*1e-12*co['energy_scale']*co['vector_energy_multiplier'],
        'edram_read':8*ledger['read_bytes']*m['read_pj_bit']*1e-12*m['energy_scale'],
        'edram_write':8*ledger['write_bytes']*m['write_pj_bit']*1e-12*m['energy_scale'],
        'refresh':capacity*m['refresh_pj_bit']/m['refresh_period_ps']*m['energy_scale']*seconds,
        'memory_leak':capacity*m['leakage_pw_bit']*1e-12*seconds,
        'rf_lo':(h['vdies']+h['base_dies'])*e['lo_w_per_node']*seconds,
        'rf_tx':tx*e['tx_increment_w_per_group'],
        'rf_rx':rx*e['rx_increment_w_per_group'],
        'control_idle':(h['vdies']*co['extra_idle_power_w']+co['base_io_idle_power_w'])*seconds,
    }
    frontend=math.fsum(parts.values())
    parts.update(online_eq=rx*e['equalizer_extra_w_per_rx'],
                 ofdma_extra=ledger['ofdma_ps']*1e-12*e['ofdma_extra_w_per_active_group'])
    return dict(parts,frontend_proxy=frontend,
                baseband_stress=math.fsum(parts.values()))
