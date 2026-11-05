from fractions import Fraction
import math

def control_ps(c,nodes):
    ma,ra=c['mac'],c['radio']
    scale=Fraction(str(ra['rate_scale']))
    seconds=(Fraction(8*nodes*ma['queue_report_bytes'],ra['ptp_raw_bps'])/scale+
             Fraction(8*(ma['grant_header_bytes']+ma['resource_units']*ma['grant_bytes_per_ru']),ra['broadcast_raw_bps'])/scale+
             Fraction(ma['arbitration_cycles'],c['hardware']['clock_hz']))
    return max(ma['control_cycles']*5000,math.ceil(seconds*200000000)*5000)
