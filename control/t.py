import re

response = 'pitch:-25;roll:-143;yaw:118;vgx:0;vgy:0;vgz:-2;templ:64;temph:67;tof:290;h:0;bat:100;baro:51.48;time:0;agx:-756.00;agy:667.00;agz:898.00;'

match_obj = re.search('tof:',response)
print(match_obj+1)