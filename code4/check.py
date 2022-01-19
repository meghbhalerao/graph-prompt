import os
import re
import sys
fpattern = sys.argv[1]
lpattern = sys.argv[2]
files = os.listdir()

files = [i for i in files if re.search(fpattern, i)]
print('files: ', files)

from collections import defaultdict
ret = defaultdict(list)

for f in files:
 print('file=', f)
 #os.system('cat {} | grep {}'.format(f, lpattern))
 for i in open(f):
  if re.search(lpattern, i):
   print(i.strip())
   ret[f].append(i.strip())
 print()

print('='*20)
print(dict(ret))
