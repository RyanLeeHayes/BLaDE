#! /usr/bin/env python

import sys

tstring=sys.argv[1]
tstring=tstring.strip()
if (tstring[-1]!='s'):
  quit()
tstring=tstring[:-1]
s=float(tstring.split('m')[-1])
if len(tstring.split('m'))>1:
  tstring=tstring.split('m')[0]
else:
  tstring='0'
m=float(tstring.split('h')[-1])
if len(tstring.split('h'))>1:
  tstring=tstring.split('h')[0]
else:
  tstring='0'
h=float(tstring)

print(60*h+m+s/60)
