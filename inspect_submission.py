from __future__ import division
import sys
import os


def stats(filename):
  with open(filename, 'r') as fh:
    i = 0
    group = {}
    for line in fh:
      if i == 0:
        i += 1
        continue
      key = line.split(',')[-1].strip()
      if not key in group:
        group[key] = 0
      group[key] += 1
      i += 1

  for k, v in group.iteritems():
    group[k] = round(v / i * 100, 2)
  return group


if __name__ == '__main__':
  if len(sys.argv) < 2:
    raise ValueError('No submission file(s) provided.')

  files = sys.argv[1:]

  if len(files) == 1 and os.path.isdir(files[0]):
    root = files[0]
    files = ['{}/{}'.format(root, file) for file in os.listdir(files[0])]

  for file in files:
    print(file)
    print(stats(file))
    print('---')
