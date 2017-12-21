import sys

files = sys.argv[1:]


def get_max_vote(l0, l1, l2, l3, l4, l5):
  l0 = l0.strip().split(',')
  l1 = l1.strip().split(',')
  l2 = l2.strip().split(',')
  l3 = l3.strip().split(',')
  l4 = l4.strip().split(',')
  l5 = l5.strip().split(',')

  keys = {}
  for l in [l0, l1, l2, l3, l4, l5]:
    if l[1] not in keys:
      keys[l[1]] = 0
    keys[l[1]] += 1
  # print(keys.items())
  # print(reversed(keys.items()))
  # return 'a', 'b'
  k = [row for row in keys.items()]
  k = reversed(k)
  label = max(k, key=lambda x: x[1])[0]
  # print('{}: {}, {}, {}, {}, {} => {}'.format(l0[0], l0[1], l1[1], l2[1], l3[1], l4[1], label))
  return l0[0], label


i = 0
with open(files[0], 'r') as f0, open(files[1], 'r') as f1, open(files[2], 'r') as f2, open(files[3], 'r') as f3, open(files[4], 'r') as f4, open(files[5], 'r') as f5:
  for l0, l1, l2, l3, l4, l5 in zip(f0, f1, f2, f3, f4, f5):
    if i == 0:
      print(l0.strip())
      i += 1
      continue
    filename, label = get_max_vote(l0, l1, l2, l3, l4, l5)
    print('{},{}'.format(filename, label))
    # if i > 15:
    #   break
    # i += 1
