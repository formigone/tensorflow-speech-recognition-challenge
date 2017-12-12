import argparse
from util import labels

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, help='CSV file to check classes')

FLAGS, unused = parser.parse_known_args()

print('Parsing {}'.format(FLAGS.file))
total = 0
with open(FLAGS.file, 'r') as fh:
    classes = {}
    for line in fh:
        key, file, label = line.split(' ')
        total += 1
        if key in classes:
            classes[key] += 1
        else:
            classes[key] = 1

print(classes)
for k in sorted(classes.keys()):
    print('{} {} ({}%)'.format(k, classes[k], int(classes[k] / total * 100)))
