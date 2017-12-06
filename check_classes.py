import argparse
from util import labels

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, help='CSV file to check classes')

FLAGS, unused = parser.parse_known_args()

print('Parsing {}'.format(FLAGS.file))
with open(FLAGS.file, 'r') as fh:
    classes = {}
    first_line = True
    for line in fh:
        if first_line:
            first_line = False
            continue
        cls = line[-5:].split(',')
        key = cls[-1].strip()
        if key in classes:
            classes[key] += 1
        else:
            classes[key] = 1

print(classes)
for k in sorted(classes.keys()):
    print(labels.int2label(int(k)).ljust(8) + ' = ' + str(classes[k]))
