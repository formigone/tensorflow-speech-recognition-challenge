classes = [
    'up',
    'down',
    'left',
    'right',
    'go',
    'stop',
    'yes',
    'no',
    'on',
    'off',
]

label_key_map = {k: v for v, k in enumerate(classes)}
key_label_map = {v: k for v, k in enumerate(classes)}


def int2label(value):
    return key_label_map[value]


def label2int(value):
    return label_key_map[value]


if __name__ == '__main__':
    print(label_key_map)
    print(key_label_map)
    print(label2int('left'))
    print(int2label(3))
