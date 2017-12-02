classes = [
    'up',      # 0
    'down',    # 1
    'left',    # 2
    'right',   # 3
    'go',      # 4
    'stop',    # 5
    'yes',     # 6
    'no',      # 7
    'on',      # 8
    'off',     # 9
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
