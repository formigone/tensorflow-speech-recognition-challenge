classes = [
    'up',       # 0
    'down',     # 1
    'left',     # 2
    'right',    # 3
    'go',       # 4
    'stop',     # 5
    'yes',      # 6
    'no',       # 7
    'on',       # 8
    'off',      # 9
    'silence',  # 10
    'unknown',  # 11
]

label_key_map = {k: v for v, k in enumerate(classes)}
key_label_map = {v: k for v, k in enumerate(classes)}


def int2label(value, default='unknown'):
    try:
        return key_label_map[value]
    except KeyError:
        return default


def label2int(value, default=11, v2=False):
    try:
        if v2 and value == 'noise':
            raise ValueError('BOOM')
        return label_key_map[value]
    except KeyError:
        return default


if __name__ == '__main__':
    print(label_key_map)
    print(key_label_map)
    print(label2int('left'))
    print(int2label(3))
