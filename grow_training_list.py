variants = ['long', 'deep', 'short', 'wn', 'roll']

with open('input_train_shuff', 'r') as input:
    with open('input_train_synth', 'w+') as output:
        i = 0
        for line in input:
            label, path = line.strip().split(' ')
            output.write('{} {} {}\n'.format('data_speech_commands_v0.01', label, path))
            for variant in variants:
                output.write('{} {} {}\n'.format('data_synth', label, path.replace('.wav', '-' + variant + '.wav')))
            i += 1
            if i % 100 == 0:
                print(i)
