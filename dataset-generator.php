<?php

if ($argc === 1) {
    echo 'Usage: ', $argv[0], ' <label_to_include, ...>', PHP_EOL;
    exit();
}

const BASE_DIR = __DIR__ . '/data_speech_commands_v0.01/';

$train = [];
$test = [];
$validation = [];

foreach ($argv as $i => $label) {
    if ($i === 0) {
        continue;
    }
    $dir = array_filter(scandir(BASE_DIR . $label), function ($filename) {
        return preg_match('|\.wav$|', $filename);
    });

    $dir = array_map(function ($filename) use ($label) {
        return $label . ' ' . $filename;
    }, $dir);

    shuffle($dir);
    $size = count($dir);
    $train = array_merge($train, array_splice($dir, 0, $size * .8));
    $test = array_merge($test, array_splice($dir, 0, $size * .1));
    $validation = array_merge($validation, $dir);
}


shuffle($train);
shuffle($test);
shuffle($validation);

file_put_contents(__DIR__ . '/training-set.text', implode(PHP_EOL, $train));
file_put_contents(__DIR__ . '/testing-set.text', implode(PHP_EOL, $test));
file_put_contents(__DIR__ . '/validation-set.text', implode(PHP_EOL, $validation));
