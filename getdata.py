import json
import tensorflow as tf
import numpy as np

sr = 22050
hop_size = 2048

def get_cqt(name):
    data = []
    count = 0
    for path in name:
        count += 1
        id = path.split("\\")[-2]
        with open(path, newline='') as json_file:
            file = json.load(json_file)
            data.append({"id": id, "cqt": file["CQT"]})
        print(count)
    return data

def get_label(name):
    data = []
    count = 0
    for path in name:
        count += 1
        id = path.split("\\")[-2]
        f = open(path, "r")
        text = f.read().split('\n')
        label_list = truth_to_list(text)
        data.append({"id": id, "label": label_list})
        f.close()
    return data

def truth_to_list(text):
    sec_to_size = 1 / sr * hop_size
    size = int(float(text[-2].split("	")[1]) / sec_to_size)
    array = np.zeros(size, dtype=int)
    for i in text:
        if i == '':
            break
        j = i.split("	")
        head = float(j[0])/sec_to_size
        tail = float(j[1])/sec_to_size
        for k in range(int(head), int(tail)):
            array[k] = int(mylabel[label_convert(j[2])])
    return array



def create_tf_dataset(cqt, label, bs = 4):
    ds = [_["cqt"] for _ in cqt]
    time_range = 10 / (1 / sr * hop_size)
    gap = 50
    new_ds = []
    new_labelds = []
    for i in range(len(ds)):
        start = 0;
        end = int(time_range);
        maxlen = len(ds[i][0])
        while end < maxlen:
            cqt_split = []
            label_split = split_label(label[i]["label"], start, end)
            for j in ds[i]:
                cqt_split.append(j[start:end])
            cqt_split = np.array(cqt_split)
            cqt_split = np.reshape(cqt_split, (107, 84), order='F')
            #print(cqt_split.shape)
            if label_split.shape[0] == 25:
                new_ds.append(cqt_split)
                new_labelds.append(label_split)
            start += gap
            end += gap
    new_labelds = tf.data.Dataset.from_tensor_slices(new_labelds)
    new_ds = tf.data.Dataset.from_tensor_slices(new_ds)
    #new_ds = new_ds.map(process, num_parallel_calls= tf.data.AUTOTUNE)
    ds = tf.data.Dataset.zip((new_ds, new_labelds))
    ds = ds.map(lambda x, y: {"source": x, "target": y})
    print("------")
    tf.print(ds)
    ds = ds.shuffle(500)
    ds = ds.batch(bs)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def process(data):
    print("before ", data)
    x = tf.math.log(tf.abs(data))
    # normalization
    means = tf.math.reduce_mean(x, 1, keepdims=True)
    stddevs = tf.math.reduce_std(x, 1, keepdims=True)
    x = (x - means) / stddevs
    print("after", x)
    return x

def split_label(label,start, end):
    output = np.append([26], label[start:end])
    output = np.append(output, [27])
    target = []
    last = 0
    for i in output:
        if i != last:
            last = i
            target.append(i)
    output = np.array(target)
    if output.shape[0] < 5:
        return output
    pad = 25 - output.shape[0]
    output = np.append(output, [0]*pad)
    return output


mylabel = {
    "": 0,
    "C:maj": 1,
    "C#:maj": 2,
    "D:maj": 3,
    "D#:maj": 4,
    "E:maj": 5,
    "F:maj": 6,
    "F#:maj": 7,
    "G:maj": 8,
    "G#:maj": 9,
    "A:maj": 10,
    "A#:maj": 11,
    "B:maj": 12,
    "C:min": 13,
    "C#:min": 14,
    "D:min": 15,
    "D#:min": 16,
    "E:min": 17,
    "F:min": 18,
    "F#:min": 19,
    "G:min": 20,
    "G#:min": 21,
    "A:min": 22,
    "A#:min": 23,
    "B:min": 24,
    "N": 25,
    "<": 26,
    ">": 27
}

def label_convert(origin_label):
    new_label = "N"
    if (origin_label[0:2] == "Db") | (origin_label[0:2] == "C#"):
        new_label = "C#:maj"
    if (origin_label[0:2] == "Eb") | (origin_label[0:2] == "D#"):
        new_label = "D#:maj"
    if (origin_label[0:2] == "Gb") | (origin_label[0:2] == "F#"):
        new_label = "F#:maj"
    if (origin_label[0:2] == "Ab") | (origin_label[0:2] == "G#"):
        new_label = "G#:maj"
    if (origin_label[0:2] == "Bb") | (origin_label[0:2] == "A#"):
        new_label = "A#:maj"
    if ((origin_label[0:2] == "Db") | (origin_label[0:2] == "C#")) & (origin_label[3:6] == "min"):
        new_label = "C#:min"
    if ((origin_label[0:2] == "Eb") | (origin_label[0:2] == "D#")) & (origin_label[3:6] == "min"):
        new_label = "D#:min"
    if ((origin_label[0:2] == "Gb") | (origin_label[0:2] == "F#")) & (origin_label[3:6] == "min"):
        new_label = "F#:min"
    if ((origin_label[0:2] == "Ab") | (origin_label[0:2] == "G#")) & (origin_label[3:6] == "min"):
        new_label = "G#:min"
    if ((origin_label[0:2] == "Bb") | (origin_label[0:2] == "A#")) & (origin_label[3:6] == "min"):
        new_label = "A#:min"
    if origin_label[0:2] == "C:":
        new_label = "C:maj"
    if origin_label[0:2] == "D:":
        new_label = "D:maj"
    if origin_label[0:2] == "E:":
        new_label = "E:maj"
    if origin_label[0:2] == "F:":
        new_label = "F:maj"
    if origin_label[0:2] == "G:":
        new_label = "G:maj"
    if origin_label[0:2] == "A:":
        new_label = "A:maj"
    if origin_label[0:2] == "B:":
        new_label = "B:maj"
    if (origin_label[0:2] == "C:") & (origin_label[2:5] == "min"):
        new_label = "C:min"
    if (origin_label[0:2] == "D:") & (origin_label[2:5] == "min"):
        new_label = "D:min"
    if (origin_label[0:2] == "E:") & (origin_label[2:5] == "min"):
        new_label = "E:min"
    if (origin_label[0:2] == "F:") & (origin_label[2:5] == "min"):
        new_label = "F:min"
    if (origin_label[0:2] == "G:") & (origin_label[2:5] == "min"):
        new_label = "G:min"
    if (origin_label[0:2] == "A:") & (origin_label[2:5] == "min"):
        new_label = "A:min"
    if (origin_label[0:2] == "B:") & (origin_label[2:5] == "min"):
        new_label = "B:min"
    return new_label

def get_chord_vocab():
    vocab = ("" ,"C:maj",
    "C#:maj",
    "D:maj",
    "D#:maj",
    "E:maj",
    "F:maj",
    "F#:maj",
    "G:maj",
    "G#:maj",
    "A:maj",
    "A#:maj",
    "B:maj",
    "C:min",
    "C#:min",
    "D:min",
    "D#:min",
    "E:min",
    "F:min",
    "F#:min",
    "G:min",
    "G#:min",
    "A:min",
    "A#:min",
    "B:min",
    "N",
    "<",
    ">" )
    return vocab



