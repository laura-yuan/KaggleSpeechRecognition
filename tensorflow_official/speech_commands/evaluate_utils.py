def record_wrong_predicted_file(save_path, wav_path, wav_prediction, wav_ground_truth=None, write_label_flag = True, append_flag = True):
    # with word index, as well as label number? not sure.
    # only the wrong file would be writedown...
    import csv
    if append_flag:
        f = open(save_path, 'a', newline='')
    else:
        f = open(save_path,'w', newline='')
    writer = csv.writer(f, delimiter=',')
    if write_label_flag:
        if not append_flag:
            writer.writerow(['fname', 'prediction','label'])
        for wav, prediction, truth in zip(wav_path, wav_prediction, wav_ground_truth):
            writer.writerow([wav, prediction, truth])
    else:
        if not append_flag:
            writer.writerow(['fname', 'prediction'])
        for wav, prediction in zip(wav_path, wav_prediction):
            writer.writerow([wav, prediction])
    f.close()
def eliminate_underscore(word):
    if word == '_unknown_':
        return  'unknown'
    elif word == '_silence_':
        return  'silence'
    else:
        return word