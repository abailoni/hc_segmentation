import sys
sys.path.append("/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/")
import vigra
import os
from os.path import join
import json
import csv
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=int)

    args = parser.parse_args()

    if args.model == 1:
        project_folder = '/export/home/abailoni/learnedHC/plain_unstruct/pureDICE_wholeTrainingSet'
        name = 'unstruct'
    elif args.model == 2:
        project_folder = '/export/home/abailoni/learnedHC/model_090_v2/pureDICE_wholeDtSet'
        name = 'CNN-2D'
    elif args.model == 3:
        project_folder = '/export/home/abailoni/learnedHC/model_050_A_v3/pureDICE_wholeDtSet'
        name = 'CNN-3D'
    else:
        raise NotImplementedError("Entered model-ID was not recognized")

    post_proc_dir = join(project_folder, 'postprocess')

    scores_collected = {}

    csv_list = []

    for _, subdirList, _ in os.walk(post_proc_dir):
        for model in subdirList:
            model_path = join(post_proc_dir, model)
            eval_file = os.path.join(model_path, 'scores.json')
            if os.path.exists(eval_file):
                with open(eval_file, 'r') as f:
                    scores = json.load(f)
            else:
                continue

            separator = '_'
            splitted_string = model.split(separator)
            inferName = splitted_string[1]
            sample = splitted_string[-1]
            aggl_name = separator.join(splitted_string[2:-1])


            csv_list.append({"Infer. name": inferName,
                             "Aggl. name": aggl_name,
                             "Sample": sample})

            if inferName not in scores_collected:
                scores_collected[inferName] = {}
            if aggl_name not in scores_collected[inferName]:
                scores_collected[inferName][aggl_name] = {}

            assert sample not in scores_collected[inferName][aggl_name], "Double scores found"
            scores_collected[inferName][aggl_name][sample] = {}
            for key in scores[sample]:
                if key == "adapted-rand":
                    scores_collected[inferName][aggl_name][sample]['ARAND'] = scores[sample][key]
                    csv_list[-1]['ARAND'] = scores[sample][key]
                elif key == "cremi-score":
                    scores_collected[inferName][aggl_name][sample]['CS'] = scores[sample][key]
                    csv_list[-1]['CS'] = scores[sample][key]
                elif key == "vi-merge":
                    scores_collected[inferName][aggl_name][sample]['VIm'] = scores[sample][key]
                    csv_list[-1]['VIm'] = scores[sample][key]
                elif key == "vi-split":
                    scores_collected[inferName][aggl_name][sample]['VIs'] = scores[sample][key]
                    csv_list[-1]['VIs'] = scores[sample][key]

    # Save dictionary:
    with open(join(post_proc_dir, 'collected_scores_{}.json'.format(name)), 'w+') as f:
        json.dump(scores_collected, f, indent=4, sort_keys=True)


    # Save csv file with:
    # infer_name; aggl_name; sample; scores...
    keys = [k for k in csv_list[0].keys()]
    with open(join(post_proc_dir, 'collected_scores_{}.csv'.format(name)), 'w+') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(csv_list)


