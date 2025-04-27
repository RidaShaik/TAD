---
annotations_creators:
- crowdsourced
language:
- en
language_creators:
- expert-generated
license:
- cc-by-nc-4.0
multilinguality:
- monolingual
pretty_name: MultiSports
size_categories: []
source_datasets:
- original
tags:
- video
- action detection
- spatial-temporal action localization
task_categories:
- image-classification
- object-detection
- other
task_ids:
- multi-class-image-classification
extra_gated_heading: "Acknowledge license to accept the repository"
extra_gated_prompt: "This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License"
extra_gated_fields:
 I agree to use this dataset for non-commerical use ONLY: checkbox

---

# Dataset Card for MultiSports

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage:** https://deeperaction.github.io/datasets/multisports.html
- **Repository:** https://github.com/MCG-NJU/MultiSports
- **Paper:** https://arxiv.org/abs/2105.07404
- **Leaderboard:** https://paperswithcode.com/dataset/multisports
- **Point of Contact:** mailto: runyu_he@smail.nju.edu.cn

### Dataset Summary

Spatio-temporal action localization is an important and challenging problem in video understanding. Previous action detection benchmarks are limited in aspects of small numbers of instances in a trimmed video or low-level atomic actions. MultiSports is a multi-person dataset of spatio-temporal localized sports actions. Please refer to [this paper](https://arxiv.org/abs/2105.07404) for more details. Please refer to [this repository](https://github.com/MCG-NJU/MultiSports) for evaluation.

### Supported Tasks and Leaderboards

- `Spatial-temporal action localization`

Details about evaluation can be found in the [GitHub Repository](https://github.com/mcG-NJU/MultiSports). Previous challenge results can be found in [this page](https://deeperaction.github.io/results/index.html) and [this CodaLab challenge](https://codalab.lisn.upsaclay.fr/competitions/3736).

### Languages

The class labels in the dataset are in English.

## Dataset Structure

### Data Instances

Demo is available on [dataset homepage](https://deeperaction.github.io/datasets/multisports.html). 


The dataset contains ```rawframes.tar``` and ```multisports_GT.pkl```. The GT pkl file is a dictionary with the following structure:

```
{
    'labels': ['label1', 'label2', ...],
    'train_videos': [['train_vid_1', 'train_vid_2', ...]],
    'test_videos': [['test_vid_1', 'test_vid_2', ...]],
    'nframes': {
        'vid_1': nframes_1,
        'vid_2': nframes_2,
        ...
    },
    'resolution': {
        'vid_1': resolution_1,
        'vid_2': resolution_2,
        ...
    },
    'gttubes': {
        'vid_1': {
            'label_1': [tube_1, tube_2, ...],
            'label_2': [tube_1, tube_2, ...],
            ...
        }
        ...
    }
}
```

Here a ```tube``` is a ```numpy.ndarray``` with ```nframes``` rows and 5 columns ```<frame number> <x1> <y1> <x2> <y2>```.

### Data Fields

Raw frames are organized according to their sport category. The pickle file of GT contains the following fields.

- labels: list of labels

- train_videos: a list with one split element containing the list of training videos

- test_videos: a list with one split element containing the list of validation videos

- nframes: dictionary that gives the number of frames for each video

- resolution: dictionary that output a tuple ```(h,w)``` of the resolution for each video

- gttubes: dictionary that contains the gt tubes for each video. Gt tubes are dictionary that associates from each index of label, a list of tubes. A ```tube``` is a ```numpy.ndarray``` with ```nframes``` rows and 5 columns ```<frame number> <x1> <y1> <x2> <y2>```.

Please note that the label index starts from 0 and the frame index starts from 1. For the label index ```i```, the label name is ```labels[i]```.

<details>
  <summary>
  Click here to see the full list of MultiSports class labels mapping:
  </summary>

  |id|Class|
  |--|-----|
  | 0 | aerobic push up |
  | 1 | aerobic explosive push up |
  | 2 | aerobic explosive support |
  | 3 | aerobic leg circle |
  | 4 | aerobic helicopter |
  | 5 | aerobic support |
  | 6 | aerobic v support |
  | 7 | aerobic horizontal support |
  | 8 | aerobic straight jump |
  | 9 | aerobic illusion |
  | 10 | aerobic bent leg(s) jump |
  | 11 | aerobic pike jump |
  | 12 | aerobic straddle jump |
  | 13 | aerobic split jump |
  | 14 | aerobic scissors leap |
  | 15 | aerobic kick jump |
  | 16 | aerobic off axis jump |
  | 17 | aerobic butterfly jump |
  | 18 | aerobic split |
  | 19 | aerobic turn |
  | 20 | aerobic balance turn |
  | 21 | volleyball serve |
  | 22 | volleyball block |
  | 23 | volleyball first pass |
  | 24 | volleyball defend |
  | 25 | volleyball protect |
  | 26 | volleyball second pass |
  | 27 | volleyball adjust |
  | 28 | volleyball save |
  | 29 | volleyball second attack |
  | 30 | volleyball spike |
  | 31 | volleyball dink |
  | 32 | volleyball no offensive attack |
  | 33 | football shoot |
  | 34 | football long pass |
  | 35 | football short pass |
  | 36 | football through pass |
  | 37 | football cross |
  | 38 | football dribble |
  | 39 | football trap |
  | 40 | football throw |
  | 41 | football diving |
  | 42 | football tackle |
  | 43 | football steal |
  | 44 | football clearance |
  | 45 | football block |
  | 46 | football press |
  | 47 | football aerial duels |
  | 48 | basketball pass |
  | 49 | basketball drive |
  | 50 | basketball dribble |
  | 51 | basketball 3-point shot |
  | 52 | basketball 2-point shot |
  | 53 | basketball free throw |
  | 54 | basketball block |
  | 55 | basketball offensive rebound |
  | 56 | basketball defensive rebound |
  | 57 | basketball pass steal |
  | 58 | basketball dribble steal |
  | 59 | basketball interfere shot |
  | 60 | basketball pick-and-roll defensive |
  | 61 | basketball sag |
  | 62 | basketball screen |
  | 63 | basketball pass-inbound |
  | 64 | basketball save |
  | 65 | basketball jump ball |


</details>

### Data Splits

|             |train  |validation| test  |
|-------------|------:|---------:|------:|
|# of tubes   |28514  |10116     | -     |

*GT for test split is not provided. Please wait for the new competition to start. Information will be updated in [dataset homepage](https://deeperaction.github.io/datasets/multisports.html).*

## Dataset Creation

### Curation Rationale

Spatio-temporal action detection is an important and challenging problem in video understanding. Previous action detection benchmarks are limited in aspects of small numbers of instances in a trimmed video or low-level atomic actions.

### Source Data

#### Initial Data Collection and Normalization

> After choosing the four sports, we search for their competition videos by querying the name of sports like volleyball and the name of competition levels like Olympics and World Cup on YouTube, and then down- load videos from top search results. For each video, we only select high-resolution, e.g. 720P or 1080P, competition records and then manually cut them into clips of minutes, with less shot changes in each clip and to be more suitable for action detection.

#### Who are the source language producers?

The annotators of action categories and temporal boundaries are professional athletes of the corresponding sports. Please refer to [the paper](https://arxiv.org/abs/2105.07404) for more information.

### Annotations

#### Annotation process

1. (FIRST STAGE) A team of professional athletes generate records of the action la- bel, the starting and ending frame, and the person box in the starting frame, which can ensure the efficiency, accu- racy and consistency of our annotation results.

2. At least one annotator with domain knowledge double-check the annotations, correct wrong or inaccurate ones and also add missing annotations

3. (SECOND STAGE) With the help of FCOT tracking algorithm, a team of crowd-sourced annotators adjust bounding boxes of tracking results at each frame for each record.

4. Double-check each instance by playing it in 5fps and manually correct the inaccurate bounding boxes.

#### Who are the annotators?

For the first stage, annotators are professional athletes. For the second stage, annotators are common volunteers.

### Personal and Sensitive Information

[More Information Needed]

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed]

### Discussion of Biases

[More Information Needed]

### Other Known Limitations

[More Information Needed]

## Additional Information

### Dataset Curators

Authors of [this paper](https://arxiv.org/abs/2105.07404)

- Yixuan Li

- Lei Chen

- Runyu He

- Zhenzhi Wang

- Gangshan Wu

- Limin Wang

### Licensing Information

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

### Citation Information

If you find this dataset useful, please cite as

```
@InProceedings{Li_2021_ICCV,
    author    = {Li, Yixuan and Chen, Lei and He, Runyu and Wang, Zhenzhi and Wu, Gangshan and Wang, Limin},
    title     = {MultiSports: A Multi-Person Video Dataset of Spatio-Temporally Localized Sports Actions},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {13536-13545}
}
```

### Contributions

Thanks to [@Judie1999](https://github.com/Judie1999) for adding this dataset.