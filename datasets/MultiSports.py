# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO: Address all TODOs and remove all explanatory comments
"""MultiSports: A Multi-Person Video Dataset of Spatio-Temporally Localized Sports Actions"""


import json
import os

import datasets


_CITATION = """\
@InProceedings{Li_2021_ICCV,
    author    = {Li, Yixuan and Chen, Lei and He, Runyu and Wang, Zhenzhi and Wu, Gangshan and Wang, Limin},
    title     = {MultiSports: A Multi-Person Video Dataset of Spatio-Temporally Localized Sports Actions},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {13536-13545}
}
"""

_DESCRIPTION = """\
This is a multi-person video dataset of spatio-temporally localized sports actions. Please refer to the github repo for evaluation.
"""

_HOMEPAGE = "https://deeperaction.github.io/datasets/multisports.html"

_LICENSE = "CC BY-NC 4.0"

_SPORT_LIST = ["aerobic_gymnastics", "basketball", "football", "volleyball"]

_VIDEO_URLS = {
    "trainval": [
        "https://huggingface.co/datasets/MCG-NJU/MultiSports/resolve/main/data/trainval/{}.tar".format(sport) for sport in _SPORT_LIST
    ],
    "test": [
        "https://huggingface.co/datasets/MCG-NJU/MultiSports/resolve/main/data/test/{}.tar".format(sport) for sport in _SPORT_LIST
    ]
}

_META_URLS = {
    "trainval": ["https://huggingface.co/datasets/MCG-NJU/MultiSports/resolve/main/data/trainval/multisports_GT.pkl"],
    "test": [
        "https://huggingface.co/datasets/MCG-NJU/MultiSports/resolve/main/data/test/multisports_half_test.pkl",
        "https://huggingface.co/datasets/MCG-NJU/MultiSports/resolve/main/data/test/multisports_test.pkl",
    ]
}


# Name of the dataset usually matches the script name with CamelCase instead of snake_case
class MultiSports(datasets.GeneratorBasedBuilder):
    """MultiSports: A Multi-Person Video Dataset of Spatio-Temporally Localized Sports Actions"""

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="trainval", version=VERSION, description="Data for training and validation"),
        datasets.BuilderConfig(name="test", version=VERSION, description="Data for testing (gt will not be provided)"),
    ]

    DEFAULT_CONFIG_NAME = "trainval"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        features = datasets.Features(
            {
                "frame": datasets.Image(),
                "annotations": dict,
            }
        )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        video_urls = _VIDEO_URLS[self.config.name]
        meta_urls = _META_URLS[self.config.name]
        video_dir = dl_manager.download_and_extract(video_urls)
        meta_dir = dl_manager.download_and_extract(meta_urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "videopath": video_dir,
                    "metapath": os.path.join(meta_dir, "multisports_GT.pkl"),
                    "split": "train",
                },
            ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.VALIDATION,
            #     # These kwargs will be passed to _generate_examples
            #     gen_kwargs={
            #         "filepath": os.path.join(data_dir, "dev.jsonl"),
            #         "split": "dev",
            #     },
            # ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "videopath": video_dir,
                    "metapath": os.path.join(meta_dir, "multisports_test.pkl"),
                    "split": "test"
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, videopath, metapath, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        # with open(filepath, encoding="utf-8") as f:
        #     for key, row in enumerate(f):
        #         data = json.loads(row)
        #         if self.config.name == "first_domain":
        #             # Yields examples as (key, example) tuples
        #             yield key, {
        #                 "sentence": data["sentence"],
        #                 "option1": data["option1"],
        #                 "answer": "" if split == "test" else data["answer"],
        #             }
        #         else:
        #             yield key, {
        #                 "sentence": data["sentence"],
        #                 "option2": data["option2"],
        #                 "second_domain_answer": "" if split == "test" else data["second_domain_answer"],
        #             }
        pass