## Installation
brainsets is available for Python 3.10+

To install the package, run the following command:
```bash
pip install brainsets
```

## List of available brainsets

| brainset_id | Documentation | Raw Data Size | Processed Data Size |
|-------------|---------------|---------------|--------------------|
| churchland_shenoy_neural_2012 | [Link](https://brainsets.readthedocs.io/en/latest/_generated/brainsets.datasets.ChurchlandShenoyNeural2012.html) | 46 GB | 25 GB |
| flint_slutzky_accurate_2012 | [Link](https://brainsets.readthedocs.io/en/latest/_generated/brainsets.datasets.FlintSlutzkyAccurate2012.html) | 3.2 GB | 151 MB |
| odoherty_sabes_nonhuman_2017 | [Link](https://brainsets.readthedocs.io/en/latest/_generated/brainsets.datasets.OdohertySabesNonhuman2017.html) | 22 GB | 26 GB |
| pei_pandarinath_nlb_2021  | [Link](https://brainsets.readthedocs.io/en/latest/_generated/brainsets.datasets.PeiPandarinathNLB2021.html) | 688 KB | 22 MB |
| perich_miller_population_2018 | [Link](https://brainsets.readthedocs.io/en/latest/_generated/brainsets.datasets.PerichMillerPopulation2018.html) | 13 GB | 2.9 GB |
| kemp_sleep_edf_2013 | [Link](https://brainsets.readthedocs.io/en/latest/_generated/brainsets.datasets.KempSleepEDF2013.html) | 8.2 GB | 60 GB |
| neuroprobe_2025 | [Link](https://brainsets.readthedocs.io/en/latest/_generated/brainsets.datasets.Neuroprobe2025.html) | 138 GB | 257 GB |
| allen_visual_coding_ophys_2016 | [Link](https://brainsets.readthedocs.io/en/latest/_generated/brainsets.datasets.AllenVisualCodingOphys2016.html) | 356 GB | 58 GB |
| vollan_moser_alternating_2025 | [Link](https://brainsets.readthedocs.io/en/latest/_generated/brainsets.datasets.VollanMoserAlternating2025.html) | 16.4 GB | 4.5 GB |


## Acknowledgements

This work is only made possible thanks to the public release of these valuable datasets by the original researchers. If you use any of the datasets processed by brainsets in your research, please make sure to cite the appropriate original papers and follow any usage guidelines specified by the dataset creators. Proper attribution not only gives credit to the researchers who collected and shared the data but also helps promote open science practices in the neuroscience community. You can find the original papers and usage guidelines for each dataset in the [brainsets documentation](https://brainsets.readthedocs.io/en/latest/datasets/index.html).


## Using the brainsets CLI

### Configuring data directories
First, configure the directories where brainsets will store raw and processed data:
```bash
brainsets config set
```

You will be prompted to enter the paths to the raw and processed data directories.
```bash
$> brainsets config set
Enter raw data directory: ./data/raw
Enter processed data directory: ./data/processed
```

You can update the configuration at any time by running the `config set` command again.

To view the current configuration:
```bash
brainsets config show
```

### Listing available datasets
You can list the available datasets by running the `list` command:
```bash
brainsets list
```

### Preparing data
You can prepare a dataset by running the `prepare` command:
```bash
brainsets prepare <brainset>
```

Data preparation involves downloading the raw data from the source then processing it, 
following a set of rules defined in `pipelines/<brainset>/`.

For example, to prepare the Perich & Miller (2018) dataset, you can run:
```bash
brainsets prepare perich_miller_population_2018 --cores 8
```
