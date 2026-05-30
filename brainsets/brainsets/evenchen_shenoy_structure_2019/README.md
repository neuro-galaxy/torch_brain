### Dataset description

Extracellular physiology data of monkeys implanted with 96 channel Blackrock Utah arrays in the Motor cortex and Premotor cortex. Each file contains a sessions worth of data for one monkey (total 2) performing 1 of the four cursor movement task designs. The data contains hand, eye and cursor position data, LFP, sorted spikes and other task related trialized data.

### Downloading the data

The [data](https://dandiarchive.org/dandiset/000121/0.220124.2156) can be downloaded from the DANDI archive using the following command:
```bash
mkdir raw/ && cd $_
dandi download DANDI:000121/0.220124.2156
cd -
```

### Processing the data
```bash
python3 prepare_data.py
```
