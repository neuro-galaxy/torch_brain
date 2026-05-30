# Allen Brain Observatory: Visual Behavior Neuropixels


## Files sizes:
- CSV files containing information about sessions, probes, channels and units (58.1 MB)
- NWB files containing spike times, behavior data, and stimulus information for each session (146.5 GB total, min file size = 1.7 GB, max file size = 3.3 GB)
- NWB files containing LFP data for each probe (707 GB total, min file size = 0.9 GB, max file size = 2.7 GB)

## Preparation:

### Decoding information:
    - Drifting Gratings (for orientation classification and temporal frequency classification)
- The following stimuli are extracted
    - Static Gratings (for orientation classification)
    - Gabors (for orientation and 2D position classification)
- The following Behaviors are extracted
    - Running Speed (Running speed regression)
    - Pupil Area (Pupil size regression)
    - Eye Tracking (Gaze position regression)

### Splitting
- The intervals for each session are split into train/val/test sets.
- Each interval `(start, end)` part of the the split is guaranteed to be > 1.0 seconds.
- The splitting is done in many steps due to the way the raw data is extracted from Allen SDK:
    - First split for each stimuli (Drifting Gratings, Static Gratings, Gabors). All of these stimuli are done one after the other (or interleaved). But the important thing is that they are disjoint.
    The splitting logic is dependent on the specifics of how the stimuli is presented.
    For example, drifting gratings are presenting in 2.0 second trials (which are greater than our fixed context window of 1.0 seconds), so each trial can be dropped in one of the splits (so the sampler can sample 1.0 second patches from each of these intervals in a split).
    But gabors and static gratings are presented in 0.2-0.25 second trials. Therefore their `(start, end)` cannot be directly put into one of the train/test/val as they are too small to be sampled by the sampler (whose minimum context window for sampling is 1.0 seconds). Therefore, they need to be coalesced into larger intevals and then distributed thereafter to train/test/val.
    - Note how we don't include behavior data (running speed, gaze, pupil size) in the above splitting. This is because it is not trial-based, but is instant, timestamp-based data.
    Therefore, behavior label information is spread throughout each session and overlaps with the stimulus presentation intervals. Therefore, a time-patch that is sampled by the sampler during the data-loading phase for a given stimuli patch, will have overlapping behavior data too.
    - Once we have split each of the stimuli, we then collate it overall into a bigger bag of trian/val/test splits. The behavior data as mentioned previously are sprinkled within these intervals so will be included in the data.
    - Optionally, the script also helps include the residual regions of the session that don't have the stimulis we want to gather behavior data to train.

- An example visualization of the resulting splitting is shown below:
![split_all.png](split_all.png)

