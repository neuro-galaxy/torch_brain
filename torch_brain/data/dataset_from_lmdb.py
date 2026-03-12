"""
DatasetFromLmdb: A Dataset implementation that reads directly from LMDB files
and provides POYO-compatible Data objects.

This module provides an alternative to the HDF5-based Dataset class, allowing
direct integration of LMDB data (commonly used in DIVER) with the POYO pipeline.
"""

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import bisect

import lmdb
import numpy as np
import pickle
import torch
from temporaldata import Data, Interval, RegularTimeSeries, IrregularTimeSeries, ArrayDict

from .dataset import DatasetIndex


logger = logging.getLogger(__name__)


@dataclass
class LmdbConfig:
    """Configuration for LMDB dataset loading.
    
    Args:
        lmdb_path: Path to the LMDB database
        brainset_id: Identifier for the brainset (e.g., 'faced_from_lmdb')
        session_id: Identifier for the session
        sampling_rate: Sampling rate of the data in Hz
        patch_duration: Duration of each patch in seconds (samples / sampling_rate)
        sample_duration: Duration of each sample in seconds
        channel_key: Key in data_info for channel names (default: 'channel_names')
        xyz_key: Key in data_info for channel coordinates (default: 'xyz_id')
        multitask_readout: List of readout configurations for tasks
        sampling_intervals_modifier: Optional code to modify sampling intervals
    """
    lmdb_path: str
    brainset_id: str = "lmdb_dataset"
    session_id: str = "session_001" #TODO : what should be session in our case?
    sampling_rate: float = 500.0
    patch_duration: Optional[float] = None  # Auto-computed if None
    sample_duration: Optional[float] = None  # Auto-computed if None
    channel_key: str = "channel_names"
    xyz_key: str = "xyz_id"
    multitask_readout: List[Dict] = field(default_factory=list) #TODO : Should be handled via FINAL_TASK_DICT
    sampling_intervals_modifier: Optional[str] = None


class DatasetFromLmdb(torch.utils.data.Dataset):
    """Dataset that reads directly from LMDB and presents data in POYO-compatible format.
    
    The LMDB samples are treated as a virtual continuous recording where samples 
    are concatenated in time order within each split (train/valid/test).
    
    This class provides the same interface as the HDF5-based Dataset class, making it
    a drop-in replacement for POYO's DataModule.
    
    Args:
        lmdb_config: Configuration specifying LMDB path and data parameters.
            Can be an LmdbConfig object or a dict with the same fields.
        split: The split to use ('train', 'valid', 'test', or None for all)
        transform: Transform to apply to data samples
        unit_id_prefix_fn: Function to generate unit ID prefixes
        session_id_prefix_fn: Function to generate session ID prefixes
        subject_id_prefix_fn: Function to generate subject ID prefixes
    
    Example:
        >>> config = LmdbConfig(
        ...     lmdb_path="/path/to/lmdb",
        ...     brainset_id="faced",
        ...     session_id="session_001",
        ...     sampling_rate=500.0,
        ... )
        >>> dataset = DatasetFromLmdb(lmdb_config=config, split="train")
        >>> sample = dataset[DatasetIndex("faced/session_001", 0.0, 1.0)]
        
(Pdb) lmdb_config
LmdbConfig(lmdb_path='/global/cfs/cdirs/m4750/DIVER/PRETRAINING_DATA_LMDB/arch_search_June_2025/FACED', brainset_id='faced', session_id='faced_session', sampling_rate=500.0, patch_duration=None, sample_duration=None, channel_key='channel_names', xyz_key='xyz_id', multitask_readout=[{'readout_id': 'emotion_classification', 'num_classes': 9, 'weight': 1.0, 'metrics': [{'metric': {'_target_': 'torchmetrics.Accuracy', 'task': 'multiclass', 'num_classes': 9}}, {'metric': {'_target_': 'torchmetrics.F1Score', 'task': 'multiclass', 'num_classes': 9, 'average': 'macro'}}]}], sampling_intervals_modifier=None)

    """
    
    _check_for_data_leakage_flag: bool = True
    
    def __init__(
        self,
        lmdb_config: LmdbConfig | Dict,
        *,
        split: Optional[str] = None,
        transform: Optional[Callable[[Data], Any]] = None,
        unit_id_prefix_fn: Optional[Callable[[Data], str]] = None,
        session_id_prefix_fn: Optional[Callable[[Data], str]] = None,
        subject_id_prefix_fn: Optional[Callable[[Data], str]] = None,
    ):
        super().__init__()
        import pdb; pdb.set_trace()
        # Convert dict to LmdbConfig if needed
        if isinstance(lmdb_config, dict):
            lmdb_config = LmdbConfig(**lmdb_config)
        
        self.config = lmdb_config #(Pdb) lmdb_config
                                    #LmdbConfig(lmdb_path='/global/cfs/cdirs/m4750/DIVER/PRETRAINING_DATA_LMDB/arch_search_June_2025/FACED', brainset_id='faced', session_id='faced_session', sampling_rate=500.0, patch_duration=1.0, sample_duration=10.0, channel_key='channel_names', xyz_key='xyz_id', multitask_readout=[], sampling_intervals_modifier=None)
        self.split = split
        self.transform = transform 
        # TODO : should get transform from model. 수인쌤이랑 이야기해보기. Also possible to do 
        # TODO : POYO tokenizer.
        
        # Default prefix functions (Local ID → Global Unique ID)
        self.unit_id_prefix_fn = unit_id_prefix_fn or (lambda data: f"{data.brainset.id}/{data.session.id}/")
        self.session_id_prefix_fn = session_id_prefix_fn or (lambda data: f"{data.brainset.id}/")
        self.subject_id_prefix_fn = subject_id_prefix_fn or (lambda data: f"{data.brainset.id}/")
        
        # Recording ID follows POYO convention: brainset_id/session_id
        self.recording_id = f"{self.config.brainset_id}/{self.config.session_id}"
        
        # Open LMDB and load metadata
        self._env = lmdb.open(self.config.lmdb_path, readonly=True, lock=False)
        self._load_metadata()
        
        # Build time index for efficient slicing
        self._build_time_index()
        
        # Create recording dict for compatibility with Dataset interface
        self.recording_dict = {
            self.recording_id: {
                "filename": Path(self.config.lmdb_path),
                "config": {
                    "multitask_readout": self.config.multitask_readout,
                    "sampling_intervals_modifier": self.config.sampling_intervals_modifier,
                }
            }
        }
        
        logger.info(f"Loaded LMDB dataset from {self.config.lmdb_path}")
        logger.info(f"  Split: {split}, Samples: {len(self._current_keys)}")
        logger.info(f"  Duration: {self._total_duration:.2f}s, Channels: {self._n_channels}")
    
    def _load_metadata(self):
        """Load keys and sample metadata from LMDB."""
        with self._env.begin() as txn:
            # Load keys dictionary
            keys_data = txn.get('__keys__'.encode())
            if keys_data is None:
                raise ValueError("LMDB database missing '__keys__' entry")
            
            self._keys_dict = pickle.loads(keys_data)
            
            # Map split names (LMDB uses 'val', POYO uses 'valid')
            split_map = {'valid': 'val', 'train': 'train', 'test': 'test'}
            lmdb_split = split_map.get(self.split, self.split) if self.split else None
            
            # Get keys for current split or all splits
            if lmdb_split and lmdb_split in self._keys_dict:
                self._current_keys = self._keys_dict[lmdb_split]
            elif self.split is None:
                # Concatenate all splits
                self._current_keys = []
                for split_name in ['train', 'val', 'test']:
                    if split_name in self._keys_dict:
                        self._current_keys.extend(self._keys_dict[split_name])
            else:
                raise ValueError(f"Split '{self.split}' not found in LMDB. Available: {list(self._keys_dict.keys())}")
            
            # Load first sample to get data info
            if len(self._current_keys) > 0:
                first_sample = pickle.loads(txn.get(self._current_keys[0].encode()))
                self._parse_sample_metadata(first_sample)
            else:
                raise ValueError("No samples found for the specified split")
    
    def _parse_sample_metadata(self, sample: Dict):
        """Extract metadata from a sample."""
        data_info = sample.get('data_info', {})
        
        # Get channel information
        self._channel_names = data_info.get(self.config.channel_key, [])
        self._xyz_coords = data_info.get(self.config.xyz_key, None)
        self._n_channels = len(self._channel_names) if self._channel_names else sample['sample'].shape[0]
        
        # Get subject ID from data_info (fallback to 'subject_001' if not present)
        self._subject_id = data_info.get('subject_id', 'subject_001')
        
        # Get sampling rate from data_info (overrides config if present)
        # LMDB data_info uses 'resampling_rate' key
        if 'resampling_rate' in data_info:
            self.config.sampling_rate = float(data_info['resampling_rate'])
            logger.info(f"Using sampling_rate from LMDB data_info: {self.config.sampling_rate} Hz")
        
        # Infer sample shape and compute durations
        sample_data = sample['sample']  # Expected shape: (channels, patches, samples_per_patch)
        if sample_data.ndim == 3:
            n_channels, n_patches, samples_per_patch = sample_data.shape
            
            # Compute durations from shape and sampling_rate
            if self.config.patch_duration is None:
                self.config.patch_duration = samples_per_patch / self.config.sampling_rate
            if self.config.sample_duration is None:
                self.config.sample_duration = n_patches * self.config.patch_duration
        else:
            # Handle 2D data (channels, time)
            if self.config.sample_duration is None:
                self.config.sample_duration = sample_data.shape[1] / self.config.sampling_rate
        
        self._sample_shape = sample_data.shape
        logger.info(f"Sample shape: {self._sample_shape}, sample_duration: {self.config.sample_duration}s")
    
    def _build_time_index(self):
        """Build index mapping time to sample indices for efficient slicing. Needed when sampler requests specific time intervals -> used in get()"""
        self._sample_start_times = []
        self._sample_end_times = []
        
        current_time = 0.0
        for i, key in enumerate(self._current_keys):
            start_time = current_time
            end_time = current_time + self.config.sample_duration
            
            self._sample_start_times.append(start_time)
            self._sample_end_times.append(end_time)
            
            current_time = end_time
        
        self._total_duration = current_time
        
        # Store domain intervals for each split
        self._split_domains = {}
        cumulative_time = 0.0
        for split_name in ['train', 'val', 'test']:
            if split_name in self._keys_dict:
                n_samples = len(self._keys_dict[split_name])
                split_duration = n_samples * self.config.sample_duration
                self._split_domains[split_name] = (cumulative_time, cumulative_time + split_duration)
                cumulative_time += split_duration
    
    def _get_sample_by_key(self, key: str) -> Dict:
        """Load a single sample from LMDB."""
        with self._env.begin() as txn:
            data = txn.get(key.encode())
            if data is None:
                raise KeyError(f"Sample key '{key}' not found in LMDB")
            return pickle.loads(data)
    
    def _find_samples_in_range(self, start: float, end: float) -> List[Tuple[int, float, float]]:
        """Find all samples that overlap with the given time range.
        
        Returns:
            List of (sample_index, overlap_start, overlap_end) tuples
        """
        overlapping = []
        
        # Use binary search to find starting point
        start_idx = bisect.bisect_right(self._sample_end_times, start)
        
        for i in range(start_idx, len(self._current_keys)):
            sample_start = self._sample_start_times[i]
            sample_end = self._sample_end_times[i]
            
            # Check if sample overlaps with query range
            if sample_start >= end:
                break
            
            if sample_end > start:
                overlap_start = max(start, sample_start)
                overlap_end = min(end, sample_end)
                overlapping.append((i, overlap_start, overlap_end))
        
        return overlapping
    
    def _sample_to_continuous_array(self, sample_data: np.ndarray) -> np.ndarray:
        """Convert sample data to continuous time series format.
        
        Input: (channels, patches, samples_per_patch) or (channels, time)
        Output: (time, channels)
        """
        if sample_data.ndim == 3:
            # (channels, patches, samples_per_patch) -> (time, channels)
            n_channels, n_patches, samples_per_patch = sample_data.shape
            # Transpose to (patches, samples_per_patch, channels) then reshape
            return sample_data.transpose(1, 2, 0).reshape(-1, n_channels)
        else:
            # (channels, time) -> (time, channels)
            return sample_data.T
    
    def get(self, recording_id: str, start: float, end: float) -> Data:
        """Extract a slice of data from the virtual continuous recording.
        
        Args:
            recording_id: The recording ID (should match self.recording_id)
            start: Start time in seconds
            end: End time in seconds
            
        Returns:
            A temporaldata.Data object containing the requested slice
        """
        if recording_id != self.recording_id:
            raise ValueError(f"Unknown recording_id: {recording_id}. Expected: {self.recording_id}")
        
        # Clamp to valid range
        start = max(0, start)
        end = min(end, self._total_duration)
        
        if start >= end:
            raise ValueError(f"Invalid time range: [{start}, {end}]")
        
        # Find overlapping samples
        overlapping_samples = self._find_samples_in_range(start, end)
        
        if not overlapping_samples:
            raise ValueError(f"No data found in time range [{start}, {end}]")
        
        # Load and concatenate overlapping samples
        all_data = []
        all_labels = []
        label_timestamps = []
        
        for sample_idx, overlap_start, overlap_end in overlapping_samples:
            key = self._current_keys[sample_idx]
            sample = self._get_sample_by_key(key)
            
            sample_data = self._sample_to_continuous_array(sample['sample'])
            sample_start = self._sample_start_times[sample_idx]
            sample_end = self._sample_end_times[sample_idx]
            
            # Calculate slice indices within the sample
            sample_rate = self.config.sampling_rate
            n_timepoints = sample_data.shape[0]
            
            # Convert overlap times to indices within sample
            start_offset = overlap_start - sample_start
            end_offset = overlap_end - sample_start
            
            start_idx = int(start_offset * sample_rate)
            end_idx = int(end_offset * sample_rate)
            
            # Clamp indices
            start_idx = max(0, min(start_idx, n_timepoints))
            end_idx = max(0, min(end_idx, n_timepoints))
            
            if end_idx > start_idx:
                all_data.append(sample_data[start_idx:end_idx])
            
            # Store label with timestamp at OVERLAP center (not sample center)
            # This ensures the label falls within the requested time range
            # For DIVER pipeline, label MUST exist in LMDB
            assert 'label' in sample, f"LMDB sample '{key}' missing 'label' key. DIVER pipeline requires labels."
            all_labels.append(sample['label'])
            # Use overlap_start/overlap_end to ensure timestamp is within [start, end]
            label_timestamps.append((overlap_start + overlap_end) / 2)
        
        # Concatenate all data
        continuous_data = np.concatenate(all_data, axis=0).astype(np.float32)
        n_timepoints = continuous_data.shape[0]
        
        # Build the Data object
        data = Data(
            _absolute_start=start,
            brainset=Data(id=self.config.brainset_id),
            session=Data(id=self.config.session_id),
            subject=Data(id=self._subject_id),
            domain=Interval(
                start=np.array([0.0]),
                end=np.array([end - start])
            ),
        )
        
        # Add units (channels)
        if self._channel_names:
            units_data = {
                'id': np.array(self._channel_names, dtype='U'),
            }
            if self._xyz_coords is not None:
                xyz_array = np.array(self._xyz_coords).astype(np.float32)
                # Store full xyz_id (3D coordinates) for DIVER STCPE
                units_data['xyz_id'] = xyz_array
                # Also store imaging_plane_xy (2D) for backward compatibility
                if xyz_array.shape[1] >= 2:
                    units_data['imaging_plane_xy'] = xyz_array[:, :2]
            data.units = ArrayDict(**units_data)
        
        # Add time series data as RegularTimeSeries
        # For EEG data: use data.eeg with values
        # continuous_data shape: (T, N_channels)
        # Note: RegularTimeSeries computes timestamps automatically from sampling_rate and domain
        data.eeg = RegularTimeSeries(
            values=continuous_data,
            sampling_rate=self.config.sampling_rate,
            domain=Interval(
                start=np.array([0.0]),
                end=np.array([end - start])
            )
        )
        
        # Also keep calcium_traces for backward compatibility with POYO transforms
        data.calcium_traces = RegularTimeSeries(
            df_over_f=continuous_data,
            sampling_rate=self.config.sampling_rate,
            domain=Interval(
                start=np.array([0.0]),
                end=np.array([end - start])
            )
        )
        
        # Add task labels (required for DIVER pipeline)
        # Uses generic 'target' instead of task-specific names like 'drifting_gratings'
        assert all_labels, "No labels found in the requested time range. DIVER pipeline requires labels."
        
        label_timestamps = np.array(label_timestamps) - start  # Relative to slice start
        # Filter labels within the slice
        mask = (label_timestamps >= 0) & (label_timestamps < (end - start))
        
        assert np.any(mask), f"No labels within time range [{start}, {end}]. Check your sampling intervals."
        
        filtered_timestamps = label_timestamps[mask]
        filtered_labels = np.array(all_labels)[mask]
        
        # Generic target structure for any classification/regression task
        data.target = IrregularTimeSeries(
            timestamps=filtered_timestamps,
            label=filtered_labels.astype(np.int64),  # Classification label
            domain=Interval(
                start=np.array([0.0]),
                end=np.array([end - start])
            )
        )
        
        # Keep drifting_gratings for backward compatibility with POYO tokenizers
        # TODO: Remove this once all tokenizers use data.target
        data.drifting_gratings = IrregularTimeSeries(
            timestamps=filtered_timestamps,
            orientation_id=filtered_labels.astype(np.int64),
            orientation=filtered_labels.astype(np.float32),
            temporal_frequency_id=np.zeros_like(filtered_labels, dtype=np.int64),
            temporal_frequency=np.ones_like(filtered_labels, dtype=np.float32),
            domain=Interval(
                start=np.array([0.0]),
                end=np.array([end - start])
            )
        )
        
        # Add config for downstream processing
        data.config = self.recording_dict[recording_id]["config"]
        
        # Apply ID prefixes
        self._update_data_with_prefixed_ids(data)
        
        return data
    
    def _update_data_with_prefixed_ids(self, data: Data):
        """Add prefixes to unit ids, session id, and subject id (in place)."""
        if hasattr(data, "units") and hasattr(data.units, "id"):
            prefix_str = self.unit_id_prefix_fn(data)
            if np.__version__ >= "2.0":
                data.units.id = np.strings.add(prefix_str, data.units.id.astype(str))
            else:
                data.units.id = np.core.defchararray.add(prefix_str, data.units.id.astype(str))
        
        if hasattr(data, "session"):
            data.session.id = f"{self.session_id_prefix_fn(data)}{data.session.id}"
        
        if hasattr(data, "subject"):
            data.subject.id = f"{self.subject_id_prefix_fn(data)}{data.subject.id}"
    
    def get_sampling_intervals(self) -> Dict[str, Interval]:
        """Get the sampling intervals for the current split.
        
        Returns:
            Dictionary mapping recording_id to Interval object
        
        For DIVER LMDB: Returns intervals aligned with sample boundaries.
        Each sample becomes one interval, so RandomFixedWindowSampler will
        pick windows that exactly match sample boundaries (no mid-sample windows).
        """
        # DIVER LMDB: Return intervals aligned with each sample's boundaries
        # This ensures RandomFixedWindowSampler picks windows that match samples exactly
        sampling_intervals = Interval(
            start=np.array(self._sample_start_times),
            end=np.array(self._sample_end_times)
        )
        
        # Apply sampling_intervals_modifier if specified
        modifier_code = self.config.sampling_intervals_modifier
        if modifier_code is not None:
            # Create a mock data object for the modifier
            mock_data = self._create_mock_data_for_modifier()
            local_vars = {
                "data": mock_data,
                "sampling_intervals": sampling_intervals,
                "split": self.split,
            }
            try:
                exec(modifier_code, {}, local_vars)
                sampling_intervals = local_vars.get("sampling_intervals")
            except Exception as e:
                logger.warning(f"Error executing sampling_intervals_modifier: {e}")
        
        return {self.recording_id: sampling_intervals}
    
    def _create_mock_data_for_modifier(self) -> Data:
        """Create a minimal Data object for sampling_intervals_modifier execution."""
        data = Data(
            domain=Interval(start=np.array([0.0]), end=np.array([self._total_duration]))
        )
        
        # Add drifting_gratings if we have labels
        if len(self._current_keys) > 0:
            timestamps = np.array([(s + e) / 2 for s, e in 
                                   zip(self._sample_start_times, self._sample_end_times)])
            starts = np.array(self._sample_start_times)
            ends = np.array(self._sample_end_times)
            
            data.drifting_gratings = Interval(
                start=starts,
                end=ends,
                timestamps=timestamps,
            )
        
        return data
    #TODO : handle this. for get_unit_ids, get_session_ids, get_subject_ids they are set with mock_data
    def get_unit_ids(self) -> List[str]:
        """Get all unit (channel) IDs with prefixes applied."""
        if not self._channel_names:
            return []
        
        # Create a mock data object to apply prefix function
        mock_data = Data(  
            brainset=Data(id=self.config.brainset_id),
            session=Data(id=self.config.session_id),
        )
        prefix = self.unit_id_prefix_fn(mock_data)
        
        return [f"{prefix}{ch}" for ch in self._channel_names]
    
    def get_session_ids(self) -> List[str]:
        """Get all session IDs with prefixes applied."""
        mock_data = Data(
            brainset=Data(id=self.config.brainset_id),
            session=Data(id=self.config.session_id),
        )
        return [f"{self.session_id_prefix_fn(mock_data)}{self.config.session_id}"]
    
    def get_subject_ids(self) -> List[str]:
        """Get all subject IDs with prefixes applied."""
        mock_data = Data(
            brainset=Data(id=self.config.brainset_id),
            subject=Data(id=self._subject_id),
        )
        return [f"{self.subject_id_prefix_fn(mock_data)}{self._subject_id}"]
    
    def get_brainset_ids(self) -> List[str]:
        """Get all brainset IDs."""
        return [self.config.brainset_id]
    
    def get_recording_config_dict(self) -> Dict[str, Dict]:
        """Get configuration dictionary for each recording."""
        return {
            self.recording_id: self.recording_dict[self.recording_id]["config"]
        }
    
    def disable_data_leakage_check(self):
        """Disable data leakage checking (for compatibility with Dataset interface)."""
        self._check_for_data_leakage_flag = False
        logger.warning("Data leakage check is disabled.")
    
    def __getitem__(self, index: DatasetIndex) -> Any:
        """Get a data sample by DatasetIndex.
        
        Args:
            index: DatasetIndex with recording_id, start, and end times
            
        Returns:
            Data object, or transformed output if transform is set
        """
        sample = self.get(index.recording_id, index.start, index.end)
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
    
    def __len__(self):
        """Length is not defined for continuous datasets (same as Dataset)."""
        raise NotImplementedError("Length of dataset is not defined. Use sampler to generate indices.")
    
    def __del__(self):
        """Clean up LMDB environment."""
        if hasattr(self, '_env') and self._env:
            self._env.close()
    
    def __repr__(self):
        return (f"DatasetFromLmdb(lmdb_path={self.config.lmdb_path}, "
                f"recording_id={self.recording_id}, split={self.split}, "
                f"duration={self._total_duration:.2f}s)")
