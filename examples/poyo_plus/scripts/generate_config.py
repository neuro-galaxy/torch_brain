import csv
import os
import yaml
from collections import defaultdict
from typing import Dict, List, Tuple

from torch_brain.registry import MODALITY_REGISTRY, DataType

# Holdout sessions from capoyo_single_session.yaml
HOLDOUT_SESSIONS = {
    "710504563",
    "623339221",
    "589441079",
    "603763073",
    "676503588",
    "652092676",
    "649409874",
    "671164733",
    "623347352",
    "649401936",
    "555042467",
    "646016204",
    "595273803",
    "539487468",
    "637669284",
    "539497234",
    "652737678",
    "654532828",
    "669233895",
    "560926639",
    "547388708",
    "595806300",
    "689388034",
    "649938038",
    "645689073",
    "510514474",
    "505695962",
    "512326618",
    "562122508",
    "653122667",
}

def read_task_contents(csv_path: str, holdout_sessions: set = None) -> Dict[str, Dict[str, bool]]:
    """Read CSV and return dict mapping session_id to task flags, excluding holdout sessions."""
    if holdout_sessions is None:
        holdout_sessions = HOLDOUT_SESSIONS
    
    sessions = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            session_id = row['session_id'].replace('.h5', '')
            # Skip holdout sessions
            if session_id in holdout_sessions:
                continue
            
            tasks = {
                'pupil.location': row['pupil.location'] == 'True',
                'running.running_speed': row['running.running_speed'] == 'True',
                'drifting_gratings.orientation': row['drifting_gratings.orientation'] == 'True',
                'drifting_gratings.temporal_frequency': row['drifting_gratings.temporal_frequency'] == 'True',
                'static_gratings.phase': row['static_gratings.phase'] == 'True',
                'static_gratings.spatial_frequency': row['static_gratings.spatial_frequency'] == 'True',
                'static_gratings.orientation': row['static_gratings.orientation'] == 'True',
                'natural_scenes.frame': row['natural_scenes.frame'] == 'True',
                'natural_movie_one.frame': row['natural_movie_one.frame'] == 'True',
                'natural_movie_two.frame': row['natural_movie_two.frame'] == 'True',
                'natural_movie_three.frame': row['natural_movie_three.frame'] == 'True',
                'locally_sparse_noise.frame': row['locally_sparse_noise.frame'] == 'True',
            }
            sessions[session_id] = tasks
    return sessions

def group_sessions_by_tasks(sessions: Dict[str, Dict[str, bool]]) -> Dict[Tuple, List[str]]:
    """Group sessions by their exact task combination."""
    groups = defaultdict(list)
    for session_id, tasks in sessions.items():
        # Create a tuple key from all task values
        task_key = tuple(sorted(tasks.items()))
        groups[task_key].append(session_id)
    return groups

# Mapping from CSV task flags to registry modality names
TASK_TO_MODALITY = {
    'drifting_gratings.orientation': 'drifting_gratings_orientation',
    'drifting_gratings.temporal_frequency': 'drifting_gratings_temporal_frequency',
    'static_gratings.orientation': 'static_gratings_orientation',
    'static_gratings.spatial_frequency': 'static_gratings_spatial_frequency',
    'static_gratings.phase': 'static_gratings_phase',
    'natural_scenes.frame': 'natural_scenes',
    'natural_movie_one.frame': 'natural_movie_one_frame',
    'natural_movie_two.frame': 'natural_movie_two_frame',
    'natural_movie_three.frame': 'natural_movie_three_frame',
    'locally_sparse_noise.frame': 'locally_sparse_noise_frame',
    'running.running_speed': 'running_speed',
    'pupil.location': 'pupil_location',
}

# Normalization values from calculate_normalization_scales.py output
NORMALIZATION_VALUES = {
    'running_speed': {'mean': 6.80354332, 'std': 13.87822103},
    'pupil_location.x': {'mean': 11.02599208, 'std': 15.94543917},
    'pupil_location.y': {'mean': 16.91118513, 'std': 6.83344030},
}

# Task weights from Table A2
TASK_WEIGHTS = {
    'drifting_gratings_orientation': 1.0,
    'drifting_gratings_temporal_frequency': 1.0,
    'natural_movie_one_frame': 0.25,
    'natural_movie_two_frame': 0.2,
    'natural_movie_three_frame': 0.2,
    'locally_sparse_noise_frame': 1.0,
    'static_gratings_orientation': 1.0,
    'static_gratings_spatial_frequency': 1.0,
    'static_gratings_phase': 1.0,
    'natural_scenes': 0.3,
    'running_speed': 1.5,
    'pupil_location': 8.0,  # Applied to both x and y
}

def create_readout_config(tasks: Dict[str, bool]) -> List[Dict]:
    """Create multitask_readout config based on which tasks are True, using registry."""
    readouts = []
    
    for task_key, is_active in tasks.items():
        if not is_active:
            continue
        
        # Get modality name from mapping
        modality_name = TASK_TO_MODALITY.get(task_key)
        if not modality_name:
            continue
        
        # Get modality spec from registry
        if modality_name not in MODALITY_REGISTRY:
            continue
        
        modality_spec = MODALITY_REGISTRY[modality_name]
        
        # Special handling for pupil_location - split into x and y
        if modality_name == 'pupil_location' and modality_spec.type == DataType.CONTINUOUS and modality_spec.dim == 2:
            # Create two separate readouts for x and y
            for coord in ['x', 'y']:
                readout_id = f'{modality_name}.{coord}'
                readout = {
                    'readout_id': readout_id,
                    'timestamp_key': modality_spec.timestamp_key,
                    'value_key': modality_spec.value_key,
                    'metrics': [{
                        'metric': {
                            '_target_': 'torchmetrics.MeanSquaredError'
                        }
                    }]
                }
                
                # Add normalization values if available
                if readout_id in NORMALIZATION_VALUES:
                    readout['normalize_mean'] = NORMALIZATION_VALUES[readout_id]['mean']
                    readout['normalize_std'] = NORMALIZATION_VALUES[readout_id]['std']
                else:
                    readout['normalize_mean'] = 0.0
                    readout['normalize_std'] = 1.0
                
                # Add weight for pupil_location (applies to both x and y)
                if modality_name in TASK_WEIGHTS:
                    readout['weights'] = TASK_WEIGHTS[modality_name]
                
                readouts.append(readout)
        else:
            # Create readout config for other modalities
            readout = {
                'readout_id': modality_name,
                'timestamp_key': modality_spec.timestamp_key,
                'value_key': modality_spec.value_key,
                'metrics': []
            }
            
            # Add weight if available
            if modality_name in TASK_WEIGHTS:
                readout['weights'] = TASK_WEIGHTS[modality_name]
            
            # Add normalization for continuous variables (like running_speed)
            if modality_spec.type == DataType.CONTINUOUS:
                # Use normalization values if available
                if modality_name in NORMALIZATION_VALUES:
                    readout['normalize_mean'] = NORMALIZATION_VALUES[modality_name]['mean']
                    readout['normalize_std'] = NORMALIZATION_VALUES[modality_name]['std']
                else:
                    readout['normalize_mean'] = 0.0
                    readout['normalize_std'] = 1.0
                
                readout['metrics'].append({
                    'metric': {
                        '_target_': 'torchmetrics.MeanSquaredError'
                    }
                })
            elif modality_spec.type == DataType.MULTINOMIAL:
                readout['metrics'].append({
                    'metric': {
                        '_target_': 'torchmetrics.Accuracy',
                        'task': 'multiclass',
                        'num_classes': modality_spec.dim
                    }
                })
            else:
                # For other types, use MSE as default
                readout['metrics'].append({
                    'metric': {
                        '_target_': 'torchmetrics.MeanSquaredError'
                    }
                })
            
            readouts.append(readout)
    
    return readouts

def get_sampling_intervals_modifier(tasks: Dict[str, bool]) -> str:
    """Determine sampling_intervals_modifier based on tasks."""
    has_drifting = tasks['drifting_gratings.orientation'] or tasks['drifting_gratings.temporal_frequency']
    has_static = tasks['static_gratings.orientation'] or tasks['static_gratings.spatial_frequency'] or tasks['static_gratings.phase']
    
    if has_drifting:
        return "sampling_intervals = sampling_intervals & data.drifting_gratings.coalesce(0.3)"
    elif has_static:
        return "sampling_intervals = sampling_intervals & data.static_gratings.coalesce(0.3)"
    else:
        return None

def generate_yaml_config(groups: Dict[Tuple, List[str]], sessions: Dict[str, Dict[str, bool]]) -> List[Dict]:
    """Generate YAML config structure from grouped sessions."""
    config_groups = []
    
    for task_key, session_list in sorted(groups.items(), key=lambda x: len(x[1]), reverse=True):
        # Convert tuple back to dict
        tasks = dict(task_key)
        
        # Get first session's tasks to determine config
        first_session = session_list[0]
        tasks_dict = sessions[first_session]
        
        # Create config - note: YAML format uses '-' at top level
        group_config = {
            'selection': [{
                'brainset': 'allen_visual_coding_ophys_2016',
                'sessions': sorted(session_list)
            }],
            'config': {}
        }
        
        # Add sampling_intervals_modifier if needed
        modifier = get_sampling_intervals_modifier(tasks_dict)
        if modifier:
            group_config['config']['sampling_intervals_modifier'] = modifier
        
        # Add multitask_readout
        readouts = create_readout_config(tasks_dict)
        if readouts:
            group_config['config']['multitask_readout'] = readouts
        
        config_groups.append(group_config)
    
    return config_groups

def format_task_name(task_key: str) -> str:
    """Format task key for display in comments."""
    if 'drifting_gratings.orientation' in task_key:
        return 'DG_orient'
    elif 'drifting_gratings.temporal_frequency' in task_key:
        return 'DG_temp_freq'
    elif 'static_gratings.orientation' in task_key:
        return 'SG_orient'
    elif 'static_gratings.spatial_frequency' in task_key:
        return 'SG_spat_freq'
    elif 'static_gratings.phase' in task_key:
        return 'SG_phase'
    elif 'natural_scenes.frame' in task_key:
        return 'NS'
    elif 'natural_movie_one.frame' in task_key:
        return 'NM1'
    elif 'natural_movie_two.frame' in task_key:
        return 'NM2'
    elif 'natural_movie_three.frame' in task_key:
        return 'NM3'
    elif 'locally_sparse_noise.frame' in task_key:
        return 'LSN'
    elif 'running.running_speed' in task_key:
        return 'running'
    elif 'pupil.location' in task_key:
        return 'pupil'
    return task_key

def main():
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Excluding {len(HOLDOUT_SESSIONS)} holdout sessions")
    
    # Read task contents, excluding holdout sessions
    csv_path = os.path.join(script_dir, 'task_contents.csv')
    sessions = read_task_contents(csv_path, HOLDOUT_SESSIONS)
    
    # Group sessions by exact task combination
    groups = group_sessions_by_tasks(sessions)
    
    # Print number of groups
    print(f"Number of groups: {len(groups)}")
    print(f"Total sessions: {sum(len(sessions) for sessions in groups.values())}")
    
    # Print group sizes
    print("\nGroup sizes:")
    for i, (task_key, session_list) in enumerate(sorted(groups.items(), key=lambda x: len(x[1]), reverse=True), 1):
        tasks_dict = dict(task_key)
        active_tasks = [format_task_name(k) for k, v in tasks_dict.items() if v]
        print(f"  Group {i}: {len(session_list)} sessions - Tasks: {', '.join(active_tasks) if active_tasks else 'None'}")
    
    # Generate YAML config
    config_groups = generate_yaml_config(groups, sessions)
    
    # Write to YAML file with proper formatting
    output_lines = []
    for i, group in enumerate(config_groups, 1):
        # Find the task key for this group
        session_list = group['selection'][0]['sessions']
        task_key = None
        for tk, sl in groups.items():
            if set(sl) == set(session_list):
                task_key = tk
                break
        
        if task_key:
            tasks_dict = dict(task_key)
            active_tasks = [format_task_name(k) for k, v in tasks_dict.items() if v]
            comment = f"# Group {i} - {len(session_list)} sessions"
            if active_tasks:
                comment += f" - {', '.join(active_tasks)}"
            output_lines.append(comment)
        
        # Format as YAML - each group is a list item starting with '-'
        output_lines.append('- selection:')
        output_lines.append('  - brainset: allen_visual_coding_ophys_2016')
        output_lines.append('    sessions:')
        for session in sorted(session_list):
            output_lines.append(f'    - "{session}"')
        
        # Add config section
        config = group['config']
        if config:
            output_lines.append('  config:')
            
            # Add sampling_intervals_modifier if present
            if 'sampling_intervals_modifier' in config:
                output_lines.append('    sampling_intervals_modifier: |')
                modifier = config['sampling_intervals_modifier']
                for line in modifier.split('\n'):
                    output_lines.append(f'      {line}')
            
            # Add multitask_readout
            if 'multitask_readout' in config:
                output_lines.append('    multitask_readout:')
                for readout in config['multitask_readout']:
                    output_lines.append('      - readout_id: ' + readout['readout_id'])
                    if 'weights' in readout:
                        output_lines.append(f"        weights: {readout['weights']}")
                    if 'normalize_mean' in readout:
                        output_lines.append(f"        normalize_mean: {readout['normalize_mean']}")
                    if 'normalize_std' in readout:
                        output_lines.append(f"        normalize_std: {readout['normalize_std']}")
                    output_lines.append(f"        timestamp_key: {readout['timestamp_key']}")
                    output_lines.append(f"        value_key: {readout['value_key']}")
                    output_lines.append('        metrics:')
                    for metric in readout['metrics']:
                        output_lines.append('          - metric:')
                        metric_dict = metric['metric']
                        output_lines.append(f"              _target_: {metric_dict['_target_']}")
                        if 'task' in metric_dict:
                            output_lines.append(f"              task: {metric_dict['task']}")
                        if 'num_classes' in metric_dict:
                            output_lines.append(f"              num_classes: {metric_dict['num_classes']}")
        
        output_lines.append('')  # Add blank line between groups
    
    output_path = os.path.join(script_dir, 'capoyo_regenerated.yaml')
    with open(output_path, 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f"\nGenerated config written to {output_path}")

if __name__ == '__main__':
    main()
