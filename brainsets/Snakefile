from snakemake.utils import min_version
min_version("6.0")

local_env = snakemake.shell(f"source ./detect_environment.sh", read=True).strip()
print(f"Using environment {local_env}, as determined by detect_environment.sh")
configfile: f"environments/{local_env}.yaml"

def expand_path(path):
    # Expands environment variables like $VAR
    path = os.path.expandvars(path)

    # Expands the '~' symbol to the user's home directory
    path = os.path.expanduser(path)

    return path

# get various paths from config file
config["TMP_DIR"] = expand_path(f"{config['tmp_dir']}")
config["RAW_DIR"] = expand_path(f"{config['raw_dir']}/raw")
config["PROCESSED_DIR"] = expand_path(f"{config['processed_dir']}/processed")
config["COMPRESSED_DIR"] = expand_path(f"{config['compressed_dir']}/compressed")
config["UNCOMPRESSED_DIR"] = expand_path(f"{config['uncompressed_dir']}/uncompressed")


# include all snakefiles for all individual datasets
# includes are relative to the directory of the Snakefile in which they occur
module allen_brain_observatory_calcium_module:
    snakefile: "pipelines/allen_brain_observatory_calcium/Snakefile"
    config: config
use rule * from allen_brain_observatory_calcium_module as allen_brain_observatory_calcium_*
use rule all from allen_brain_observatory_calcium_module as allen_brain_observatory_calcium

module allen_natural_movie_calcium_module:
    snakefile: "pipelines/allen_natural_movie_calcium/Snakefile"
    config: config
use rule * from allen_natural_movie_calcium_module as allen_natural_movie_calcium_*
use rule all from allen_natural_movie_calcium_module as allen_natural_movie_calcium

module allen_visual_behavior_neuropixels_module:
    snakefile: "pipelines/allen_visual_behavior_neuropixels/Snakefile"
    config: config
use rule * from allen_visual_behavior_neuropixels_module as allen_visual_behavior_neuropixels_*
use rule all from allen_visual_behavior_neuropixels_module as allen_visual_behavior_neuropixels

module gillon_richards_responses_2023_module:
    snakefile: "pipelines/gillon_richards_responses_2023/Snakefile"
    config: config
use rule * from gillon_richards_responses_2023_module as gillon_richards_responses_2023_*
use rule all from gillon_richards_responses_2023_module as gillon_richards_responses_2023

module perich_miller_population_2018_module:
    snakefile: "pipelines/perich_miller_population_2018/Snakefile"
    config: config
use rule * from perich_miller_population_2018_module as perich_miller_population_2018_*
use rule all from perich_miller_population_2018_module as perich_miller_population_2018

module evenchen_shenoy_structure_2019_module:
    snakefile: "pipelines/evenchen_shenoy_structure_2019/Snakefile"
    config: config
use rule * from evenchen_shenoy_structure_2019_module as evenchen_shenoy_structure_2019_*
use rule all from evenchen_shenoy_structure_2019_module as evenchen_shenoy_structure_2019

module willett_shenoy_module:
    snakefile: "pipelines/willett_shenoy/Snakefile"
    config: config
use rule * from willett_shenoy_module as willett_shenoy_*
use rule all from willett_shenoy_module as willett_shenoy

module odoherty_sabes_nonhuman_2017_module:
    snakefile: "pipelines/odoherty_sabes_nonhuman_2017/Snakefile"
    config: config
use rule * from odoherty_sabes_nonhuman_2017_module as odoherty_sabes_nonhuman_2017_*
use rule all from odoherty_sabes_nonhuman_2017_module as odoherty_sabes_nonhuman_2017

module churchland_shenoy_neural_2012_module:
    snakefile: "pipelines/churchland_shenoy_neural_2012/Snakefile"
    config: config
use rule * from churchland_shenoy_neural_2012_module as churchland_shenoy_neural_2012_*
use rule all from churchland_shenoy_neural_2012_module as churchland_shenoy_neural_2012

module mc_maze_small_module:
    snakefile: "pipelines/mc_maze_small/Snakefile"
    config: config
use rule * from mc_maze_small_module as mc_maze_small_*
use rule all from mc_maze_small_module as mc_maze_small

module bouchard_chang_module:
    snakefile: "pipelines/bouchard_chang/Snakefile"
    config: config
use rule * from bouchard_chang_module as bouchard_chang_*
use rule all from bouchard_chang_module as bouchard_chang

module flint_slutzky_accurate_2012_module:
    snakefile: "pipelines/flint_slutzky_accurate_2012/Snakefile"
    config: config
use rule * from flint_slutzky_accurate_2012_module as flint_slutzky_accurate_2012_*
use rule all from flint_slutzky_accurate_2012_module as flint_slutzky_accurate_2012

module willett_henderson_speech_2023_module:
    snakefile: "pipelines/willett_henderson_speech_2023/Snakefile"
    config: config
use rule * from willett_henderson_speech_2023_module as willett_henderson_speech_2023_*
use rule all from willett_henderson_speech_2023_module as willett_henderson_speech_2023

module orsborn_lab_ecog_reaching_2024_module:
    snakefile: "pipelines/orsborn_lab_ecog_reaching_2024/Snakefile"
    config: config
use rule * from orsborn_lab_ecog_reaching_2024_module as orsborn_lab_ecog_reaching_2024_*
use rule all from orsborn_lab_ecog_reaching_2024_module as orsborn_lab_ecog_reaching_2024


module yu_smith_selective_2022_module:
    snakefile: "pipelines/yu_smith_selective_2022/Snakefile"
    config: config
use rule * from yu_smith_selective_2022_module as yu_smith_selective_2022_*
use rule all from yu_smith_selective_2022_module as yu_smith_selective_2022

# make rules that combine multiple datasets
rule poyo_neurips:
    input:
        perich_miller_population_2018_module.rules.all.input,
        churchland_shenoy_neural_2012_module.rules.all.input,
        flint_slutzky_accurate_2012_module.rules.all.input,
        odoherty_sabes_nonhuman_2017_module.rules.all.input,
