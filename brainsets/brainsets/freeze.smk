def aggregate_input(wildcards):
    with checkpoints.download_data.get(**wildcards).output[0].open() as manifest:
        files = [line.strip() for line in manifest]
    return expand(f"{PROCESSED_DIR}/{DATASET}/tmp/{{file}}.txt", file=files)

rule merge_manifests:
    input:
        aggregate_input
    output:
        f"{PROCESSED_DIR}/{DATASET}/manifest.txt"
    shell:
        f"""
        find {PROCESSED_DIR}/{DATASET}/ -type f -name "*.h5" | sed "s|^{PROCESSED_DIR}/{DATASET}//||" > {{output}}        
        """

rule all:
    input:
        f"{PROCESSED_DIR}/{DATASET}/manifest.txt"


COMPRESSED_DIR = config["COMPRESSED_DIR"]
UNCOMPRESSED_DIR = config["UNCOMPRESSED_DIR"]

rule freeze:
    input:
        description = f"{PROCESSED_DIR}/{DATASET}/description.mpk"
    output:
        data_tar = f"{COMPRESSED_DIR}/{DATASET}/dataset.tar.lz4",
        desc_out = f"{COMPRESSED_DIR}/{DATASET}/description.mpk"
    shell:
        f"""
        mkdir -p {COMPRESSED_DIR}/{DATASET}

        # Single lz4 archive.
        echo "Compressing"
        cd {PROCESSED_DIR}/{DATASET} && \
            tar -cf - --exclude=description.mpk . | lz4 -1 > {COMPRESSED_DIR}/{DATASET}/dataset.tar.lz4
        cd - > /dev/null
        pwd
        cp {{input.description}} {COMPRESSED_DIR}/{DATASET}/description.mpk
        """

rule unfreeze:
    input:
        data_tar = f"{COMPRESSED_DIR}/{DATASET}/dataset.tar.lz4",
        desc_in = f"{COMPRESSED_DIR}/{DATASET}/description.mpk"
    output:
        unfreeze_out = f"{UNCOMPRESSED_DIR}/{DATASET}/unfreeze.done"
    shell:
        f"""
        # Uncompress and untar the data.
        mkdir -p {UNCOMPRESSED_DIR}/{DATASET}/
        lz4 -d -c {COMPRESSED_DIR}/{DATASET}/dataset.tar.lz4 | tar -xf - -C {UNCOMPRESSED_DIR}/{DATASET}/
        cp {{input.desc_in}} {UNCOMPRESSED_DIR}/{DATASET}/description.mpk
        touch {{output.unfreeze_out}}
        """
