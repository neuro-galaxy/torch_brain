from torch_brain.pipeline import BrainsetPipeline


class Pipeline(BrainsetPipeline):
    @classmethod
    def get_manifest(cls, raw_dir, processed_dir, args): ...

    def download(self, manifest_item): ...

    def process(self, download_output): ...
