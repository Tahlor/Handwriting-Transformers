from pathlib import Path
import site
try:
    from download_resources.download import s3_download
except:
    print("""Could not import s3_download from download_resources, cannot download from S3. Please install download_resources: 
    github.com/tahlor/download_resources""")
import traceback
import logging
import site
from pathlib import Path
from hwgen.params import set_globals
logger = logging.getLogger("root")
logger.setLevel(logging.INFO)

class HandwritingResourceManager:

    def __init__(self,
                 hwgen_resource_path=None,
                 dataset="IAM",
                 english_words_path=None,
                 link_resources=True):
        self.logger = logging.getLogger("root")
        self.logger.setLevel(logging.INFO)
        self.hwgen_package_resource_path = Path(site.getsitepackages()[0]) / "hwgen/resources"
        if hwgen_resource_path is None:
            self.hwgen_resource_path = self.hwgen_package_resource_path
        else:
            self.hwgen_resource_path = Path(hwgen_resource_path)

        self.hw_models = self.hwgen_resource_path / "models"
        self.hw_generated = self.hwgen_resource_path / "generated"
        self.hw_models.mkdir(exist_ok=True, parents=True)
        self.hw_generated.mkdir(exist_ok=True, parents=True)
        self.models, self.styles = self.get_model_paths(self.hwgen_resource_path)
        set_globals(hw_model_path=self.hw_models, dataset=dataset)
        self.english_words_path = english_words_path
        if english_words_path is None:
            if (self.hwgen_resource_path / "english_words.txt").exists():
                self.english_words_path = self.hwgen_resource_path / "english_words.txt"
            elif (self.hw_models / "english_words.txt").exists():
                self.english_words_path = self.hw_models / "english_words.txt"

        if link_resources:
            if not self.hwgen_package_resource_path.exists():
                self.link_resources()
            else:
                self.logger.info(f"{self.hwgen_package_resource_path} already exists, not linking")

    def link_resources(self):
        """ Link resources from the package to the site-packages folder
        """
        self.logger.info(f"Linking resources from {self.hwgen_resource_path} to {self.hwgen_package_resource_path}")
        try:
            self.hwgen_package_resource_path.symlink_to(self.hwgen_resource_path)
        except Exception as e:
            self.logger.warning(f"Could not link resources from {self.hwgen_resource_path} to {self.hwgen_package_resource_path}")
            self.logger.warning(traceback.format_exc())

    @staticmethod
    def get_model_paths(model_folder=None):
        models = {"IAM": model_folder / 'iam_model.pth', "CVL": model_folder / 'cvl_model.pth'}
        styles = {"IAM": model_folder / 'IAM-32.pickle', "CVL": model_folder / 'CVL-32.pickle'}
        return models, styles

    def download_handwriting(self, overwrite=False):
        """ Download a single handwriting style (for dev)
        """
        return s3_download(s3_path=s3_hw_sample, local_folder=self.hw_generated, overwrite=overwrite, is_zip=False)
    

    def download_handwriting_zip(self, version="CVL", overwrite=False):
        """ Download collection of handwriting styles (for prod)
        """
        s3_path = s3_generated_handwriting_paths[version]
        return s3_download(s3_path=s3_path, local_folder=self.hw_generated, overwrite=overwrite, is_zip=True)


    def download_handwriting_zip_set(self, version="eng_latest",overwrite=False):
        """ Download all handwriting styles (for prod)
        """
        if version in ["eng_latest", "IAM+CVL"]:
            s3_paths = s3_generated_handwriting_paths["IAM"], s3_generated_handwriting_paths["CVL"]
        else:
            s3_paths = [s3_generated_handwriting_paths[version]]

        for s3_path in s3_paths:
            try:
                yield s3_download(s3_path=s3_path, local_folder=self.hw_generated, overwrite=overwrite, is_zip=True)
            except:
                logger.error(f"Failed to download {s3_path}")
                logger.error(traceback.format_exc())

    def download_model_resources(self, overwrite=False):
        """ Downloads handwriting models (.pt)
        """
        return s3_download(s3_path=s3_hwr_models, local_path=self.hw_models, overwrite=overwrite, is_zip=False, is_s3_folder=True)

    def download_model(self, model_name):
        if model_name == "sample":
            dataset_root = Path(self.download_handwriting()).parent
            dataset_files = list(dataset_root.glob("*.npy"))
        elif model_name in s3_generated_handwriting_paths.keys():
            dataset_root = self.download_handwriting_zip(version=model_name)
            dataset_files = list(dataset_root.rglob("*.npy"))
        elif model_name in ["IAM+CVL", "eng_latest"]:
            paths = self.download_handwriting_zip_set(model_name)
            dataset_files = []
            for path in paths:
                if path.exists():
                    dataset_files += list(path.rglob("*.npy"))
            dataset_root = self.hw_generated
        else:
            return Path(model_name), None
        return dataset_root, dataset_files


# Handwriting Models
s3_hwr_models = "s3://datascience-computervision-l3apps/HWR/synthetic-data/python-package-resources/handwriting-models"

# Pregenerate Handwriting
s3_hw_sample = "s3://datascience-computervision-l3apps/HWR/synthetic-data/python-package-resources/generated-handwriting/single-word-datasets/iam-cvl-32px-top10k-words/style_600_IAM_IAM_samples.npy"
s3_hw_cvl = "s3://datascience-computervision-l3apps/HWR/synthetic-data/python-package-resources/generated-handwriting/single-word-datasets/iam-cvl-32px-top10k-words/CVL.zip"
s3_hw_iam = "s3://datascience-computervision-l3apps/HWR/synthetic-data/python-package-resources/generated-handwriting/single-word-datasets/iam-cvl-32px-top10k-words/IAM.zip"

s3_generated_handwriting_paths = {"CVL": s3_hw_cvl, "IAM": s3_hw_iam}

# OLD GLOBALS - DEPRECATED - TRY NOT TO USE
HWGEN_RESOURCES = Path(site.getsitepackages()[0]) / "hwgen/resources"
HW_MODELS = HWGEN_RESOURCES / "models"
HW_GENERATED = HWGEN_RESOURCES / "generated"
HW_MODELS.mkdir(exist_ok=True, parents=True)
HW_GENERATED.mkdir(exist_ok=True, parents=True)


if __name__=='__main__':
    hmm = HandwritingResourceManager()
    #download_handwriting()
    hmm.download_handwriting_zip()
    #download_model_resources()