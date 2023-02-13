from pathlib import Path
import site
try:
    from download_resources.download import s3_download
except:
    print("""Could not import s3_download from download_resources, cannot download from S3. Please install download_resources: 
    github.com/tahlor/download_resources""")

import logging
import traceback

logger = logging.getLogger("root")
logger.setLevel(logging.INFO)

HWGEN_RESOURCES = Path(site.getsitepackages()[0]) / "hwgen/resources"
HW_MODELS = HWGEN_RESOURCES / "models"
HW_GENERATED = HWGEN_RESOURCES / "generated"
HW_MODELS.mkdir(exist_ok=True, parents=True)
HW_GENERATED.mkdir(exist_ok=True, parents=True)

models = {"IAM": HW_MODELS / 'iam_model.pth', HW_MODELS/ "CVL": 'cvl_model.pth'}
styles = {"IAM": HW_MODELS / 'IAM-32.pickle', HW_MODELS/ "CVL": 'CVL-32.pickle'}

# Handwriting Models
s3_hwr_models = "s3://datascience-computervision-l3apps/HWR/synthetic-data/python-package-resources/handwriting-models"

# Pregenerate Handwriting
s3_hw_sample = "s3://datascience-computervision-l3apps/HWR/synthetic-data/python-package-resources/generated-handwriting/single-word-datasets/iam-cvl-32px-top10k-words/style_600_IAM_IAM_samples.npy"
s3_hw_cvl = "s3://datascience-computervision-l3apps/HWR/synthetic-data/python-package-resources/generated-handwriting/single-word-datasets/iam-cvl-32px-top10k-words/CVL.zip"
s3_hw_iam = "s3://datascience-computervision-l3apps/HWR/synthetic-data/python-package-resources/generated-handwriting/single-word-datasets/iam-cvl-32px-top10k-words/IAM.zip"

s3_generated_handwriting_paths = {"CVL": s3_hw_cvl, "IAM": s3_hw_iam}

def download_handwriting(overwrite=False):
    """ Download a single handwriting style (for dev)
    """
    return s3_download(s3_path=s3_hw_sample, local_folder=HW_GENERATED, overwrite=overwrite, is_zip=False)


def download_handwriting_zip(version="CVL", overwrite=False):
    """ Download collection of handwriting styles (for prod)
    """
    s3_path = s3_generated_handwriting_paths[version]
    return s3_download(s3_path=s3_path, local_folder=HW_GENERATED, overwrite=overwrite, is_zip=True)


def download_handwriting_zip_set(version="eng_latest",overwrite=False):
    """ Download all handwriting styles (for prod)
    """
    if version in ["eng_latest", "IAM+CVL"]:
        s3_paths = s3_generated_handwriting_paths["IAM"], s3_generated_handwriting_paths["CVL"]
    else:
        s3_paths = [s3_generated_handwriting_paths[version]]

    for s3_path in s3_paths:
        try:
            yield s3_download(s3_path=s3_path, local_folder=HW_GENERATED, overwrite=overwrite, is_zip=True)
        except:
            logger.error(f"Failed to download {s3_path}")
            logger.error(traceback.format_exc())

def download_model_resources(overwrite=False):
    """ Downloads handwriting models (.pt)
    """
    return s3_download(s3_path=s3_hwr_models, local_path=HW_MODELS, overwrite=overwrite, is_zip=False, is_s3_folder=True)

def download_model(model_name):
    if model_name == "sample":
        dataset_root = Path(download_handwriting()).parent
        dataset_files = list(dataset_root.glob("*.npy"))
    elif model_name in s3_generated_handwriting_paths.keys():
        dataset_root = download_handwriting_zip(version=model_name)
        dataset_files = list(dataset_root.rglob("*.npy"))
    elif model_name in ["IAM+CVL", "eng_latest"]:
        paths = download_handwriting_zip_set(model_name)
        dataset_files = []
        for path in paths:
            if path.exists():
                dataset_files += list(path.rglob("*.npy"))
        dataset_root = HW_GENERATED
    else:
        return Path(model_name), None
    return dataset_root, dataset_files

if __name__=='__main__':
    #download_handwriting()
    download_handwriting_zip()
    #download_model_resources()