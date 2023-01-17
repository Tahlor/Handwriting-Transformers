from pathlib import Path
import site
from download_resources.download import s3_download

import logging
import traceback

logger = logging.getLogger("root")
logger.setLevel(logging.DEBUG)

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
s3_hw_all = "s3://datascience-computervision-l3apps/HWR/synthetic-data/python-package-resources/generated-handwriting/single-word-datasets/iam-cvl-32px-top10k-words/synth_hw.zip"


def download_handwriting(overwrite=False):
    """ Download a single handwriting style (for dev)
    """
    return s3_download(s3_path=s3_hw_sample, local_folder=HW_GENERATED, overwrite=overwrite, is_zip=False)

def download_handwriting_zip(overwrite=False):
    """ Download all handwriting styles (for prod)
    """
    return s3_download(s3_path=s3_hw_all, local_folder=HW_GENERATED, overwrite=overwrite, is_zip=True)

def download_model_resources(overwrite=False):
    """ Downloads handwriting models (.pt)
    """
    return s3_download(s3_path=s3_hwr_models, local_path=HW_MODELS, overwrite=overwrite, is_zip=False)


if __name__=='__main__':
    download_handwriting()
    download_handwriting_zip()
    download_model_resources()