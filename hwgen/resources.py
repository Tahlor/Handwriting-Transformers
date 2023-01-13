from pathlib import Path
import site


import logging

logger = logging.getLogger("root")
logger.setLevel(logging.DEBUG)

HWR_MODEL_PATH = Path(site.getsitepackages()[0]) / "hwgen"

models = {"IAM": HWR_MODEL_PATH / 'iam_model.pth', HWR_MODEL_PATH/ "CVL": 'cvl_model.pth'}
styles = {"IAM": HWR_MODEL_PATH / 'IAM-32.pickle', HWR_MODEL_PATH/ "CVL": 'CVL-32.pickle'}

def download_resources(force=False):
    s3_hwr = "s3://datascience-computervision-l3apps/HWR/synthetic-data/python-package-resources/handwriting-models"

    try:
        from download_resources.download import download_s3_folder, directory_is_empty
        logger.info(f"Resources path: {HWR_MODEL_PATH}")
        if force or not HWR_MODEL_PATH.exists() or directory_is_empty(HWR_MODEL_PATH):
            download_s3_folder(s3_hwr, HWR_MODEL_PATH)
        else:
            logger.info(f"Resources path: {HWR_MODEL_PATH} already exists with files")
        return HWR_MODEL_PATH

    except:
        logger.error("Could not download resources")
        logger.warning(f"You will need to download the models manually from {s3_hwr} and put them in {HWR_MODEL_PATH}")
        return False

if __name__=='__main__':
    download_resources(force=True)