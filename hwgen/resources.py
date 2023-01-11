from pathlib import Path
import site


HWR_MODEL_PATH = Path(site.getsitepackages()[0]) / "hwgen"

models = {"IAM": HWR_MODEL_PATH / 'iam_model.pth', HWR_MODEL_PATH/ "CVL": 'cvl_model.pth'}
styles = {"IAM": HWR_MODEL_PATH / 'IAM-32.pickle', HWR_MODEL_PATH/ "CVL": 'CVL-32.pickle'}

def download_resources():
    from download_resources.download import download_s3_folder
    if not HWR_MODEL_PATH.exists():
        s3_hwr = "s3://datascience-computervision-l3apps/HWR/synthetic-data/python-package-resources/handwriting-models"
        download_s3_folder(s3_hwr, HWR_MODEL_PATH)


if __name__=='__main__':
    pass