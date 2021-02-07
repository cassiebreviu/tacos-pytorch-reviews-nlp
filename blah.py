from azureml.core import Workspace, Dataset

subscription_id = '91d27443-f037-45d9-bb0c-428256992df6'
resource_group = 'robots'
workspace_name = 'hal'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='prepared_reviews_ds', version="20")
dataset.download(target_path='.data', overwrite=False)