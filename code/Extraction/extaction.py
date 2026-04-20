import time
import pandas as pd
import re
import requests
from requests.exceptions import JSONDecodeError

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from huggingface_hub import HfApi, login, SpaceHardware
from huggingface_hub.hf_api import ModelInfo, RepoFile
from huggingface_hub import ModelSearchArguments, DatasetSearchArguments, ModelCard
from huggingface_hub import hf_hub_url, get_hf_file_metadata

import os
from dotenv import load_dotenv

class HFExtraction():


    def __init__(self):
        api = HfApi()
        print("holi")

    def get_models(self):
        models = api.list_models(
            cardData=True,
            full=True,
            fetch_config=True)


def retrieve_emission_parameters(self, model):
    """
    Retrieve emission parameters from a given model.

    Args:
        model: The model object.

    Returns:
        A tuple containing emissions, source, training_type, geographical_location, and hardware_used.
    """

    if 'co2_eq_emissions' not in model.cardData:
        return None, None, None, None, None

    if type(model.cardData["co2_eq_emissions"]) is dict:
        emissions_dict = model.cardData["co2_eq_emissions"]
    else:
        emissions = model.cardData["co2_eq_emissions"]
        return emissions, None, None, None, None

    emissions = emissions_dict["emissions"]

    if 'source' in emissions_dict:
        source = emissions_dict['source']
    else:
        source = None

    if 'training_type' in emissions_dict:
        training_type = emissions_dict['training_type']
    else:
        training_type = None

    if 'geographical_location' in emissions_dict:
        geographical_location = emissions_dict['geographical_location']
    else:
        geographical_location = None

    if 'hardware_used' in emissions_dict:
        hardware_used = emissions_dict['hardware_used']
    else:
        hardware_used = None

    return emissions, source, training_type, geographical_location, hardware_used


def find_model_accuracy(self, modelId):
    """
    Find the accuracy of a model given its ID.

    Args:
        modelId: The ID of the model.

    Returns:
        The accuracy of the model or None if not found.
    """

    try:
        modelCard_text = ModelCard.load(modelId).text
    except:
        return None

    accuracy_regex = r"Accuracy\s*:\s*.*?(?=\n)"

    accuracy_match = re.search(accuracy_regex, modelCard_text)

    if accuracy_match is None:
        return None

    accuracy_match = accuracy_match.group(0)

    accuracy_match = float(accuracy_match.split(':')[1])

    return accuracy_match


def find_model_validation_metric(self, modelId, metric):
    """
    Find the validation metric of a model given its ID and metric.

    Args:
        modelId: The ID of the model.
        metric: The evaluation metric to search for.

    Returns:
        The validation metric value or None if not found.
    """

    try:
        modelCard_text = ModelCard.load(modelId).text
    except:
        return None

    accuracy_regex = fr'{metric}\s*:\s*.*?(?=\n)'

    accuracy_match = re.search(accuracy_regex, modelCard_text)

    if accuracy_match is None:
        return None

    accuracy_match = accuracy_match.group(0)

    accuracy_match = float(accuracy_match.split(':')[1])

    return accuracy_match


def retrieve_model_tags(self, model):
    """
    Retrieve tags from a given model.

    Args:
        model: The model object.

    Returns:
        A list of tags associated with the model.
    """

    tags = list(set(model.tags + [model.pipeline_tag]))
    if hasattr(model, 'cardData') and 'tags' in model.cardData:
        if type(model.cardData['tags']) is list:
            try:  # we only lose 3 rows by doing this
                tags = list(set(tags + model.cardData['tags']))
            except:
                print(model.cardData['tags'])
        else:
            tags = list(set(tags + [model.cardData['tags']]))

    tags = [tag for tag in tags if tag is not None]

    return tags


def find_model_size(self, modelId):
    """
    Find the size of a model given its ID.

    Args:
        modelId: The ID of the model.

    Returns:
        The size of the model or None if not found.
    """

    try:
        return get_hf_file_metadata(hf_hub_url(repo_id=modelId, filename="pytorch_model.bin")).size
    except:
        return None


def retrieve_model_datasets(self, model):
    """
    Retrieve the datasets used by a given model.

    Args:
        model: The model object.

    Returns:
        A list of datasets used by the model.
    """

    if hasattr(model, 'cardData') and 'datasets' in model.cardData:
        if type(model.cardData["datasets"]) is list:
            datasets = model.cardData["datasets"]
        else:
            datasets = model.cardData["datasets"]
    else:
        datasets = ['']

    return datasets


def find_datasets_size(self, datasets):
    """
    Find the size of datasets used by a given model.

    Args:
        datasets: A list of datasets.

    Returns:
        The total size of the datasets or None if not found.
    """

    datasets_size = 0
    if datasets is None:
        return None

    for dataset in datasets:
        try:
            datasets_size += api.dataset_info(dataset).cardData["dataset_info"]["dataset_size"]
        except:
            pass

    return datasets_size


def extract_from_model_index(self, model_index):
    """
    Extract evaluation metrics from a model index.

    Args:
        model_index: The model index object.

    Returns:
        A tuple containing accuracy, f1, loss, rouge1, and rougeL.
    """

    accuracy = f1 = loss = rouge1 = rougeL = None
    if model_index is not None and 'results' in model_index and model_index['results'] and isinstance(
            model_index['results'][0], dict) \
            and 'metrics' in model_index['results'][0]:
        metrics_list = model_index['results'][0]['metrics']
        for metric_type in metrics_list:

            metric = metric_value = None

            if 'type' in metric_type and 'value' in metric_type:
                if 'value' in metric_type:
                    metric = metric_type['type']
                    metric_value = metric_type['value']

            if metric == 'accuracy':
                accuracy = metric_value
            elif metric_value == 'f1':
                f1 = metric_value
            elif metric_value == 'loss':
                loss = metric_value
            elif metric_value == 'rouge1':
                rouge1 = metric_value
            elif metric_value == 'rougeL':
                rougeL = metric_value

    return accuracy, f1, loss, rouge1, rougeL


def extract_evaluation_from_modelcard(self, model):
    """
    Extract evaluation metrics from a model card.

    Args:
        model: The model object.

    Returns:
        A tuple containing accuracy, f1, loss, rouge1, and rougeL.
    """

    accuracy = f1 = loss = rouge1 = rougeL = None
    if hasattr(model, 'cardData'):
        if 'model-index' in model.cardData:
            model_index = model.cardData['model-index'][0] if isinstance(model.cardData['model-index'], list) else \
            model.cardData['model-index']
            accuracy, f1, loss, rouge1, rougeL = self.extract_from_model_index(model_index)
        elif 'model_index' in model.cardData:
            model_index = model.cardData['model_index'][0] if isinstance(model.cardData['model_index'], list) else \
            model.cardData['model_index']
            accuracy, f1, loss, rouge1, rougeL = self.extract_from_model_index(model_index)

        elif 'metrics' in model.cardData and model.cardData["metrics"] is not None and isinstance(
                model.cardData["metrics"][0], dict):
            for metric_dict in model.cardData["metrics"]:
                metric, metric_value = list(metric_dict.items())[0]
                if metric == 'accuracy':
                    accuracy = metric_value
                elif metric_value == 'f1':
                    f1 = metric_value
                elif metric_value == 'loss':
                    loss = metric_value
                elif metric_value == 'rouge1':
                    rouge1 = metric_value
                elif metric_value == 'rougeL':
                    rougeL = metric_value

    return accuracy, f1, loss, rouge1, rougeL


def extract_evaluation_metrics(self, model, auto):
    """
    Extract evaluation metrics from a model with the option to use auto mode.

    Args:
        model: The model object.
        auto: A boolean flag to indicate if autotrain/autonlp tags should be considered.

    Returns:
        A tuple containing accuracy, f1, loss, rouge1, and rougeL.
    """

    accuracy, f1, loss, rouge1, rougeL = self.extract_evaluation_from_modelcard(model)

    if auto:
        if accuracy == None:
            accuracy = self.find_model_validation_metric(model.modelId, 'Accuracy')
        if f1 == None:
            f1 = self.find_model_validation_metric(model.modelId, r'(F1|Macro F1)')
        if loss == None:
            loss = self.find_model_validation_metric(model.modelId, 'Loss')
        if rouge1 == None:
            rouge1 = self.find_model_validation_metric(model.modelId, 'Rouge1')
        if rougeL == None:
            rougeL = self.find_model_validation_metric(model.modelId, 'RougeL')

    return accuracy, f1, loss, rouge1, rougeL


def api_calls_parameters(self, model, datasets):
    """
    Get size, datasets size, and creation date from API calls.

    Args:
        model: The model object.
        datasets: A list of datasets.

    Returns:
        A tuple containing size, datasets_size, and created_at.
    """

    size = datasets_size = created_at = None
    api_token =  os.getenv("HF_TOKEN")
    headers = {"authorization": f"Bearer {api_token}"}

    try:
        commits = requests.get(f'https://huggingface.co/api/models/{model.modelId}/commits/main', timeout=2,
                               headers=headers)
    except requests.exceptions.Timeout:
        print(f'Timeout error for commits on model {model.modelId}')
        created_at = None
    except JSONDecodeError:
        print(f'JSON decode error for commits on model {model.modelId}')
        created_at = None
    except Exception as e:
        print(f'Unexpected error for commits on model {model.modelId}: {e}')
        created_at = None
    else:
        try:
            created_at = commits.json()[-1]['date']
        except Exception as e:
            print(f'Error extracting "created_at" for model {model.modelId}')
            created_at = None

    return size, datasets_size, created_at


def get_modelcard_text(self, model):
    """
    Get the text of a model card.

    Args:
        model: The model object.

    Returns:
        The text of the model card or None if not found.
    """

    card_text = None
    try:
        card_text = ModelCard.load(model.modelId).text
    except:
        pass
    return card_text


def process_model(self, model):
    """
    Process a model and extract relevant information.

    Args:
    model: A tuple containing the model object.

    Returns:
        A dictionary containing the processed model information.
    """

    if model[0] % 10 == 0:
        print(model[0])

    model = model[1]
    try:
        tags = self.retrieve_model_tags(model)
        datasets = self.retrieve_model_datasets(model)
        auto = 'autotrain' in tags or 'autonlp' in tags
        library_name = model.library_name if hasattr(model, 'library_name') else None
        accuracy, f1, loss, rouge1, rougeL = extract_evaluation_metrics(model, auto)

        size, datasets_size, created_at = self.api_calls_parameters(model, datasets)

        card_text = get_modelcard_text(model)
        emissions = source = training_type = geographical_location = hardware_used = None

        if hasattr(model, 'cardData') and "co2_eq_emissions" in model.cardData:
            size = self.find_model_size(model.modelId)
            datasets_size = self.find_datasets_size(datasets)
            emissions, source, training_type, geographical_location, hardware_used = self.retrieve_emission_parameters(model)

        return {'modelId': model.modelId,
                'tags': tags,
                'datasets': datasets,
                'datasets_size': datasets_size,
                'co2_eq_emissions': emissions,
                'source': source,
                'training_type': training_type,
                'geographical_location': geographical_location,
                'hardware_used': hardware_used,
                'accuracy': accuracy,
                'loss': loss,
                'f1': f1,
                'rouge1': rouge1,
                'rougeL': rougeL,
                'size': size,
                'auto': auto,
                'downloads': model.downloads,
                'likes': model.likes,
                'library_name': library_name,
                'lastModified': model.lastModified,
                'created_at': created_at,
                'modelcard_text': card_text}
    except Exception as e:
        print(f'{model.modelId} could not be processed: ', str(e))
