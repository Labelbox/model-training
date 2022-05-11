import uuid
from io import BytesIO

from tqdm import tqdm
from scipy.io import loadmat
import requests

from labelbox import (
    Client,
    LabelImport,
    Classification,
    OntologyBuilder,
    Option
)

def create_ontology(client, class_names):
    ontology_builder = OntologyBuilder(classifications=[
        Classification(Classification.Type.CHECKLIST,
                       "description",
                       options=[Option(name) for name in class_names]),
    ])
    return client.create_ontology("image_multi_classification_ontology", ontology_builder.asdict())

def get_feature_schema_lookup(ontology):
    classification = ontology.classifications()[0]
    return {
        'classification': classification.feature_schema_id,
        'options': {
            option.value: option.feature_schema_id
            for option in classification.options
        }
    }

def setup_project(client, class_names):
    project = client.create_project(name="image_multi_classification_project")
    dataset = client.create_dataset(name="image_multi_classification_dataset")
    ontology = create_ontology(client, class_names)
    project.setup_editor(ontology)
    project.datasets.connect(dataset)
    return project, dataset, ontology


def load_source_data():
    miml_data = requests.get("https://storage.googleapis.com/vertex-matt-test/multi_classification_seed_datarows/miml_data.mat")
    return  loadmat(BytesIO(miml_data.content))

def process_source_data(class_names, feature_schema_lookup):
    source_data = load_source_data()

    seed_data = {}
    for example_idx in tqdm(range(source_data['targets'].shape[-1])):
        classes = [
            class_names[i]
            for i in range(len(class_names))
            if source_data['targets'][i, example_idx] > 0
        ]
        image_path = f"https://storage.googleapis.com/vertex-matt-test/multi_classification_seed_datarows/images/{1+example_idx}.jpg"
        external_id = str(uuid.uuid4())
        seed_data[external_id] = {
            'row_data' : image_path,
            'label_data' : {
                "uuid":
                    str(uuid.uuid4()),
                "schemaId":
                    feature_schema_lookup['classification'],
                "dataRow": {
                    "id": external_id
                },
                "answers": [{
                    "schemaId": feature_schema_lookup['options'][class_name]
                } for class_name in classes]
            }
        }
    return seed_data


def assign_data_row_ids_to_label_data(client, seed_data):
    for external_id, data_row_ids in tqdm(
            client.get_data_row_ids_for_external_ids(list(seed_data.keys())).items()):
        data_row_id = data_row_ids[0]
        seed_data[external_id]['label_data']['dataRow'] = {'id': data_row_id}


def main(client):
    class_names = ['desert', 'mountains', 'sea', 'sunset', 'trees']
    project, dataset, ontology = setup_project(client, class_names)
    feature_schema_lookup = get_feature_schema_lookup(ontology)
    seed_data = process_source_data(class_names, feature_schema_lookup)
    task = dataset.create_data_rows([{'external_id' : k, 'row_data' : v['row_data']} for k,v in seed_data.items()])
    task.wait_till_done()
    assign_data_row_ids_to_label_data(client, seed_data)

    annotations =  [label_seed_data['label_data'] for label_seed_data in seed_data.values() ]
    print(f"Uploading {len(annotations)} annotations.")
    job = LabelImport.create_from_objects(client, project.uid, str(uuid.uuid4()),
                                        annotations)
    job.wait_until_done()
    errors = job.errors
    if not len(errors):
        print("Successfully uploaded")
        lb_model = client.create_model(name=f"image_multi_classification_model",
                               ontology_id=project.ontology().uid)
        print(f"Successfully seeded data and created model. Setup a model run here: https://app.labelbox.com/models/{lb_model.uid}")
    else:
        print("Upload contained errors: ", errors)


if __name__ == '__main__':
    client = Client()
    main(client)
