from uuid import uuid4

import tensorflow_datasets as tfds
from tqdm import tqdm

from labelbox import Client, LabelImport, Classification, OntologyBuilder, Option



CLASS_MAPPINGS = {0: 'negative', 1: 'positive'}


def get_feature_schema_lookup(ontology):
    classification = ontology.classifications()[0]
    return {
        'classification': classification.feature_schema_id,
        'options': {
            option.value: option.feature_schema_id
            for option in classification.options
        }
    }

def create_ontology(client):
    ontology_builder = OntologyBuilder(classifications=[
        Classification(Classification.Type.RADIO,
                       "positive or negative",
                       options=[Option("positive"),
                                Option("negative")]),
    ])
    return client.create_ontology("text_single_classification_ontology", ontology_builder.asdict())


def setup_project(client):
    project = client.create_project(name="text_single_classification_project")
    dataset = client.create_dataset(name="text_single_classification_dataset")
    project.datasets.connect(dataset)
    ontology = create_ontology(client)
    project.setup_editor(ontology)
    return project, dataset, ontology


def assign_data_row_ids_to_label_data(client, seed_data):
    for external_id, data_row_ids in tqdm(
            client.get_data_row_ids_for_external_ids(list(seed_data.keys())).items()):
        data_row_id = data_row_ids[0]
        seed_data[external_id]['label_data']['dataRow'] = {'id': data_row_id}


def process_source_data(feature_schema_lookup):
    source_data = tfds.load('imdb_reviews', split='train')
    seed_data = {}

    for idx, example in enumerate(tqdm(source_data.as_numpy_iterator())):
        if idx == 2000:
            break

        external_id = str(uuid4())
        seed_data[external_id] = {
            'row_data' : example['text'].decode('utf8'),
            'label_data' : {
            "uuid": str(uuid4()),
            "schemaId": feature_schema_lookup['classification'],
            "dataRow": {
                "id": None
            },
            "answer": {
                "schemaId":
                    feature_schema_lookup['options']
                    [CLASS_MAPPINGS[example['label']]]
            }
        }
    }
    return seed_data


def main(client):
    project, dataset, ontology = setup_project(client)
    feature_schema_lookup = get_feature_schema_lookup(ontology)
    seed_data = process_source_data(feature_schema_lookup)
    task = dataset.create_data_rows([{'external_id' : k, 'row_data' : v['row_data']} for k,v in seed_data.items()])
    task.wait_till_done()
    assign_data_row_ids_to_label_data(client, seed_data)

    annotations =  [label_seed_data['label_data'] for label_seed_data in seed_data.values() ]
    print(f"Uploading {len(annotations)} annotations.")
    job = LabelImport.create_from_objects(client, project.uid, str(uuid4()),
                                        annotations)
    job.wait_until_done()
    errors = job.errors
    if not len(errors):
        print("Successfully uploaded")
        lb_model = client.create_model(name=f"text_single_classification_model",
                               ontology_id=project.ontology().uid)
        print(f"Successfully seeded data and created model. Setup a model run here: https://app.labelbox.com/models/{lb_model.uid}")
    else:
        print("Upload contained errors: ", errors)


if __name__ == '__main__':
    client = Client()
    main(client)
