import uuid
from collections import defaultdict

from tqdm import tqdm
import tensorflow_datasets as tfds

from labelbox import Client, LabelImport, Classification, OntologyBuilder, Option


CLASS_MAPPINGS_COURSE = ["DESC", "ENTY", "ABBR", "HUM", "NUM", "LOC"]
CLASS_MAPPINGS_FINE = [
    "manner",
    "cremat",
    "animal",
    "exp",
    "ind",
    "gr",
    "title",
    "def",
    "date",
    "reason",
    "event",
    "state",
    "desc",
    "count",
    "other",
    "letter",
    "religion",
    "food",
    "country",
    "color",
    "termeq",
    "city",
    "body",
    "dismed",
    "mount",
    "money",
    "product",
    "period",
    "substance",
    "sport",
    "plant",
    "techmeth",
    "volsize",
    "instru",
    "abb",
    "speed",
    "word",
    "lang",
    "perc",
    "code",
    "dist",
    "temp",
    "symbol",
    "ord",
    "veh",
    "weight",
    "currency",
]

def create_ontology(client):
    ontology_builder = OntologyBuilder(classifications=[
        Classification(Classification.Type.RADIO,
                       "fine",
                       options=[Option(name) for name in CLASS_MAPPINGS_FINE]),
        Classification(Classification.Type.RADIO,
                       "course",
                       options=[Option(name)
                                for name in CLASS_MAPPINGS_COURSE]),
    ])
    return client.create_ontology("text_multi_classification_ontology", ontology_builder.asdict())


def get_feature_schema_lookup(ontology):
    classification = ontology.classifications()
    return {
        classification[0].name: classification[0].feature_schema_id,
        classification[1].name: classification[1].feature_schema_id,
        f'{classification[0].name}-options': {
            option.value: option.feature_schema_id
            for option in classification[0].options
        },
        f'{classification[1].name}-options': {
            option.value: option.feature_schema_id
            for option in classification[1].options
        }
    }

def setup_project(client):
    project = client.create_project(name="text_multi_classification_project")
    dataset = client.create_dataset(name="text_multi_classification_dataset")
    ontology = create_ontology(client)
    project.setup_editor(ontology)
    project.datasets.connect(dataset)
    return project, dataset, ontology


def process_source_data(feature_schema_lookup):
    ds = tfds.load('trec', split='train')
    seed_labels = defaultdict(list)
    data_row_data = []
    for idx, example in tqdm(enumerate(ds.as_numpy_iterator())):
        if idx == 2000:
            break
        fine_class_name, course_class_name = example['label-fine'], example[
            'label-coarse']
        external_id = str(uuid.uuid4())
        data_row_data.append({
            'external_id': external_id,
            'row_data': example['text'].decode('utf8')
        })
        seed_labels[external_id].extend([{
            "uuid": str(uuid.uuid4()),
            "schemaId": feature_schema_lookup['fine'],
            "dataRow": {
                "id": external_id
            },
            "answer": {
                "schemaId":
                    feature_schema_lookup['fine-options']
                    [CLASS_MAPPINGS_FINE[fine_class_name]]
            }
        }, {
            "uuid": str(uuid.uuid4()),
            "schemaId": feature_schema_lookup['course'],
            "dataRow": {
                "id": external_id
            },
            "answer": {
                "schemaId":
                    feature_schema_lookup['course-options']
                    [CLASS_MAPPINGS_COURSE[course_class_name]]
            }
        }])
    return seed_labels, data_row_data


def assign_data_row_ids(client, labels):
    for external_id, data_row_ids in tqdm(client.get_data_row_ids_for_external_ids(list(labels.keys())).items()):
        data = labels[external_id]
        data_row_id = data_row_ids[0]
        for annot in data:
            annot['dataRow'] = {'id': data_row_id}


def flatten_labels(labels):
    return [annotation for label in labels.values() for annotation in label]


def main(client):
    project, dataset, ontology = setup_project(client)
    feature_schema_lookup = get_feature_schema_lookup(ontology)
    seed_labels, data_row_data = process_source_data(feature_schema_lookup)
    task = dataset.create_data_rows(data_row_data)
    task.wait_till_done()

    assign_data_row_ids(client, seed_labels)
    annotations = flatten_labels(seed_labels)
    print(f"Uploading {len(annotations)} annotations.")
    job = LabelImport.create_from_objects(client, project.uid, str(uuid.uuid4()),
                                          annotations)
    job.wait_until_done()
    errors = job.errors
    if not len(errors):
        print("Successfully uploaded")
        lb_model = client.create_model(name=f"text_multi_classification_model",
                               ontology_id=project.ontology().uid)
        print(f"Successfully seeded data and created model. Setup a model run here: https://app.labelbox.com/models/{lb_model.uid}")
    else:
        print("Upload contained errors: ", errors)


if __name__ == '__main__':
    client = Client()
    main(client)
