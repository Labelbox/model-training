import uuid
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm
import tensorflow_datasets as tfds

from labelbox import Client, LabelImport , Classification, OntologyBuilder, Option


def create_ontology(client):
    ontology_builder = OntologyBuilder(classifications=[
        Classification(Classification.Type.RADIO,
                       "dog or cat",
                       options=[Option("dog"), Option("cat")]),
    ])
    return client.create_ontology("image_single_classification_ontology", ontology_builder.asdict())

def get_feature_schema_lookup(ontology):
    classification = ontology.classifications()[0]
    return  {
        'classification': classification.feature_schema_id,
        'options': {
            option.value: option.feature_schema_id
            for option in classification.options
        }
    }

def setup_project(client):
    project = client.create_project(name="image_single_classification_project")
    dataset = client.create_dataset(name="image_single_classification_dataset")
    ontology = create_ontology(client)
    project.setup_editor(ontology)
    project.datasets.connect(dataset)
    return project, dataset, ontology

def generate_annotation(dataset, example, class_mappings, feature_schema_lookup):
    im_bytes = BytesIO()
    Image.fromarray(example['image']).save(im_bytes, format="jpeg")
    uri = client.upload_data(content=im_bytes.getvalue(),
                                filename=f"{uuid.uuid4()}.jpg")
    data_row = dataset.create_data_row(row_data=uri)
    return {
            "uuid": str(uuid.uuid4()),
            "schemaId": feature_schema_lookup['classification'],
            "dataRow": {
                "id": data_row.uid
            },
            "answer": {
                "schemaId":
                    feature_schema_lookup['options']
                    [class_mappings[example['label']]]
            }
        }

def generate_annotations(dataset, feature_schema_lookup):
    ds = tfds.load('cats_vs_dogs', split='train')
    class_mappings = {0: 'cat', 1: 'dog'}
    annotations = []
    max_examples = 350
    futures = []
    with ThreadPoolExecutor() as exc:
        for idx, example in enumerate(ds.as_numpy_iterator()):
            if idx > max_examples:
                break
            futures.append(exc.submit(generate_annotation, dataset, example, class_mappings, feature_schema_lookup))
        for f in tqdm(as_completed(futures)):
            annotations.append(f.result())
    return annotations

def main(client):
    project, dataset, ontology = setup_project(client)
    feature_schema_lookup = get_feature_schema_lookup(ontology)
    annotations = generate_annotations(dataset, feature_schema_lookup)

    print(f"Uploading {len(annotations)} annotations.")
    job = LabelImport.create_from_objects(client, project.uid, str(uuid.uuid4()),
                                          annotations)
    job.wait_until_done()
    errors = job.errors
    if not len(errors):
        print("Successfully uploaded")
        lb_model = client.create_model(name=f"image_single_classification_model",
                            ontology_id=project.ontology().uid)
        print(f"Successfully seeded data and created model. Setup a model run here: https://app.labelbox.com/models/{lb_model.uid}")
    else:
        print("Upload contained errors: ", errors)


if __name__ == '__main__':
    client = Client()
    main(client)

