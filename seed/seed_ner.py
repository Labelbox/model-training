import uuid

from datasets import load_dataset
from tqdm import tqdm

from labelbox import Client, LabelImport, OntologyBuilder, Tool


ENTITIES = {
    0: '[PAD]',
    1: 'B-PER',
    2: 'I-PER',
    3: 'B-ORG',
    4: 'I-ORG',
    5: 'B-LOC',
    6: 'I-LOC',
    7: 'B-MISC',
    8: 'I-MISC'
}

def create_ontology(client):
    ontology_builder = OntologyBuilder(tools=[
        Tool(tool=Tool.Type.NER, name=name)
        for name in list(ENTITIES.values())[1:]
    ])
    return client.create_ontology("ner_ontology", ontology_builder.asdict())

def get_feature_schema_lookup(ontology):
    return {
        tool.name: tool.feature_schema_id
        for tool in ontology.tools()
    }

def setup_project(client):
    project = client.create_project(name="ner_training_project")
    dataset = client.create_dataset(name="ner_training_dataset")
    ontology = create_ontology(client)
    project.setup_editor(ontology)
    project.datasets.connect(dataset)
    return project, dataset, ontology


def generate_label(feature_schema_lookup, tokens, ner_tags):
    text = ""
    idx = 0
    annotations = []
    for token, ner_tag in zip(tokens, ner_tags):
        text += token + " "
        if ner_tag != 0:
            annotations.append({
                "uuid": str(uuid.uuid4()),
                "schemaId": feature_schema_lookup[ENTITIES[ner_tag]],
                "dataRow": {
                    "id": None
                },
                "location": {
                    "start": idx,
                    "end": idx + len(token) - 1
                }
            })
        idx += len(token) + 1
    return annotations, text


def create_labels(feature_schema_lookup):
    conll_data = load_dataset("conll2003")
    label_data = {}
    for tokens, ner_tags in tqdm(
            zip(conll_data['train']['tokens'],
                conll_data['train']['ner_tags'])):
        annotations, text = generate_label(feature_schema_lookup, tokens,
                                           ner_tags)
        if len(annotations):
            label_data[str(uuid.uuid4())] = {
                'text': text,
                'annotations': annotations
            }

        if len(label_data) > 3000:
            break
    return label_data



def assign_data_row_ids(client, labels):
    for external_id, data_row_ids in tqdm(
            client.get_data_row_ids_for_external_ids(list(
                labels.keys())).items()):
        data = labels[external_id]['annotations']
        data_row_id = data_row_ids[0]
        for annot in data:
            annot['dataRow'] = {'id': data_row_id}


def flatten_labels(labels):
    return [
        annotation for label in labels.values()
        for annotation in label['annotations']
    ]


def main(client):
    project, dataset, ontology = setup_project(client)
    feature_schema_lookup = get_feature_schema_lookup(ontology)
    labels = create_labels(feature_schema_lookup)
    task = dataset.create_data_rows([{
        'row_data': data['text'],
        'external_id': external_id
    } for external_id, data in labels.items()])
    task.wait_till_done()
    assign_data_row_ids(client, labels)
    annotations = flatten_labels(labels)
    print(f"Uploading {len(annotations)} annotations.")
    job = LabelImport.create_from_objects(client, project.uid,
                                          str(uuid.uuid4()), annotations)
    job.wait_until_done()
    errors = job.errors
    if not len(errors):
        print("Successfully uploaded")
        lb_model = client.create_model(name=f"ner_model",
                            ontology_id=project.ontology().uid)
        print(f"Successfully seeded data and created model. Setup a model run here: https://app.labelbox.com/models/{lb_model.uid}")
    else:
        print("Upload contained errors: ", errors)


if __name__ == '__main__':
    client = Client()
    main(client)

