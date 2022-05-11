import uuid
import json
import pathlib

from labelbox import Client, DataRow, OntologyBuilder, Tool, LabelImport
from labelbox.data.serialization import NDJsonConverter, LBV1Converter


def load_labels():
    with open(pathlib.Path(__file__).parent.resolve() / "assets/proj_ckq778m4g0edr0yao004l41l7_export.json") as f:
        labels = json.load(f)
        return LBV1Converter().deserialize(labels).as_list()

def generate_annotations(project, dataset):
    labels = load_labels()
    for label in labels:
        annots = label.annotations
        for annotation in annots:
            annotation.feature_schema_id = None
        external_id = label.data.external_id.split("/")[-1]
        label.data.uid = dataset.data_row_for_external_id(external_id).uid
    labels.assign_feature_schema_ids(OntologyBuilder().from_project(project))
    return list(NDJsonConverter.serialize(labels))


def create_ontology(client):
    ontology_builder = OntologyBuilder()
    ontology_builder.add_tool(Tool(tool=Tool.Type.BBOX, name="person"))
    ontology_builder.add_tool(Tool(tool=Tool.Type.BBOX, name="animal"))
    return client.create_ontology("bbox_training_ontology",ontology_builder.asdict() )

def append_data_rows(dataset):
    datarows = []
    # filenames are 0 to 259 .jpg
    for image_idx in range(259):
        datarows.append({
            DataRow.row_data: f"https://storage.googleapis.com/vertex-matt-test/bbox_seed_datarows/{image_idx}.jpg",
            DataRow.external_id: f"{image_idx}.jpg"
        })

    task = dataset.create_data_rows(datarows)
    task.wait_till_done()

def setup_project(client):
    """Creates a project and dataset.
    Uses assets in repository to create datarows
    """
    project = client.create_project(name="bbox_training_project")
    dataset = client.create_dataset(name="bbox_training_dataset")
    ontology = create_ontology(client)
    project.setup_editor(ontology)
    project.datasets.connect(dataset)
    return project, dataset


def main(client):
    project, dataset = setup_project(client)
    append_data_rows(dataset)
    annotations = generate_annotations(project, dataset)
    print(f"Uploading {len(annotations)} annotations.")
    job = LabelImport.create_from_objects(client, project.uid, str(uuid.uuid4()),
                                      annotations)

    errors = job.errors
    if not len(errors):
        print("Successfully uploaded")
        lb_model = client.create_model(name=f"bounding_box_model",
                               ontology_id=project.ontology().uid)
        print(f"Successfully seeded data and created model. Setup a model run here: https://app.labelbox.com/models/{lb_model.uid}")
    else:
        print("Upload contained errors: ", errors)

if __name__ == '__main__':
    client = Client()
    main(client)







