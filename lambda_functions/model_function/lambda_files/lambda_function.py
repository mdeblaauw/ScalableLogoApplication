import json
import boto3
from io import BytesIO
import os
import time
from torchvision import transforms
from PIL import Image
from resnet_model import ResNet18
from protonet import *

s3 = boto3.client('s3')
dynamodb = boto3.client('dynamodb')
embedding_dim = 256
distance = "gaussian"

def lambda_handler(event, context):
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        file_key = record['s3']['object']['key']
        print(bucket)
        print(file_key)
        
    if bucket == os.environ['preparation_bucket']:
        prepare_model("/tmp/preparation-files")
    else:
        load_and_store_files("/tmp/model-weights", "/tmp/preparation-files")
        print(event)
        query,file_key = get_image(event)
        model = install_model()
        values = make_inference(query, model)
        store_values(values, file_key)

    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }


def load_and_store_files(weight_path, prepare_path):
    if not os.path.isdir(weight_path):
        os.mkdir(weight_path)
        content = s3.list_objects_v2(Bucket=os.environ['preparation_bucket'], Prefix='model-weights')
        key = content['Contents'][0]['Key']
        print('/tmp/' + key)
        s3.download_file(os.environ['preparation_bucket'], key, '/tmp/' + key)
        print('no weights')
    if not os.path.isdir(prepare_path):
        os.mkdir(prepare_path)
        print('no preparation files')
        s3.download_file(os.environ['preparation_bucket'], 'preparation-files/S.pt', prepare_path + '/S.pt')
        s3.download_file(os.environ['preparation_bucket'], 'preparation-files/prototypes.pt',
                         prepare_path + '/prototypes.pt')
        s3.download_file(os.environ['preparation_bucket'], 'preparation-files/reverse_class_mapping.json',
                         prepare_path + '/reverse_class_mapping.json')

def prepare_model(prepare_path):
    if not os.path.isdir(prepare_path):
        os.mkdir(prepare_path)
    s3.download_file(os.environ['preparation_bucket'], 'preparation-files/S.pt', prepare_path + '/S.pt')
    s3.download_file(os.environ['preparation_bucket'], 'preparation-files/prototypes.pt',prepare_path + '/prototypes.pt')
    s3.download_file(os.environ['preparation_bucket'], 'preparation-files/reverse_class_mapping.json',prepare_path + '/reverse_class_mapping.json')

def get_image(event):
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        file_key = record['s3']['object']['key']
        print(bucket)
        print(file_key)

    with BytesIO() as files:
        s3.download_fileobj(bucket, file_key, files)
        query = Image.open(files).convert("RGB")

        return query,file_key


def install_model():
    model_name = os.listdir("/tmp/model-weights")[0]
    model = ResNet18(flatten=True).to("cpu", dtype=torch.float)
    model.load_state_dict(torch.load(os.path.join("/tmp/model-weights", model_name), map_location="cpu"))
    model.eval()
    return model


def make_inference(query, model):
    prototypes = torch.load('/tmp/preparation-files/prototypes.pt')
    S = torch.load('/tmp/preparation-files/S.pt')
    with open('/tmp/preparation-files/reverse_class_mapping.json') as json_file:
        reverse_class_mapping = json.load(json_file)

    processed_query = preprocess_image(query)

    query_output = model(processed_query)
    query_embedding, _ = torch.split(query_output, [embedding_dim, embedding_dim], dim=1)

    distances = pairwise_distances(query_embedding, prototypes, distance, S)
    y_pred = (-distances).softmax(dim=1)

    min_dist = torch.min(distances).data.numpy()
    max_prob = torch.max(y_pred).data.numpy()
    prob_label = reverse_class_mapping[str(torch.argmax(y_pred).data.numpy())]
    dist_label = reverse_class_mapping[str(torch.argmin(distances).data.numpy())]

    print('min distance:', min_dist)
    print('max prob:', max_prob)
    print('prob label:', prob_label)

    return [min_dist, max_prob, prob_label, dist_label]


def preprocess_image(query):
    transform = transforms.Compose([
        transforms.Resize((60, 60)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    return torch.unsqueeze(transform(query), 0)


def store_values(values,file_key):

    current_milli_time = lambda: int(round(time.time() * 1000))
    current_time = current_milli_time()

    dynamodb.put_item(
        TableName = os.environ['TableName'],
        Item = {
            "ImageId": {'S': file_key.split('.')[0]},
            "Timestamp": {'N': str(current_time)},
            "MinDist": {'N': str(values[0])},
            "MaxProb": {'N': str(values[1])},
            "ProbLabel": {'S': values[2]},
            "DistLabel": {'S': values[3]}
        }
    )
