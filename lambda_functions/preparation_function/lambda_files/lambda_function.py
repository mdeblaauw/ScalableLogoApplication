import json
from io import BytesIO
import boto3
from zipfile import ZipFile
import os
import shutil
import torch
from PIL import Image
from torchvision import transforms
from resnet_model import ResNet18
from protonet import *

s3 = boto3.client('s3')
embedding_dim = 256
distance = "gaussian"
n_shot = 5

def lambda_handler(event, context):
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        file_key = record['s3']['object']['key']
        print(bucket)
        print(file_key)
    
    downloadAndUnzip(bucket,file_key)
    images_path, reverse_class_mapping, website_image_path = createPaths()
    support_batch = transformToTensor(images_path)
    print(support_batch.shape)
    model = install_model()
    prototypes, S = createPrototypes(model, support_batch, len(reverse_class_mapping))
    saveAndStorePreparationFiles(prototypes, S, reverse_class_mapping, bucket)
    dumpWebsiteImages(website_image_path, reverse_class_mapping)
    
    return {
        'statusCode': 200,
        'body': json.dumps('Preparation files are created and uploaded!')
    }
    
def downloadAndUnzip(bucket, file_key):
    # Download zip and unzip
    old_work_dir = os.getcwd()
    os.chdir('/tmp')
    
    if len(os.listdir('/tmp/')) != 0:
        shutil.rmtree('/tmp/', ignore_errors=True)

    s3.download_file(bucket, file_key, '/tmp/' + file_key)
    
    with ZipFile('/tmp/' + file_key, 'r') as file:
        file.extractall()
    os.chdir(old_work_dir)
    
    print(os.listdir('/tmp/'))
    
    weight_path = "/tmp/model-weights"
    os.mkdir(weight_path)
    content = s3.list_objects_v2(Bucket=bucket, Prefix='model-weights')
    key = content['Contents'][0]['Key']
    print(key)
    print('/tmp/' + key)
    s3.download_file(bucket, key, '/tmp/' + key)

def createPaths():
    images = []
    class_mapping = {}
    class_number = 0
    for root, folders, files in os.walk('/tmp/support_set/'):
        if len(files) == 0:
            continue
    
        class_name = root.split('/')[-1]
    
        if class_name != class_mapping:
            class_mapping[class_name] = class_number
            class_number += 1
    
        for f in files:
            images.append({
                'class_name': class_name,
                'filepath': os.path.join(root,f)})
                
    reverse_class_mapping =  {value: key for key, value in class_mapping.items()}
    
    website_image_path = []
    for i in reverse_class_mapping:
        for j in images:
            if j['class_name'] == reverse_class_mapping[i]:
                website_image_path.append(j['filepath'])
                break
    print(website_image_path)
    
    images_path = []
    for i in range(class_number):
        for j in images:
            if j['class_name'] == reverse_class_mapping[i]:
                images_path.append(j['filepath'])

    return images_path, reverse_class_mapping, website_image_path
    
def transformToTensor(images_path):
    pil_images = []
    for i in images_path:
        pil_images.append(Image.open(i).convert('RGB'))
        
    transform = transforms.Compose([
                transforms.Resize((60,60)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])
            
    return torch.stack([transform(i) for i in pil_images])
    
def install_model():
    model_name = os.listdir("/tmp/model-weights")[0]
    model = ResNet18(flatten=True).to("cpu", dtype=torch.float)
    model.load_state_dict(torch.load(os.path.join("/tmp/model-weights", model_name), map_location="cpu"))
    model.eval()
    return model
    
def createPrototypes(model, support_batch, k_way):
    embeddings = model(support_batch)
    support, raw_covariance_matrix = torch.split(embeddings, [embedding_dim, embedding_dim], dim=1)
    inv_covariance_matrix = calculate_inverse_covariance_matrix(raw_covariance_matrix, 1.0)
    S = compute_matrix(inv_covariance_matrix, k_way, n_shot, embedding_dim)
    prototypes = compute_prototypes(support,k_way, n_shot)
    return prototypes, S
    
def saveAndStorePreparationFiles(prototypes, S, reverse_class_mapping, bucket):
    #os.chdir('/tmp')
    torch.save(prototypes, '/tmp/prototypes.pt')
    torch.save(S, '/tmp/S.pt')
    with open('/tmp/reverse_class_mapping.json', 'w') as outfile:
        json.dump(reverse_class_mapping, outfile)
        
    putInBucket(bucket)
    
def putInBucket(bucket):
    s3.upload_file('/tmp/prototypes.pt', bucket, 'preparation-files/prototypes.pt')
    s3.upload_file('/tmp/S.pt', bucket, 'preparation-files/S.pt')
    s3.upload_file('/tmp/reverse_class_mapping.json', bucket, 'preparation-files/reverse_class_mapping.json')
    
def dumpWebsiteImages(website_image_path, reverse_class_mapping):
    file = s3.list_objects_v2(Bucket=os.environ['website_bucket'], Prefix='preparation-images')
    if file['KeyCount'] > 0:
        contents = file['Contents']
        print(contents)
        for i in contents:
            s3.delete_object(Bucket=os.environ['website_bucket'],Key=i['Key'])
            
    for i, path in enumerate(website_image_path):
        s3.upload_file(path, os.environ['website_bucket'], 'preparation-images/' + reverse_class_mapping[i] + '.' + path.split('.')[-1])