
import json
import whisper
import librosa
import io
import boto3
def read_aws_json(file_key):

    # S3存储桶名称和JSON文件路径
    bucket_name = 'handata'

    s3_client = boto3.client('s3')

    # 从S3下载JSON文件内容
    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    content = response['Body'].read().decode('utf-8')

    # 解析JSON内容
    data = json.loads(content)
    return data
def read_aws_wav(file_key):
    bucket_name = 'handata'

    # 创建S3客户端
    s3_client = boto3.client('s3')

    # 从S3下载WAV文件内容
    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    wav_data = response['Body'].read()
    wav_file = io.BytesIO(wav_data)
    y, sr = librosa.load(wav_file, sr=16000)
    return y

def read_local_json(file_path):
    with open(file_path,'r') as f:
        return json.load(f)

def read_local_wav(file_path):
    y,r = librosa.load(file_path, sr=16000)
    return y
    
def print_trainable_para(model):
    total_params = []
    trainable_params = []
    for name, param in model.named_parameters():
        
        if param.requires_grad:
            trainable_params.append((name, param.numel()))
        total_params.append((name,param.numel()))
        
    total_params_ = 0
    trainable_params_ = 0
    for name, num_params in total_params:
        print(f"Parameter name: {name}, Number of parameters: {num_params}")
        total_params_ += num_params
    for name, num_params in trainable_params:
        print(f"Parameter name: {name}, Number of parameters: {num_params}")
        trainable_params_ += num_params
        
    print(f"Total parameters: {total_params_}")
    print(f"Trainable parameters: {trainable_params_}")
