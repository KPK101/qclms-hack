import subprocess
import torch
import numpy as np

onnx_path = '/.openai_clip-clipimageencoder.onnx'
dlc_path = './test.dlc'

#create data:
input_file = "testdata.raw"

def creteadata(input_file = input_file):
    data = np.random.random((10, 224, 224)).astype(np.float32)
    
    
    data.flatten().tofile(input_file)
    

# step1 : convert model to .dlc

def onnx_to_dlc(onnx_path, dlc_path):
    command = [
        "snpe-onnx-to-dlc",
        "--input_network", onnx_path,
        "--output_path", dlc_path
    ]
    subprocess.run(command, check=True)


# step2 : push dlc to device
def push_to_device(local_path, remote_path):
    command = ["adb", "push", local_path, remote_path]
    subprocess.run(command, check=True)


def run_device_inference(model_path, input_list, output_dir):
    command = [
        "adb", "shell",
        "snpe-net-run",
        "--container", model_path,
        "--input_list", input_list,
        "-use_gpu",
        "--output_dir", output_dir
    ]
    subprocess.run(command, check=True)


def load_raw_output(output_path):
    return np.fromfile(output_path, dtype=np.float32)
    
remote_data_path = './temp/testdata.raw'
remote_model_path = './temp/test.dlc'
remote_list_path = './temp/input_list.txt'
remote_output_path = './temp/output/'

onnx_to_dlc(onnx_path, dlc_path)
push_to_device(dlc_path, remote_model_path)
push_to_device(input_file, remote_data_path)


input_list_file = "input_list_file.txt"

with open(input_list_file, "w") as f:
    f.write(remote_data_path+"\n")

push_to_device(input_list_file, remote_list_path)

run_device_inference(remote_model_path, remote_list_path, remote_output_path)

output_file = "./output_0.raw"
adb_pull_command = ["adb", "pull", f"{remote_output_path+"output_0.raw"}"]

subprocess.pull(adb_pull_command, check=True)

output_data = load_raw_output(output_file)
print(output_data)