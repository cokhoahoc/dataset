import vart
import xir
import numpy as np
import cv2
import time
import sys


def load_image(image_path, input_shape):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")
    img = cv2.resize(img, (input_shape[2], input_shape[1]))
    img = img.astype(np.float32)
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    img = np.expand_dims(img, axis=0)   # NCHW
    return img


def get_input_output_tensors(runner):
    inputTensors = runner.get_input_tensors()
    outputTensors = runner.get_output_tensors()
    input_shape = tuple(inputTensors[0].dims)
    output_shape = tuple(outputTensors[0].dims)
    return inputTensors, outputTensors, input_shape, output_shape


def xir_dtype_to_numpy(dtype):
    if dtype == "xint8":
        return np.int8
    elif dtype == "float32":
        return np.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def run_fall_detection(xmodel_path, image_path):
    print("[DEBUG] Đang load runner...")
    graph = xir.Graph.deserialize(xmodel_path)
    subgraphs = get_child_subgraph(graph)
    runner = vart.Runner.create_runner(subgraphs[0], "run")

    print("[DEBUG] Lấy input/output tensor...")
    inputTensors, outputTensors, input_shape, output_shape = get_input_output_tensors(runner)

    print("[DEBUG] Tiền xử lý ảnh...")
    start_pre = time.time()
    input_data = load_image(image_path, input_shape)
    end_pre = time.time()
    print(f"[BENCHMARK] Tiền xử lý mất: {end_pre - start_pre:.4f} giây")

    input_dtype = xir_dtype_to_numpy(inputTensors[0].dtype)
    output_dtype = xir_dtype_to_numpy(outputTensors[0].dtype)

    input_data_buffer = [np.empty(input_shape, dtype=input_dtype, order="C")]
    output_data_buffer = [np.empty(output_shape, dtype=output_dtype, order="C")]

    np.copyto(input_data_buffer[0], input_data.reshape(input_shape).astype(input_dtype))

    print("[DEBUG] Chạy inference trên DPU...")
    start_inf = time.time()
    job_id = runner.execute_async(input_data_buffer, output_data_buffer)
    runner.wait(job_id)
    end_inf = time.time()
    print(f"[BENCHMARK] Inference mất: {end_inf - start_inf:.4f} giây")

    print("[DEBUG] Output shape:", output_data_buffer[0].shape)
    print("[DEBUG] Output data:", output_data_buffer[0])

    # Post-processing tùy bạn muốn làm gì tiếp theo
    # Ví dụ: argmax hoặc threshold để phân loại

    return output_data_buffer[0]


def get_child_subgraph(graph):
    root_subgraph = graph.get_root_subgraph()
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    return [s for s in child_subgraphs if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]


if __name__ == "__main__":
    xmodel_file = "fallnet.xmodel"
    test_image = "Fall/test_images/fall-04-cam1-rgb-057.jpg"
    run_fall_detection(xmodel_file, test_image)
