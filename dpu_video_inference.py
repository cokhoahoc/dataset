import vart
import xir
import numpy as np
import cv2
import time
import imageio


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


def get_child_subgraph(graph):
    root_subgraph = graph.get_root_subgraph()
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    return [s for s in child_subgraphs if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]


def preprocess_frame(frame, input_shape):
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (input_shape[2], input_shape[1]), interpolation=cv2.INTER_LINEAR)
    img = np.expand_dims(img, axis=0)
    return img


def run_video_inference(xmodel_path, video_input_path, video_output_path):
    print("[INFO] Loading runner...")
    graph = xir.Graph.deserialize(xmodel_path)
    subgraphs = get_child_subgraph(graph)
    runner = vart.Runner.create_runner(subgraphs[0], "run")

    inputTensors, outputTensors, input_shape, output_shape = get_input_output_tensors(runner)
    input_dtype = xir_dtype_to_numpy(inputTensors[0].dtype)
    output_dtype = xir_dtype_to_numpy(outputTensors[0].dtype)

    input_data_buffer = [np.empty(input_shape, dtype=input_dtype, order="C")]
    output_data_buffer = [np.empty(output_shape, dtype=output_dtype, order="C")]

    reader = imageio.get_reader(video_input_path)
    fps = reader.get_meta_data()['fps']
    writer = imageio.get_writer(video_output_path, fps=fps)

    frame_id = 0
    for frame in reader:
        # imageio uses RGB by default
        input_data = preprocess_frame(frame, input_shape)
        np.copyto(input_data_buffer[0], input_data.reshape(input_shape).astype(input_dtype))

        job_id = runner.execute_async(input_data_buffer, output_data_buffer)
        runner.wait(job_id)

        output_q = output_data_buffer[0].flatten()[0]
        output_real = (output_q - 0) * (1.0 / 127.0)
        label = 1 if output_real >= 0.5 else 0

        # Vẽ nhãn lên frame
        text = f"{'Nga' if label == 1 else 'Khong nga'} ({output_real:.2f})"
        color = (255, 0, 0) if label == 1 else (0, 255, 0)  # RGB
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(frame_bgr, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        writer.append_data(frame_rgb)
        print(f"[INFO] Frame {frame_id}: {text}")
        frame_id += 1

    reader.close()
    writer.close()
    print("[INFO] Video output đã lưu tại:", video_output_path)


if __name__ == "__main__":
    xmodel_file = "fallet.xmodel"
    video_input = "Fall/test_videos/fall_video.mp4"
    video_output = "output_result.mp4"
    run_video_inference(xmodel_file, video_input, video_output)
