import vart
import xir
import numpy as np
import cv2
import time
import imageio
import csv


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

    # Biến theo dõi hiệu năng
    frame_id = 0
    inf_times = []
    full_times = []
    perf_log = []

    total_start_time = time.time()

    for frame in reader:
        frame_start = time.time()

        # imageio uses RGB by default
        input_data = preprocess_frame(frame, input_shape)
        np.copyto(input_data_buffer[0], input_data.reshape(input_shape).astype(input_dtype))

        inf_start = time.time()
        job_id = runner.execute_async(input_data_buffer, output_data_buffer)
        runner.wait(job_id)
        inf_end = time.time()

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

        inf_time = (inf_end - inf_start) * 1000  # milliseconds
        frame_time = (time.time() - frame_start) * 1000
        inf_times.append(inf_time)
        full_times.append(frame_time)
        perf_log.append([frame_id, inf_time, frame_time, output_real, label])

        print(f"[INFO] Frame {frame_id}: {text} | Inference: {inf_time:.2f} ms | Total: {frame_time:.2f} ms")
        frame_id += 1

    reader.close()
    writer.close()

    total_time = time.time() - total_start_time
    avg_fps = frame_id / total_time

    print("\n====== Inference Performance Summary ======")
    print(f"Total frames processed: {frame_id}")
    print(f"Total elapsed time: {total_time:.2f} sec")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average Inference Time: {np.mean(inf_times):.2f} ms")
    print(f"Min Inference Time: {np.min(inf_times):.2f} ms")
    print(f"Max Inference Time: {np.max(inf_times):.2f} ms")
    print("[INFO] Video output đã lưu tại:", video_output_path)

    # (Tùy chọn) Ghi log ra CSV
    save_log = True
    if save_log:
        with open("performance_log.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["FrameID", "InferenceTime_ms", "FullFrameTime_ms", "Score", "Label"])
            writer.writerows(perf_log)
        print("[INFO] Đã ghi log hiệu năng vào performance_log.csv")


if __name__ == "__main__":
    xmodel_file = "fallet.xmodel"
    video_input = "Fall/test_videos/fall_video.mp4"
    video_output = "output_result.mp4"
    run_video_inference(xmodel_file, video_input, video_output)
