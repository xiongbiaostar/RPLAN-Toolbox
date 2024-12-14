import os
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for rendering
matplotlib.rcParams.update({'agg.path.chunksize': 10000})  # Adjust chunk size if necessary

import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

from rplan.floorplan import Floorplan
from rplan.align import align_fp_gt
from rplan.decorate import get_dw
from rplan.plot import get_figure, get_axes, plot_category, plot_boundary, plot_graph, plot_fp

def plot_RPlan(in_path, file_name, out_path):
    try:
        
        fp = Floorplan(os.path.join(in_path, file_name))
        data = fp.to_dict()

        boxes_aligned, order, room_boundaries, edges = align_fp_gt(
            data['boundary'], data['boxes'], data['types'], data['edges'])
        data['boxes_aligned'] = boxes_aligned
        data['order'] = order
        data['room_boundaries'] = room_boundaries

        doors, windows = get_dw(data)
        data['doors'] = doors
        data['windows'] = windows

        print(f"{file_name} Data size: {len(data['boundary'])}, {len(data['boxes'])}")
        fig = get_figure([512, 512])
        plot_boundary(data['boundary'], ax=get_axes(fig=fig, rect=[0, 0.5, 0.5, 0.5]))
        ax = plot_category(fp.category, ax=get_axes(fig=fig, rect=[0.5, 0.5, 0.5, 0.5]))
        #plot_graph(data['boundary'], data['boxes'], data['types'], data['edges'], ax=ax)
        #plot_fp(data['boundary'], data['boxes_aligned'][order], data['types'][order], ax=get_axes(fig=fig, rect=[0, 0, 0.5, 0.5]))
        #plot_fp(data['boundary'], data['boxes_aligned'][order], data['types'][order], data['doors'], data['windows'], ax=get_axes(fig=fig, rect=[0.5, 0, 0.5, 0.5]))
        fig.canvas.draw()
        fig.canvas.print_figure(os.path.join(out_path, file_name))

        plt.close(fig)
    except Exception as e:
        raise RuntimeError(f"Error processing file {file_name}: {e}")

def get_png_files(folder_path):
    return [file for file in os.listdir(folder_path) if file.endswith('.png')]



def parallel_process(fp_files, in_folder, out_folder, max_workers=4, log_file='failed_files_parallel.log'):
    total_files = len(fp_files)
    success_count = 0
    failed_files = []

    print(f"Total files to process in parallel: {total_files}")

    # 使用 ThreadPoolExecutor 并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(plot_RPlan, in_folder, name, out_folder): name for name in fp_files}

        for future in as_completed(future_to_file):
            file_name = future_to_file[future]
            try:
                future.result()
                success_count += 1
                print(f"Successfully processed: {file_name}")
            except Exception as e:
                print(f"Failed to process {file_name}: {e}")
                failed_files.append(file_name)

    failed_count = len(failed_files)

    # 将失败的文件记录到日志
    if failed_files:
        with open(log_file, 'w') as log:
            log.write("\n".join(failed_files))
        print(f"Failed files have been logged to {log_file}")

    # 打印统计信息
    print("\nParallel Processing Summary:")
    print(f"Total files: {total_files}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed to process: {failed_count}")


def serial_process(fp_files, in_folder, out_folder, log_file='failed_files_serial.log'):
    total_files = len(fp_files)
    success_count = 0
    failed_files = []

    print(f"Total files to process serially: {total_files}")
    #ind_dst =  fp_files.index('21921.png')#15959.png,12978,18941

    for ind, file_name in enumerate(fp_files):
        if file_name != '15959.png' and file_name != '12978.png' and file_name != '18941.png':
            continue
        #if(ind <= ind_dst): continue
        
        try:
            plot_RPlan(in_folder, file_name, out_folder)
            success_count += 1
            print(f"Successfully processed: {file_name}")
        except Exception as e:
            print(f"Failed to process {file_name}: {e}")
            failed_files.append(file_name)

    failed_count = len(failed_files)

    # 将失败的文件记录到日志
    if failed_files:
        with open(log_file, 'w') as log:
            log.write("\n".join(failed_files))
        print(f"Failed files have been logged to {log_file}")

    # 打印统计信息
    print("\nSerial Processing Summary:")
    print(f"Total files: {total_files}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed to process: {failed_count}")


if __name__ == '__main__':
    in_folder = 'F:/RPlane/dataset/floorplan_dataset/'
    out_folder = 'F:/RPlane/dataset/floorplan_plot/'
    fp_files = get_png_files(in_folder)

    # 串行处理
    serial_process(fp_files, in_folder, out_folder)

    # 并行处理
    #parallel_process(fp_files, in_folder, out_folder, max_workers=4)
