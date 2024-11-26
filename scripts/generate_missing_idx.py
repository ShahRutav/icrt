import numpy as np
def process_numbers(file_name, scene_info_path):
    # Read numbers from the file
    with open(file_name, 'r') as file:
        numbers = [int(line.strip()) for line in file if line.strip().isdigit()]

    # Sort the numbers
    numbers.sort()

    # Calculate minimum, maximum, and missing numbers
    min_num = numbers[0]
    max_num = numbers[-1]
    all_numbers = set(range(min_num, max_num + 1))
    indices = next(iter(np.load(scene_info_path, allow_pickle=True).item().values()))
    all_numbers = set(list(range(indices[0], indices[1] + 1)))
    missing_numbers = sorted(all_numbers - set(numbers))

    missing_idx_path = scene_info_path.replace('scene_info.npy', 'missing_idx.npy')
    to_save = {'missing_idx': missing_numbers}
    np.save(missing_idx_path, to_save)

    # Display the results
    print(f"Minimum number: {min_num}")
    print(f"Maximum number: {max_num}")
    print(f"Length Missing numbers: {len(missing_numbers)}")

scene_info_location = "/mnt/nfs_client/calvin/task_D_D/training/scene_info.npy"
# Replace 'D_id_list.txt' with the actual file name
process_numbers('D_id_list.txt', scene_info_path=scene_info_location)

