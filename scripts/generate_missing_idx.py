import os
import numpy as np
# ls /mnt/nfs_client/calvin/task_D_D/training/ | grep episode | awk -F '_' '{print $2}' | awk -F '.' '{print $1}' > /mnt/nfs_client/calvin/task_D_D/training/id_list.txt
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
    print("Indices min and max from scene info", indices)
    all_numbers = set(list(range(indices[0], indices[1] + 1)))
    missing_numbers = sorted(all_numbers - set(numbers))

    missing_idx_path = scene_info_path.replace('scene_info.npy', 'missing_idx.npy')
    to_save = {'missing_idx': missing_numbers}
    np.save(missing_idx_path, to_save)

    # Display the results
    print(f"Minimum number: {min_num}")
    print(f"Maximum number: {max_num}")
    print(f"Length Missing numbers: {len(missing_numbers)}; Existing numbers: {len(numbers)}")

mode = "validation"
base_path = "/mnt/data1/rutavms/calvin"
# base_path = "/mnt/nfs_client/calvin"
scene_info_location = os.path.join(base_path, f'task_D_D/{mode}/scene_info.npy')
list_path = os.path.join(base_path, f'task_D_D/{mode}/id_list.txt')
# Replace 'D_id_list.txt' with the actual file name
process_numbers(list_path, scene_info_path=scene_info_location)

