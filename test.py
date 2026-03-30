import os

def print_tree(root_dir, indent=""):
    """
    以树形结构打印目录与文件
    """
    for item in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, item)
        print(indent + "├── " + item)

        if os.path.isdir(path):
            print_tree(path, indent + "│   ")

def save_tree(root_dir, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        def _walk(dir_path, indent=""):
            for item in sorted(os.listdir(dir_path)):
                path = os.path.join(dir_path, item)
                f.write(indent + "├── " + item + "\n")

                if os.path.isdir(path):
                    _walk(path, indent + "│   ")

        _walk(root_dir)

root_dir = r"E:\DeepAssist"
print_tree(root_dir)
save_tree(root_dir, "tree_output.txt")