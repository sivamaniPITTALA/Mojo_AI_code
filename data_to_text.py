import os 
file_paths = [
    "mojo_Dataset\\examples\\check_mod.py",
    "mojo_Dataset\\examples\\deviceinfo.mojo",
    "mojo_Dataset\\examples\\hello.🔥",
    "mojo_Dataset\\examples\\hello_interop.mojo",
    "mojo_Dataset\\examples\\mandelbrot.mojo",
    "mojo_Dataset\\examples\\matmul.mojo",
    "mojo_Dataset\\examples\\memset.mojo",
    "mojo_Dataset\\examples\\nbody.mojo",
    "mojo_Dataset\\examples\\pymatmul.py",
    "mojo_Dataset\\examples\\README.md",
    "mojo_Dataset\\examples\\reduce.mojo",
    "mojo_Dataset\\examples\\simple_interop.py",
    "mojo_Dataset\\examples\\blogs-videos\\whats_new_v0.5.ipynb",
    "mojo_Dataset\\examples\\blogs-videos\\mojo-plotter\\environment-macos-arm.yaml",
    "mojo_Dataset\\examples\\blogs-videos\\mojo-plotter\\environment.yaml",
    "mojo_Dataset\\examples\\blogs-videos\\mojo-plotter\\main.mojo",
    "mojo_Dataset\\examples\\blogs-videos\\mojo-plotter\\README.md",
    "mojo_Dataset\\examples\\blogs-videos\\tensorutils\\tensorutils.mojo",
    "mojo_Dataset\\examples\\blogs-videos\\tensorutils\\__init__.mojo",
    "mojo_Dataset\\examples\\docker\\build-image.sh",
    "mojo_Dataset\\examples\\docker\\docker-compose.yml",
    "mojo_Dataset\\examples\\docker\\Dockerfile.mojosdk",
    "mojo_Dataset\\examples\\notebooks\\BoolMLIR.ipynb",
    "mojo_Dataset\\examples\\notebooks\\HelloMojo.ipynb",
    "mojo_Dataset\\examples\\notebooks\\Mandelbrot.ipynb",
    "mojo_Dataset\\examples\\notebooks\\Matmul.ipynb",
    "mojo_Dataset\\examples\\notebooks\\Memset.ipynb",
    "mojo_Dataset\\examples\\notebooks\\programming-manual.ipynb",
    "mojo_Dataset\\examples\\notebooks\\RayTracing.ipynb",
    "mojo_Dataset\\examples\\notebooks\\README.md",
    "mojo_Dataset\\examples\\notebooks\\images\\background.png",
    "mojo_Dataset\\proposals\\lifetimes-and-provenance.md",
    "mojo_Dataset\\proposals\\lifetimes-keyword-renaming.md",
    "mojo_Dataset\\proposals\\mojo-and-dynamism.md",
    "mojo_Dataset\\proposals\\README.md",
    "mojo_Dataset\\proposals\\value-ownership.md",
    "mojo_Dataset\\User\\data",
    "mojo_Dataset\\User\\data.txt",
    "mojo_Dataset\\workshops\\mojo_for_python_developers\\1_mojo_language_basics.ipynb",
    "mojo_Dataset\\workshops\\mojo_for_python_developers\\2_speeding_up_python.ipynb",
    "mojo_Dataset\\workshops\\mojo_for_python_developers\\3_parallelization_speedup_tensor_mean.ipynb",
    "mojo_Dataset\\workshops\\mojo_for_python_developers\\4_writing_custom_structs.ipynb",
    "mojo_Dataset\\workshops\\mojo_for_python_developers\\tensorprint.mojo",
    "mojo_Dataset\\workshops\\mojo_for_python_developers\\solutions\\1_mojo_language_basics.ipynb",
    "mojo_Dataset\\workshops\\mojo_for_python_developers\\solutions\\2_speeding_up_python.ipynb",
    "mojo_Dataset\\workshops\\mojo_for_python_developers\\solutions\\3_parallelization_speedup_tensor_mean.ipynb",
    "mojo_Dataset\\workshops\\mojo_for_python_developers\\solutions\\4_writing_custom_structs.ipynb"
    "mojo_Dataset\\content_of_ds.pkl",
]

output_file_path = 'ChatGPT\\mojo_data.txt'

with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for path in file_paths:
        # Check if the path exists and is a file
        if os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as file_to_read:
                content = file_to_read.read()
                output_file.write(f"File: {path}\n")
                output_file.write(content + '\n\n')
        else:
            output_file.write(f"File: {path} does not exist or is not a file.\n\n")

print(f"File contents written to '{output_file_path}'")